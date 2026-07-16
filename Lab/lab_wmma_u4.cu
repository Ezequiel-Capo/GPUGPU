#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <nvtx3/nvToolsExt.h>

using namespace nvcuda;

// Dimensiones obligatorias para usar la API WMMA en u4: M=8, N=8, K=32
#define WMMA_M 8
#define WMMA_N 8
#define WMMA_K 32
#define WARP_SIZE 32
#define WARPS_PER_BLOCK_MAX (4 * 4)
#define THREADS_PER_BLOCK_MAX (WARPS_PER_BLOCK_MAX * WARP_SIZE)

// K requiere 32 elementos lógicos, que equivalen a 16 bytes físicos
#define SHMEM_K_BYTES 16

// Padding para evitar Bank Conflicts
#define PAD_K_BYTES 16
#define SHMEM_STRIDE_BYTES (SHMEM_K_BYTES + PAD_K_BYTES)
#define SHMEM_STRIDE_ELEMENTS (SHMEM_STRIDE_BYTES * 2) 

#define SHMEM_C_N (TILE_N + 8)

#define CUDA_CHK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while (0)

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}



static inline size_t div_up_size(size_t a, size_t b) {
    return (a + b - 1) / b;
}

// Función para obtener el índice 1D del triángulo superior
static __device__ __host__ inline size_t get_packed_index(size_t r, size_t c, size_t m) {
    if (r > c) {
        size_t tmp = r;
        r = c;
        c = tmp;
    }
    return r * m - (r * (r + 1)) / 2 + c;
}

// EUCLIDEAN DISTANCE CON RAIZ CUADRADA, FLOAT16 Y UNSIGNED INT
__global__ void CalculateDistance(const uint32_t *XXT, const uint32_t *norms, float *distances, size_t m) {
    size_t row = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < m && row <= col) {
        size_t idx = get_packed_index(row, col, m);
        uint32_t xxt = XXT[idx];
        uint32_t norm_row = norms[row];
        uint32_t norm_col = norms[col];

        uint32_t sum_norms = norm_row + norm_col;
        float dist_f = 0.0f;
        
        // Prevenir negativos (underflow) y calcular raíz cuadrada
        if (sum_norms > 2 * xxt) {
            dist_f = sqrtf((float)(sum_norms - 2 * xxt));
        }

        // Guardar resultado como float
        distances[idx] = dist_f;
    }
}

template <int WARP_TILE_M, int WARP_TILE_N>
__global__ void XXT_WMMA_Shared(const uint8_t *X, uint32_t *C, size_t m, size_t n) {
    constexpr int TILE_M = WARP_TILE_M * WMMA_M;
    constexpr int TILE_N = WARP_TILE_N * WMMA_N;
    constexpr int WARPS_PER_BLOCK = WARP_TILE_M * WARP_TILE_N;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

    static_assert(WARP_TILE_M >= 1 && WARP_TILE_N >= 1, "WARP_TILE_M y WARP_TILE_N deben ser al menos 1.");
    static_assert(WARP_TILE_M <= 4 && WARP_TILE_N <= 4, "u4 solo soporta configuraciones 2x2 o 4x4.");
    static_assert(THREADS_PER_BLOCK <= 1024, "Demasiados warps por bloque.");

    size_t row_base = (size_t)blockIdx.y * TILE_M;
    size_t col_base = (size_t)blockIdx.x * TILE_N;

    if (col_base + TILE_N <= row_base) return;

    int warp_id = threadIdx.x / WARP_SIZE;
    size_t warp_tile_row = (size_t)warp_id / WARP_TILE_N;
    size_t warp_tile_col = (size_t)warp_id % WARP_TILE_N;

    size_t subtile_row = row_base + warp_tile_row * WMMA_M;
    size_t subtile_col = col_base + warp_tile_col * WMMA_N;

    bool compute_subtile = (subtile_col + WMMA_N > subtile_row);
    size_t n_bytes = (size_t)n / 2;

    // Memoria compartida con PADDING aplicado en la dimensión de los bytes (stride)
    __shared__ __align__(32) uint8_t a_tile[TILE_M][SHMEM_STRIDE_BYTES];
    __shared__ __align__(32) uint8_t b_tile[TILE_N][SHMEM_STRIDE_BYTES];
    __shared__ __align__(32) int32_t c_tile[TILE_M][SHMEM_C_N];

    // Uso del namespace experimental para precisión u4 (unsigned 4-bit)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::experimental::precision::u4, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::experimental::precision::u4, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag; // Obligatorio int32_t por API

    wmma::fill_fragment(c_frag, 0);

    // k0 avanza de a WMMA_K (64 elementos lógicos por iteración = 32 bytes)
    for (size_t k0 = 0; k0 < n; k0 += WMMA_K) {
        size_t k0_bytes = k0 / 2;
        
        // Carga cooperativa de A
        for (int idx = threadIdx.x; idx < TILE_M * SHMEM_K_BYTES; idx += blockDim.x) {
            int local_row = idx / SHMEM_K_BYTES;
            int local_k_byte = idx % SHMEM_K_BYTES;
            size_t global_k_byte = k0_bytes + (size_t)local_k_byte;
            size_t a_row = row_base + (size_t)local_row;

            a_tile[local_row][local_k_byte] = (a_row < m && global_k_byte < n_bytes) ? X[(size_t)a_row * n_bytes + global_k_byte] : 0;
        }

        // Carga cooperativa de B
        for (int idx = threadIdx.x; idx < TILE_N * SHMEM_K_BYTES; idx += blockDim.x) {
            int local_row = idx / SHMEM_K_BYTES;
            int local_k_byte = idx % SHMEM_K_BYTES;
            size_t global_k_byte = k0_bytes + (size_t)local_k_byte;
            size_t b_row = col_base + (size_t)local_row;

            b_tile[local_row][local_k_byte] = (b_row < m && global_k_byte < n_bytes) ? X[(size_t)b_row * n_bytes + global_k_byte] : 0;
        }

        __syncthreads();

        if (compute_subtile) {
            const uint8_t *a_ptr = &a_tile[warp_tile_row * WMMA_M][0];
            const uint8_t *b_ptr = &b_tile[warp_tile_col * WMMA_N][0];

            wmma::load_matrix_sync(a_frag, a_ptr, SHMEM_STRIDE_ELEMENTS);
            wmma::load_matrix_sync(b_frag, b_ptr, SHMEM_STRIDE_ELEMENTS);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    if (compute_subtile) {
        size_t local_row = warp_tile_row * WMMA_M;
        size_t local_col = warp_tile_col * WMMA_N;
        wmma::store_matrix_sync(&c_tile[local_row][local_col], c_frag, SHMEM_C_N, wmma::mem_row_major);
    }

    __syncthreads();

    // Guardado de datos en el vector empaquetado (Triangulo superior) y Casteo a uint32_t
    for (int idx = threadIdx.x; idx < TILE_M * TILE_N; idx += blockDim.x) {
        size_t local_row = (size_t)idx / TILE_N;
        size_t local_col = (size_t)idx % TILE_N;
        size_t global_row = row_base + local_row;
        size_t global_col = col_base + local_col;

        if (global_row < m && global_col < m && global_row <= global_col) {
            size_t packed_idx = get_packed_index(global_row, global_col, m);
            C[packed_idx] = (uint32_t)c_tile[local_row][local_col]; 
        }
    }
}

template <int WTM, int WTN>
void launch_XXT(const uint8_t* d_matrix, uint32_t* d_XXT, size_t m, size_t n) {
    constexpr int TILE_M = WTM * WMMA_M;
    constexpr int TILE_N = WTN * WMMA_N;
    constexpr int THREADS_PER_BLOCK = WTM * WTN * WARP_SIZE;

    dim3 block_syrk(THREADS_PER_BLOCK);
    dim3 grid_syrk((unsigned)div_up_size(m, (size_t)TILE_N), (unsigned)div_up_size(m, (size_t)TILE_M));
    XXT_WMMA_Shared<WTM, WTN><<<grid_syrk, block_syrk>>>(d_matrix, d_XXT, m, n);
}

void dispatch_XXT(const uint8_t* d_matrix, uint32_t* d_XXT, size_t m, size_t n, int wtm, int wtn) {
    if (wtm == 2 && wtn == 2) {
        launch_XXT<2, 2>(d_matrix, d_XXT, m, n);
    } else if (wtm == 4 && wtn == 4) {
        launch_XXT<4, 4>(d_matrix, d_XXT, m, n);
    } else {
        fprintf(stderr, "Error: Configuracion WMMA u4 no soportada (%dx%d). Use 2x2 o 4x4.\n", wtm, wtn);
        exit(1);
    }
}

__inline__ __device__ uint32_t warpReduceSum(uint32_t val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Cálculo de normas adaptado a uint32_t
__global__ void CalculateNormVector(const uint8_t *matrix, uint32_t *norms, size_t m, size_t n) {
    size_t row = blockIdx.x;
    if (row >= m) return;

    uint32_t local_norm = 0;
    size_t n_bytes = (size_t)n / 2;
    size_t row_offset = (size_t)row * n_bytes;

    for (size_t col_byte = (size_t)threadIdx.x; col_byte < n_bytes; col_byte += (size_t)blockDim.x) {
        uint8_t byte = matrix[row_offset + col_byte];
        
        uint32_t v0 = byte & 0x0F;
        uint32_t v1 = (byte >> 4) & 0x0F;
        
        local_norm += (v0 * v0) + (v1 * v1);
    }

    local_norm = warpReduceSum(local_norm);

    __shared__ uint32_t warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = (blockDim.x + warpSize - 1) / warpSize;

    if (lane == 0) warp_sums[warp_id] = local_norm;

    __syncthreads();

    if (warp_id == 0) {
        uint32_t sum = (lane < warps_per_block) ? warp_sums[lane] : 0;
        sum = warpReduceSum(sum);
        if (lane == 0) norms[row] = sum;
    }
}

__device__ static inline uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__global__ void GenerateGenomicMatrixPackedKernelU4(uint8_t *matrix, int m, int n_bytes, uint32_t seed) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int byte_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || byte_col >= n_bytes) return;

    int col0 = byte_col * 2;
    int col1 = col0 + 1;

    uint32_t k0 = seed ^ ((uint32_t)row * 0x9e3779b9u) ^ ((uint32_t)col0 * 0x85ebca6bu);
    uint32_t k1 = seed ^ ((uint32_t)row * 0xc2b2ae35u) ^ ((uint32_t)col1 * 0x27d4eb2fu);

    uint8_t v0 = (uint8_t)(hash_u32(k0) % 3u);
    uint8_t v1 = (uint8_t)(hash_u32(k1) % 3u);

    matrix[(size_t)row * (size_t)n_bytes + (size_t)byte_col] = (uint8_t)((v0 & 0x0F) | ((v1 & 0x0F) << 4));
}

void generate_genomic_matrix_packed_device(uint8_t *d_matrix, size_t m, size_t n) {
    size_t n_bytes = n / 2;
    dim3 block(32, 8);
    dim3 grid((unsigned)div_up_size(n_bytes, (size_t)block.x), (unsigned)div_up_size(m, (size_t)block.y));

    GenerateGenomicMatrixPackedKernelU4<<<grid, block>>>(d_matrix, m, n_bytes, 42u);
    CUDA_CHK(cudaGetLastError());
}

// Imprime la matriz desempaquetandola en tiempo de ejecucion
void print_matrix_packed_u8(const uint8_t *matrix, int rows, int cols) {
    int cols_bytes = cols / 2;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols_bytes; j++) {
            uint8_t byte = matrix[i * cols_bytes + j];
            printf("%u %u ", byte & 0x0F, (byte >> 4) & 0x0F);
        }
        printf("\n");
    }
}

// Validacion CPU adaptada para unsigned
bool validate_small_case(const uint8_t *X, const uint32_t *XXT, const uint32_t *norms, size_t m, size_t n) {
    size_t n_bytes = n / 2;
    for (size_t i = 0; i < m; i++) {
        uint32_t ref_norm = 0;
        for (size_t k = 0; k < n; k++) {
            size_t byte_idx = k / 2;
            size_t shift = (k % 2) * 4;
            uint32_t v = (X[i * n_bytes + byte_idx] >> shift) & 0x0F;
            ref_norm += v * v;
        }
        if (norms[i] != ref_norm) {
            printf("Error norma fila %d: GPU=%u CPU=%u\n", i, norms[i], ref_norm);
            return false;
        }

        for (size_t j = i; j < m; j++) {
            uint32_t ref = 0;
            for (size_t k = 0; k < n; k++) {
                size_t byte_idx = k / 2;
                size_t shift = (k % 2) * 4;
                uint32_t vi = (X[i * n_bytes + byte_idx] >> shift) & 0x0F;
                uint32_t vj = (X[j * n_bytes + byte_idx] >> shift) & 0x0F;
                ref += vi * vj;
            }
            
            size_t packed_idx = get_packed_index(i, j, m);
            if (XXT[packed_idx] != ref) {
                printf("Error XXT(%d,%d): GPU=%u CPU=%u\n", i, j, XXT[packed_idx], ref);
                return false;
            }
        }
    }
    return true;
}

void print_matrix_packed_u32(const uint32_t *matrix, int m) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            size_t idx = get_packed_index(i, j, m);
            printf("%u ", matrix[idx]);
        }
        printf("\n");
    }
}

// Imprime reconstruyendo la matriz cuadrada desde el vector de distancias
void print_matrix_packed_float(const float *matrix, int m) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            size_t idx = get_packed_index(i, j, m);
            printf("%6.2f ", matrix[idx]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    size_t m = (size_t)(1 << 10);  // individuos
    size_t n = (size_t)(1 << 15);  // SNPs (Debe ser multiplo de 32 para WMMA_K)
    int warp_m = 4; //performante por defecto
    int warp_n = 4;
    printf("Memoria usada: %.2f MB\n", (float)(m * n * 0.5f) / (1024.0f * 1024.0f));

    if (argc >= 5) {
        m = (size_t)strtoull(argv[1], NULL, 10);
        n = (size_t)strtoull(argv[2], NULL, 10);
        warp_m = atoi(argv[3]);
        warp_n = atoi(argv[4]);
    } else if (argc >= 3) {
        m = (size_t)strtoull(argv[1], NULL, 10);
        n = (size_t)strtoull(argv[2], NULL, 10);
    }

    if (m == 0 || n == 0 || n % 32 != 0) {
        fprintf(stderr, "Uso: %s [individuos m] [SNPs n (multiplo de 32)] [warp_m] [warp_n]\n", argv[0]);
        return 1;
    }

    if (!((warp_m == 2 && warp_n == 2) || (warp_m == 4 && warp_n == 4))) {
        fprintf(stderr, "Error: u4 solo soporta warp_m/warp_n iguales a 2x2 o 4x4.\n");
        return 1;
    }

    printf("Iniciando con configuracion WMMA u4: Warp %dx%d | Size %dx%dx%d\n", warp_m, warp_n, WMMA_M, WMMA_N, WMMA_K);

    // 1. Memoria de la matriz de entrada (Reduccion 4-bits)
    size_t matrix_elems = (size_t)m * (size_t)n;
    size_t matrix_bytes = matrix_elems * sizeof(uint8_t) / 2; 

    // 2. Memoria de la matriz de salida (Reduccion Triangulo Superior)
    size_t tri_elems = m * (m + 1) / 2;
    size_t tri_u32_bytes = tri_elems * sizeof(uint32_t);
    size_t tri_float_bytes = tri_elems * sizeof(float); // Bytes para float

    uint8_t *h_matrix = NULL;

    uint8_t *d_matrix = NULL;
    uint32_t *d_norms = NULL;
    uint32_t *d_XXT = NULL;
    float *d_distances = NULL;

    CUDA_CHK(cudaMalloc(&d_matrix, matrix_bytes));
    CUDA_CHK(cudaMalloc(&d_norms, m * sizeof(uint32_t)));
    CUDA_CHK(cudaMalloc(&d_XXT, tri_u32_bytes));
    CUDA_CHK(cudaMalloc(&d_distances, tri_float_bytes));

    printf("Generando matriz genomica empaquetada X_int4 (%zu individuos x %zu SNPs)...\n", m, n);

    nvtxRangePushA("GenX");
    generate_genomic_matrix_packed_device(d_matrix, m, n);
    CUDA_CHK(cudaDeviceSynchronize());
    nvtxRangePop();

    bool need_host_matrix = (m <= 64 && n <= 1024);
    if (need_host_matrix) {
        h_matrix = (uint8_t*)malloc(matrix_bytes);
        if (!h_matrix) {
            fprintf(stderr, "No hay memoria de host suficiente para depuracion/validacion.\n");
            cudaFree(d_matrix);
            cudaFree(d_norms);
            cudaFree(d_XXT);
            cudaFree(d_distances);
            return 1;
        }
        CUDA_CHK(cudaMemcpy(h_matrix, d_matrix, matrix_bytes, cudaMemcpyDeviceToHost));
    }

    if (m <= 64 && n <= 128 && h_matrix) {
        printf("\nX (debug):\n");
        print_matrix_packed_u8(h_matrix, m, n);
    }

    // NORMAS -----------------------------------------------------
    dim3 block_norms(256);
    dim3 grid_norms((unsigned)m);
    CalculateNormVector<<<grid_norms, block_norms>>>(d_matrix, d_norms, m, n);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    printf("Normas finalizado.\n");
    CUDA_CHK(cudaMemset(d_XXT, 0, tri_u32_bytes));

    // X*X^T -----------------------------------------------------
    dispatch_XXT(d_matrix, d_XXT, m, n, warp_m, warp_n);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaMemset(d_distances, 0, tri_float_bytes));
    printf("XXT con tensor cores finalizado\n");

    // DISTANCIAS EUCLIDEAS -----------------------------------------------------
    dim3 block_dist(16, 16);
    dim3 grid_dist((unsigned)div_up_size(m, (size_t)block_dist.x), (unsigned)div_up_size(m, (size_t)block_dist.y));
    CalculateDistance<<<grid_dist, block_dist>>>(d_XXT, d_norms, d_distances, m);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    // Loop de perfilado NVTX
    for (int i = 0; i < 10; i++) {
        

        // NORMAS
        nvtxRangePushA("Norms");
        CalculateNormVector<<<grid_norms, block_norms>>>(d_matrix, d_norms, m, n);
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();

        CUDA_CHK(cudaMemset(d_XXT, 0, tri_u32_bytes));

        // XXT WMMA int4
        nvtxRangePushA("XXT");
        dispatch_XXT(d_matrix, d_XXT, m, n, warp_m, warp_n);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();

        CUDA_CHK(cudaMemset(d_distances, 0, tri_float_bytes));

        // DISTANCIAS
        nvtxRangePushA("CalculateDistance");
        CalculateDistance<<<grid_dist, block_dist>>>(d_XXT, d_norms, d_distances, m);
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop(); 
    }

    // DEBUGS para validacion en casos pequeños
    if (m <= 64 && n <= 1024 && h_matrix) {
        uint32_t *h_norms = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
        uint32_t *h_XXT = (uint32_t*)malloc(tri_u32_bytes);
        
        CUDA_CHK(cudaMemcpy(h_norms, d_norms, (size_t)m * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHK(cudaMemcpy(h_XXT, d_XXT, tri_u32_bytes, cudaMemcpyDeviceToHost));

        printf("Normas (debug):\n");
        for (int i = 0; i < m; i++) printf("%u ", h_norms[i]);
        printf("\n");

        bool ok = validate_small_case(h_matrix, h_XXT, h_norms, m, n);
        printf("Validacion CPU/GPU: %s\n", ok ? "OK" : "FALLO");

        float *h_distances = (float*)malloc(tri_float_bytes);
        fprintf(stderr, "\nDistancias euclideas (debug):\n");
        CUDA_CHK(cudaMemcpy(h_distances, d_distances, tri_float_bytes, cudaMemcpyDeviceToHost));
        
        // Usamos la nueva funcion de impresion adaptada para float
        print_matrix_packed_float(h_distances, m);

        free(h_norms);
        free(h_XXT);
        free(h_distances);
    }

    cudaFree(d_matrix);
    cudaFree(d_norms);
    cudaFree(d_XXT);
    cudaFree(d_distances);
    if (h_matrix) free(h_matrix);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <nvtx3/nvToolsExt.h>

using namespace nvcuda;

// Dimensiones obligatorias para usar la API WMMA en u4: M=8, N=8, K=32
#define WARP_TILE_M 2
#define WARP_TILE_N 2
#define WMMA_M 8
#define WMMA_N 8
#define WMMA_K 32
#define WARP_SIZE 32
#define TILE_M (WARP_TILE_M * WMMA_M)       // Sigue siendo 16
#define TILE_N (WARP_TILE_N * WMMA_N)       // Sigue siendo 16
#define WARPS_PER_BLOCK (WARP_TILE_M * WARP_TILE_N)
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * WARP_SIZE)

// K requiere 32 elementos lógicos, que equivalen a 16 bytes físicos
#define SHMEM_K_BYTES 16

// Padding para evitar Bank Conflicts
#define PAD_K_BYTES 16
#define SHMEM_STRIDE_BYTES (SHMEM_K_BYTES + PAD_K_BYTES)
#define SHMEM_STRIDE_ELEMENTS (SHMEM_STRIDE_BYTES * 2) 

#define SHMEM_C_N (TILE_N + 8)

#if (WARP_TILE_M < 1) || (WARP_TILE_N < 1)
#error "WARP_TILE_M y WARP_TILE_N deben ser al menos 1."
#endif

#if THREADS_PER_BLOCK > 1024
#error "Demasiados warps por bloque: THREADS_PER_BLOCK no puede superar 1024."
#endif

#define CUDA_CHK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while (0)

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

static inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// EUCLIDEAN DISTANCE
__global__ void CalculateDistance(const int32_t *XXT, const int *norms, int *distances, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < m && row <= col) {
        int xxt = XXT[row * m + col];
        distances[row * m + col] = norms[row] + norms[col] - 2 * xxt;
    }
}

// XXT = X * X^T usando Tensor Cores de 4 bits (u4).
__global__ void XXT_WMMA_Shared(const uint8_t *X, int32_t *C, int m, int n) {
    int row_base = blockIdx.y * TILE_M;
    int col_base = blockIdx.x * TILE_N;

    if (col_base + TILE_N <= row_base) return;

    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_tile_row = warp_id / WARP_TILE_N;
    int warp_tile_col = warp_id % WARP_TILE_N;

    int subtile_row = row_base + warp_tile_row * WMMA_M;
    int subtile_col = col_base + warp_tile_col * WMMA_N;

    bool compute_subtile = (subtile_col + WMMA_N > subtile_row);

    // Memoria compartida con PADDING aplicado en la dimensión de los bytes (stride)
    __shared__ __align__(32) uint8_t a_tile[TILE_M][SHMEM_STRIDE_BYTES];
    __shared__ __align__(32) uint8_t b_tile[TILE_N][SHMEM_STRIDE_BYTES];
    __shared__ __align__(32) int32_t c_tile[TILE_M][SHMEM_C_N];

    // Uso del namespace experimental para precisión u4 (unsigned 4-bit)
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::experimental::precision::u4, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::experimental::precision::u4, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag;

    wmma::fill_fragment(c_frag, 0);

    int n_bytes = n / 2; 

    // k0 avanza de a WMMA_K (64 elementos lógicos por iteración = 32 bytes)
    for (int k0 = 0; k0 < n; k0 += WMMA_K) {
        int k0_bytes = k0 / 2;
        
        // Carga cooperativa de A (Copia directa de bytes hasta SHMEM_K_BYTES)
        for (int idx = threadIdx.x; idx < TILE_M * SHMEM_K_BYTES; idx += blockDim.x) {
            int local_row = idx / SHMEM_K_BYTES;
            int local_k_byte = idx % SHMEM_K_BYTES;
            int global_k_byte = k0_bytes + local_k_byte;
            int a_row = row_base + local_row;

            a_tile[local_row][local_k_byte] = (a_row < m && global_k_byte < n_bytes) ? X[a_row * n_bytes + global_k_byte] : 0;
        }

        // Carga cooperativa de B (Copia directa de bytes hasta SHMEM_K_BYTES)
        for (int idx = threadIdx.x; idx < TILE_N * SHMEM_K_BYTES; idx += blockDim.x) {
            int local_row = idx / SHMEM_K_BYTES;
            int local_k_byte = idx % SHMEM_K_BYTES;
            int global_k_byte = k0_bytes + local_k_byte;
            int b_row = col_base + local_row;

            b_tile[local_row][local_k_byte] = (b_row < m && global_k_byte < n_bytes) ? X[b_row * n_bytes + global_k_byte] : 0;
        }

        __syncthreads();

        if (compute_subtile) {
            const uint8_t *a_ptr = &a_tile[warp_tile_row * WMMA_M][0];
            const uint8_t *b_ptr = &b_tile[warp_tile_col * WMMA_N][0];

            // IMPORTANTE: El stride se pasa en ELEMENTOS lógicos (SHMEM_STRIDE_ELEMENTS)
            // Esto le indica a la API que hay padding físico al final de cada fila.
            wmma::load_matrix_sync(a_frag, a_ptr, SHMEM_STRIDE_ELEMENTS);
            wmma::load_matrix_sync(b_frag, b_ptr, SHMEM_STRIDE_ELEMENTS);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    if (compute_subtile) {
        int local_row = warp_tile_row * WMMA_M;
        int local_col = warp_tile_col * WMMA_N;
        wmma::store_matrix_sync(&c_tile[local_row][local_col], c_frag, SHMEM_C_N, wmma::mem_row_major);
    }

    __syncthreads();

    for (int idx = threadIdx.x; idx < TILE_M * TILE_N; idx += blockDim.x) {
        int local_row = idx / TILE_N;
        int local_col = idx % TILE_N;
        int global_row = row_base + local_row;
        int global_col = col_base + local_col;

        if (global_row < m && global_col < m && global_row <= global_col) {
            C[global_row * m + global_col] = c_tile[local_row][local_col];
        }
    }
}

__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Cálculo de normas (se mantiene igual, ya está optimizado para desempaquetar)
__global__ void CalculateNormVector(const uint8_t *matrix, int *norms, int m, int n) {
    int row = blockIdx.x;
    if (row >= m) return;

    int local_norm = 0;
    int n_bytes = n / 2;

    for (int col_byte = threadIdx.x; col_byte < n_bytes; col_byte += blockDim.x) {
        uint8_t byte = matrix[row * n_bytes + col_byte];
        
        int v0 = byte & 0x0F;
        int v1 = (byte >> 4) & 0x0F;
        
        local_norm += (v0 * v0) + (v1 * v1);
    }

    local_norm = warpReduceSum(local_norm);

    __shared__ int warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = (blockDim.x + warpSize - 1) / warpSize;

    if (lane == 0) warp_sums[warp_id] = local_norm;

    __syncthreads();

    if (warp_id == 0) {
        int sum = (lane < warps_per_block) ? warp_sums[lane] : 0;
        sum = warpReduceSum(sum);
        if (lane == 0) norms[row] = sum;
    }
}

// ... (Las funciones generate_genomic_matrix_packed, print_matrix_packed_u8, validate_small_case, print_matrix_i32 y el main() se mantienen idénticas, ya que la lógica de host y validación era correcta).

// Genera datos sinteticos empaquetando dos elementos de 4 bits por byte
void generate_genomic_matrix_packed(uint8_t *matrix, int m, int n) {
    srand(42);
    int n_bytes = n / 2;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n_bytes; j++) {
            uint8_t v0 = rand() % 3; // Valores 0, 1, 2
            uint8_t v1 = rand() % 3;
            
            // Empaquetado: v0 en los bajos, v1 en los altos
            matrix[i * n_bytes + j] = (v0 & 0x0F) | ((v1 & 0x0F) << 4);
        }
    }
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

// Validacion CPU adaptada para leer datos empaquetados en int4
bool validate_small_case(const uint8_t *X, const int32_t *XXT, const int *norms, int m, int n) {
    int n_bytes = n / 2;
    for (int i = 0; i < m; i++) {
        int ref_norm = 0;
        for (int k = 0; k < n; k++) {
            int byte_idx = k / 2;
            int shift = (k % 2) * 4;
            int v = (X[i * n_bytes + byte_idx] >> shift) & 0x0F;
            ref_norm += v * v;
        }
        if (norms[i] != ref_norm) {
            printf("Error norma fila %d: GPU=%d CPU=%d\n", i, norms[i], ref_norm);
            return false;
        }

        for (int j = i; j < m; j++) {
            int32_t ref = 0;
            for (int k = 0; k < n; k++) {
                int byte_idx = k / 2;
                int shift = (k % 2) * 4;
                int vi = (X[i * n_bytes + byte_idx] >> shift) & 0x0F;
                int vj = (X[j * n_bytes + byte_idx] >> shift) & 0x0F;
                ref += vi * vj;
            }
            
            if (XXT[i * m + j] != ref) {
                printf("Error XXT(%d,%d): GPU=%d CPU=%d\n", i, j, XXT[i * m + j], ref);
                return false;
            }
        }
    }
    return true;
}

void print_matrix_i32(const int32_t *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int m = (1 << 10);  // individuos
    int n = (1 << 15);  // SNPs (Debe ser multiplo de 32 para WMMA_K)

    if (argc >= 3) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }

    if (m <= 0 || n <= 0 || n % 32 != 0) {
        fprintf(stderr, "Uso: %s [individuos m] [SNPs n (multiplo de 32)]\n", argv[0]);
        return 1;
    }

    size_t matrix_elems = (size_t)m * (size_t)n;
    // Reduccion crucial: La matriz ocupa la mitad de los bytes originales
    size_t matrix_bytes = matrix_elems * sizeof(uint8_t) / 2; 
    size_t square_elems = (size_t)m * (size_t)m;
    size_t square_i32_bytes = square_elems * sizeof(int32_t);

    uint8_t *h_matrix = (uint8_t*)malloc(matrix_bytes);

    if (!h_matrix) {
        fprintf(stderr, "No hay memoria de host suficiente.\n");
        return 1;
    }

    uint8_t *d_matrix = NULL;
    int *d_norms = NULL;
    int32_t *d_XXT = NULL;
    int *d_distances = NULL;

    CUDA_CHK(cudaMalloc(&d_matrix, matrix_bytes));
    CUDA_CHK(cudaMalloc(&d_norms, (size_t)m * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_XXT, square_i32_bytes));
    CUDA_CHK(cudaMalloc(&d_distances, square_i32_bytes));

    printf("Generando matriz genomica empaquetada X_int4 (%d individuos x %d SNPs)...\n", m, n);
    generate_genomic_matrix_packed(h_matrix, m, n);

    // WARM UP:
    CUDA_CHK(cudaMemcpy(d_matrix, h_matrix, matrix_bytes, cudaMemcpyHostToDevice));
    if (m <= 64 && n <= 128) {
        printf("\nX (debug):\n");
        print_matrix_packed_u8(h_matrix, m, n);
    }

    // NORMAS -----------------------------------------------------
    dim3 block_norms(256);
    dim3 grid_norms(m);
    CalculateNormVector<<<grid_norms, block_norms>>>(d_matrix, d_norms, m, n);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    printf("Normas finalizado.\n");
    CUDA_CHK(cudaMemset(d_XXT, 0, square_i32_bytes));

    // X*X^T -----------------------------------------------------
    dim3 block_syrk(THREADS_PER_BLOCK);
    dim3 grid_syrk(div_up(m, TILE_N), div_up(m, TILE_M));
    XXT_WMMA_Shared<<<grid_syrk, block_syrk>>>(d_matrix, d_XXT, m, n);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaMemset(d_distances, 0, square_i32_bytes));
    printf("XXT con tensor cores finalizado\n");

    // DISTANCIAS EUCLIDEAS -----------------------------------------------------
    dim3 block_dist(16, 16);
    dim3 grid_dist(div_up(m, block_dist.x), div_up(m, block_dist.y));
    CalculateDistance<<<grid_dist, block_dist>>>(d_XXT, d_norms, d_distances, m);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    // Loop de perfilado NVTX
    for (int i = 0; i < 10; i++) {
        CUDA_CHK(cudaMemcpy(d_matrix, h_matrix, matrix_bytes, cudaMemcpyHostToDevice));

        // NORMAS
        nvtxRangePushA("Norms");
        CalculateNormVector<<<grid_norms, block_norms>>>(d_matrix, d_norms, m, n);
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();

        CUDA_CHK(cudaMemset(d_XXT, 0, square_i32_bytes));

        // XXT WMMA int4
        nvtxRangePushA("XXT_Int4");
        XXT_WMMA_Shared<<<grid_syrk, block_syrk>>>(d_matrix, d_XXT, m, n);
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();

        CUDA_CHK(cudaMemset(d_distances, 0, square_i32_bytes));

        // DISTANCIAS
        nvtxRangePushA("CalculateDistance");
        CalculateDistance<<<grid_dist, block_dist>>>(d_XXT, d_norms, d_distances, m);
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop(); 
    }

    // DEBUGS para validacion en casos pequeños
    if (m <= 64 && n <= 1024) {
        int *h_norms = (int*)malloc((size_t)m * sizeof(int));
        int32_t *h_XXT = (int32_t*)malloc(square_i32_bytes);
        
        CUDA_CHK(cudaMemcpy(h_norms, d_norms, (size_t)m * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHK(cudaMemcpy(h_XXT, d_XXT, square_i32_bytes, cudaMemcpyDeviceToHost));

        printf("Normas (debug):\n");
        for (int i = 0; i < m; i++) printf("%d ", h_norms[i]);
        printf("\n");

        bool ok = validate_small_case(h_matrix, h_XXT, h_norms, m, n);
        printf("Validacion CPU/GPU: %s\n", ok ? "OK" : "FALLO");

        int *h_distances = (int*)malloc(square_i32_bytes);
        fprintf(stderr, "\nDistancias euclideas al cuadrado (debug):\n");
        CUDA_CHK(cudaMemcpy(h_distances, d_distances, square_i32_bytes, cudaMemcpyDeviceToHost));
        print_matrix_i32(h_distances, m, m);

        free(h_norms);
        free(h_XXT);
        free(h_distances);
    }

    cudaFree(d_matrix);
    cudaFree(d_norms);
    cudaFree(d_XXT);
    cudaFree(d_distances);
    free(h_matrix);

    return 0;
}
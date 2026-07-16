#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <nvtx3/nvToolsExt.h>

using namespace nvcuda;

#define GENOMIC_MATRIX_SEED 42u

#define CUDA_CHK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while (0)

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

static inline size_t div_up_size(size_t a, size_t b) {
    return (a + b - 1) / b;
}

static __device__ __host__ inline size_t get_packed_index(size_t r, size_t c, size_t m) {
    return r * m - (r * (r + 1)) / 2 + c;
}

// DISTANCIAS EUCLIDEAS CON INPUTS UNSIGNED INT
__global__ void CalculateDistance(const uint32_t *XXT, const uint32_t *norms,
                                  float *distances, size_t m) {
    size_t row = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < m && row <= col) {
        size_t packed_idx = get_packed_index(row, col, m);
        uint32_t xxt = XXT[packed_idx];
        uint32_t norm_row = norms[row];
        uint32_t norm_col = norms[col];
        
        uint32_t sum_norms = norm_row + norm_col;
        float dist_f = 0.0f;
        
        if (sum_norms > 2 * xxt) {
            dist_f = sqrtf((float)(sum_norms - 2 * xxt));
        }
        
        distances[packed_idx] = dist_f;
    }
}

// XXT CON PLANTILLAS PARA PARÁMETROS DINÁMICOS
template <int WARP_TILE_M, int WARP_TILE_N, int WMMA_M, int WMMA_N, int WMMA_K>
__global__ void XXT_WMMA_Shared(const uint8_t *X, uint32_t *C, size_t m, size_t n) {

    constexpr int WARP_SIZE = 32;
    constexpr int TILE_M = WARP_TILE_M * WMMA_M;
    constexpr int TILE_N = WARP_TILE_N * WMMA_N;
    constexpr int SHMEM_K = 32;
    constexpr int SHMEM_C_N = TILE_N + 8;

    // Evaluaciones en tiempo de compilación para seguridad
    static_assert(WARP_TILE_M >= 1 && WARP_TILE_N >= 1, "WARP_TILE_M y N deben ser al menos 1.");
    static_assert((WARP_TILE_M * WARP_TILE_N * WARP_SIZE) <= 1024, "Demasiados warps por bloque.");
    static_assert(SHMEM_K >= WMMA_K, "SHMEM_K debe ser al menos WMMA_K.");
    static_assert(SHMEM_C_N >= TILE_N, "SHMEM_C_N debe ser al menos TILE_N.");

    size_t row_base = (size_t)blockIdx.y * TILE_M;
    size_t col_base = (size_t)blockIdx.x * TILE_N;

    if (col_base + TILE_N <= row_base) return;

    int warp_id = threadIdx.x / WARP_SIZE;
    size_t warp_tile_row = (size_t)warp_id / WARP_TILE_N;
    size_t warp_tile_col = (size_t)warp_id % WARP_TILE_N;

    size_t subtile_row = row_base + warp_tile_row * WMMA_M;
    size_t subtile_col = col_base + warp_tile_col * WMMA_N;

    bool compute_subtile = (subtile_col + WMMA_N > subtile_row);
    size_t n_size = n;

    __shared__ __align__(32) uint8_t a_tile[TILE_M][SHMEM_K];
    __shared__ __align__(32) uint8_t b_tile[TILE_N][SHMEM_K];
    __shared__ __align__(32) int32_t c_tile[TILE_M][SHMEM_C_N];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, unsigned char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, unsigned char, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_frag;

    wmma::fill_fragment(c_frag, 0);

    for (size_t k0 = 0; k0 < n; k0 += WMMA_K) {
        for (int idx = threadIdx.x; idx < TILE_M * WMMA_K; idx += blockDim.x) {
            int local_row = idx / WMMA_K;
            int local_k = idx % WMMA_K;
            size_t global_k = k0 + (size_t)local_k;
            size_t a_row = row_base + (size_t)local_row;

            a_tile[local_row][local_k] = (a_row < m && global_k < n) ? X[a_row * n_size + global_k] : 0;
        }

        for (int idx = threadIdx.x; idx < TILE_N * WMMA_K; idx += blockDim.x) {
            int local_row = idx / WMMA_K;
            int local_k = idx % WMMA_K;
            size_t global_k = k0 + (size_t)local_k;
            size_t b_row = col_base + (size_t)local_row;

            b_tile[local_row][local_k] = (b_row < m && global_k < n) ? X[b_row * n_size + global_k] : 0;
        }

        __syncthreads();

        if (compute_subtile) {
            const uint8_t *a_ptr = &a_tile[warp_tile_row * WMMA_M][0];
            const uint8_t *b_ptr = &b_tile[warp_tile_col * WMMA_N][0];

            wmma::load_matrix_sync(a_frag, a_ptr, SHMEM_K);
            wmma::load_matrix_sync(b_frag, b_ptr, SHMEM_K);
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

// Wrapper para lanzar el grid/block correcto dependiendo de los templates instanciados
template <int WTM, int WTN, int WM, int WN, int WK>
void launch_XXT(const uint8_t* d_matrix, uint32_t* d_XXT, size_t m, size_t n) {
    constexpr int TILE_M = WTM * WM;
    constexpr int TILE_N = WTN * WN;
    constexpr int THREADS_PER_BLOCK = WTM * WTN * 32;
    
    dim3 block_syrk(THREADS_PER_BLOCK);
    dim3 grid_syrk((unsigned)div_up_size(m, TILE_N), (unsigned)div_up_size(m, TILE_M));
    
    XXT_WMMA_Shared<WTM, WTN, WM, WN, WK><<<grid_syrk, block_syrk>>>(d_matrix, d_XXT, m, n);
}

// Despachador en tiempo de ejecución 
void dispatch_XXT(const uint8_t* d_matrix, uint32_t* d_XXT, size_t m, size_t n, 
                  int wtm, int wtn, int wm, int wn, int wk) {
    // Configuraciones 2x2
    if (wtm == 2 && wtn == 2 && wm == 16 && wn == 16 && wk == 16) {
        launch_XXT<2, 2, 16, 16, 16>(d_matrix, d_XXT, m, n);
    } else if (wtm == 2 && wtn == 2 && wm == 32 && wn == 8 && wk == 16) {
        launch_XXT<2, 2, 32, 8, 16>(d_matrix, d_XXT, m, n);
    } else if (wtm == 2 && wtn == 2 && wm == 8 && wn == 32 && wk == 16) {
        launch_XXT<2, 2, 8, 32, 16>(d_matrix, d_XXT, m, n);
    }
    // Configuraciones 4x4
    else if (wtm == 4 && wtn == 4 && wm == 16 && wn == 16 && wk == 16) {
        launch_XXT<4, 4, 16, 16, 16>(d_matrix, d_XXT, m, n);
    } else if (wtm == 4 && wtn == 4 && wm == 32 && wn == 8 && wk == 16) {
        launch_XXT<4, 4, 32, 8, 16>(d_matrix, d_XXT, m, n);
    } else if (wtm == 4 && wtn == 4 && wm == 8 && wn == 32 && wk == 16) {
        launch_XXT<4, 4, 8, 32, 16>(d_matrix, d_XXT, m, n);
    } else {
        fprintf(stderr, "Error: Configuracion WMMA no soportada (%dx%d con %dx%dx%d)\n", wtm, wtn, wm, wn, wk);
        exit(1);
    }
}

// REDUCCIÓN PARA NORMAS EN UNSIGNED INT
__inline__ __device__ uint32_t warpReduceSum(uint32_t val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void CalculateNormVector(const uint8_t *matrix, uint32_t *norms, size_t m, size_t n) {
    size_t row = blockIdx.x;
    if (row >= m) return;

    uint32_t local_norm = 0;
    size_t n_size = (size_t)n;
    size_t row_offset = (size_t)row * n_size;
    for (size_t col = (size_t)threadIdx.x; col < n; col += (size_t)blockDim.x) {
        uint32_t v = matrix[row_offset + (size_t)col];
        local_norm += v * v;
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

__global__ void GenerateGenomicMatrixKernelU8(uint8_t *matrix, size_t m, size_t n, uint32_t seed) {
    size_t row = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n) return;

    uint32_t key = seed ^ ((uint32_t)row * 0x9e3779b9u) ^ ((uint32_t)col * 0x85ebca6bu);
    matrix[(size_t)row * (size_t)n + (size_t)col] = (uint8_t)(hash_u32(key) % 3u);
}

void generate_genomic_matrix_device(uint8_t *d_matrix, size_t m, size_t n) {
    dim3 block(32, 8);
    dim3 grid((unsigned)div_up_size(n, (size_t)block.x), (unsigned)div_up_size(m, (size_t)block.y));
    GenerateGenomicMatrixKernelU8<<<grid, block>>>(d_matrix, m, n, GENOMIC_MATRIX_SEED);
    CUDA_CHK(cudaGetLastError());
}

void print_packed_matrix_float(const float *matrix, int m) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            size_t packed_idx = (i <= j) ? get_packed_index(i, j, m) : get_packed_index(j, i, m);
            printf("%6.2f ", matrix[packed_idx]);
        }
        printf("\n");
    }
}
void print_matrix_packed_float(const float *matrix, int m) {
    print_packed_matrix_float(matrix, m);
}

void print_matrix_u8(const uint8_t *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%u ", (unsigned)matrix[i * cols + j]);
        }
        printf("\n");
    }
}

bool validate_small_case(const uint8_t *X, const uint32_t *XXT, const uint32_t *norms, size_t m, size_t n) {
    size_t n_size = n;
    for (size_t i = 0; i < m; i++) {
        uint32_t ref_norm = 0;
        for (size_t k = 0; k < n; k++) {
            uint32_t v = X[i * n_size + k];
            ref_norm += v * v;
        }
        if (norms[i] != ref_norm) {
            printf("Error norma fila %zu: GPU=%u CPU=%u\n", i, norms[i], ref_norm);
            return false;
        }

        for (size_t j = i; j < m; j++) {
            uint32_t ref = 0;
            for (size_t k = 0; k < n; k++) 
                ref += (uint32_t)X[i * n_size + k] * (uint32_t)X[j * n_size + k];
            
            size_t packed_idx = get_packed_index(i, j, m);
            if (XXT[packed_idx] != ref) {
                printf("Error XXT(%zu,%zu): GPU=%u CPU=%u\n", i, j, XXT[packed_idx], ref);
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char *argv[]) {
    size_t m = (size_t)(1 << 10);  
    size_t n = (size_t)(1 << 15);  
    printf("Memoria usada: %.2f MB\n", (float)(m * n * sizeof(uint8_t)) / (1024.0f * 1024.0f));

    // Parámetros por defecto para WMMA
    int warp_m = 4, warp_n = 4; //mas perform 
    int wmma_m = 16, wmma_n = 16, wmma_k = 16;

    if (argc >= 8) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        warp_m = atoi(argv[3]);
        warp_n = atoi(argv[4]);
        wmma_m = atoi(argv[5]);
        wmma_n = atoi(argv[6]);
        wmma_k = atoi(argv[7]);
    } else if (argc >= 3) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }

    if (m == 0 || n == 0) {
        fprintf(stderr, "Uso:\n %s [individuos m] [SNPs n]\n", argv[0]);
        fprintf(stderr, " %s [m] [n] [warp_m] [warp_n] [wmma_m] [wmma_n] [wmma_k]\n", argv[0]);
        return 1;
    }

    printf("Iniciando con configuracion WMMA: Warp %dx%d | Size %dx%dx%d\n", warp_m, warp_n, wmma_m, wmma_n, wmma_k);

    size_t matrix_elems = m * n;
    size_t matrix_bytes = matrix_elems * sizeof(uint8_t);
    size_t tri_elems = m * (m + 1) / 2;
    size_t tri_u32_bytes = tri_elems * sizeof(uint32_t); 
    size_t tri_float_bytes = tri_elems * sizeof(float);

    uint8_t *h_matrix = NULL;
    uint8_t *d_matrix = NULL;
    uint32_t *d_norms = NULL;      
    uint32_t *d_XXT = NULL;        
    float *d_distances = NULL; 

    CUDA_CHK(cudaMalloc(&d_matrix, matrix_bytes));
    CUDA_CHK(cudaMalloc(&d_norms, m * sizeof(uint32_t)));
    CUDA_CHK(cudaMalloc(&d_XXT, tri_u32_bytes));
    CUDA_CHK(cudaMalloc(&d_distances, tri_float_bytes));

    printf("Generando matriz genomica X (%zu individuos x %zu SNPs)...\n", m, n);

    nvtxRangePushA("GenX");
    generate_genomic_matrix_device(d_matrix, m, n);
    CUDA_CHK(cudaDeviceSynchronize());
    nvtxRangePop();

    bool need_host_matrix = (m <= 64 && n <= 1024);
    if (need_host_matrix) {
        h_matrix = (uint8_t*)malloc(matrix_bytes);
        if (!h_matrix) {
            fprintf(stderr, "No hay memoria de host suficiente para depuracion/validacion.\n");
            return 1;
        }
        CUDA_CHK(cudaMemcpy(h_matrix, d_matrix, matrix_bytes, cudaMemcpyDeviceToHost));
    }

    if (m <= 64 && n <= 128 && h_matrix) {
        printf("\nX (debug):\n");
        print_matrix_u8(h_matrix, m, n);
    }

    // NORMAS
    dim3 block_norms(256);
    dim3 grid_norms((unsigned)m);
    CalculateNormVector<<<grid_norms, block_norms>>>(d_matrix, d_norms, m, n);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    printf("Normas finalizado.\n");
    CUDA_CHK(cudaMemset(d_XXT, 0, tri_u32_bytes));

    // X*X^T via Dispatcher
    dispatch_XXT(d_matrix, d_XXT, m, n, warp_m, warp_n, wmma_m, wmma_n, wmma_k);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaMemset(d_distances, 0, tri_float_bytes)); 
    printf("SYRK con tensor cores finalizado\n");

    // DISTANCIAS
    dim3 block_dist(16, 16);
    dim3 grid_dist((unsigned)div_up_size(m, (size_t)block_dist.x), (unsigned)div_up_size(m, (size_t)block_dist.y));
    CalculateDistance<<<grid_dist, block_dist>>>(d_XXT, d_norms, d_distances, m);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    
    // BENCHMARK LOOP
    for (int i = 0; i < 10; i++) {
        nvtxRangePushA("Norms");
        CalculateNormVector<<<grid_norms, block_norms>>>(d_matrix, d_norms, m, n);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();

        CUDA_CHK(cudaMemset(d_XXT, 0, tri_u32_bytes));

        nvtxRangePushA("XXT");
        dispatch_XXT(d_matrix, d_XXT, m, n, warp_m, warp_n, wmma_m, wmma_n, wmma_k);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();

        CUDA_CHK(cudaMemset(d_distances, 0, tri_float_bytes));
        
        nvtxRangePushA("CalculateDistance");
        CalculateDistance<<<grid_dist, block_dist>>>(d_XXT, d_norms, d_distances, m);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop(); 
    }
    
    if (m <= 64 && n <= 1024 && h_matrix) {
        uint32_t *h_norms = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
        uint32_t *h_XXT = (uint32_t*)malloc(tri_u32_bytes);
        float *h_distances = (float*)malloc(tri_float_bytes);
        
        if (h_norms && h_XXT && h_distances) {
            CUDA_CHK(cudaMemcpy(h_norms, d_norms, (size_t)m * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHK(cudaMemcpy(h_XXT, d_XXT, tri_u32_bytes, cudaMemcpyDeviceToHost)); 
            CUDA_CHK(cudaMemcpy(h_distances, d_distances, tri_float_bytes, cudaMemcpyDeviceToHost));
            
            printf("Normas (debug):\n");
            for (size_t i = 0; i < m; i++) printf("%u ", h_norms[i]);
            printf("\n");

            bool ok = validate_small_case(h_matrix, h_XXT, h_norms, m, n);
            printf("Validacion CPU/GPU: %s\n", ok ? "OK" : "FALLO");

            fprintf(stderr, "\nDistancias euclideas (debug):\n");
            print_matrix_packed_float(h_distances, (int)m);
        }
        
        if (h_norms) free(h_norms);
        if (h_XXT) free(h_XXT);
        if (h_distances) free(h_distances);
    }

    cudaFree(d_matrix);
    cudaFree(d_norms);
    cudaFree(d_XXT);
    cudaFree(d_distances);
    if (h_matrix) free(h_matrix);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>

using namespace nvcuda;

// ====================================================================================
// MACROS Y FUNCIONES AUXILIARES COMUNES
// ====================================================================================
#define CUDA_CHK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while (0)

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

static const char* cublasGetErrorStr(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
        default:                              return "UNKNOWN_CUBLAS_ERROR";
    }
}

#define CUBLAS_CHK(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "[CUBLAS ERROR] %s:%d: %s (code %d)\n", __FILE__, __LINE__, \
                cublasGetErrorStr(_s), (int)_s); \
        exit(1); \
    } \
} while (0)

static inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

static __device__ __host__ inline size_t get_packed_index(size_t r, size_t c, size_t m) {
    return r * m - (r * (r + 1)) / 2 + c;
}

static __device__ __host__ inline size_t get_packed_index_U4(size_t r, size_t c, size_t m) {
    if (r > c) {
        size_t tmp = r;
        r = c;
        c = tmp;
    }
    return r * m - (r * (r + 1)) / 2 + c;
}

// ====================================================================================
// CONFIGURACIÓN ESPECÍFICA PARA U8 (WMMA)
// ====================================================================================
#define WARP_TILE_M_U8 2
#define WARP_TILE_N_U8 2
#define WMMA_M_U8 16
#define WMMA_N_U8 16
#define WMMA_K_U8 16
#define WARP_SIZE_U8 32
#define TILE_M_U8 (WARP_TILE_M_U8 * WMMA_M_U8)
#define TILE_N_U8 (WARP_TILE_N_U8 * WMMA_N_U8)
#define WARPS_PER_BLOCK_U8 (WARP_TILE_M_U8 * WARP_TILE_N_U8)
#define THREADS_PER_BLOCK_U8 (WARPS_PER_BLOCK_U8 * WARP_SIZE_U8)

#define SHMEM_K_U8 32
#define SHMEM_C_N_U8 (TILE_N_U8 + 8)

// ====================================================================================
// CONFIGURACIÓN ESPECÍFICA PARA U4 (WMMA)
// ====================================================================================
#define WARP_TILE_M_U4 2
#define WARP_TILE_N_U4 2
#define WMMA_M_U4 8
#define WMMA_N_U4 8
#define WMMA_K_U4 32
#define WARP_SIZE_U4 32
#define TILE_M_U4 (WARP_TILE_M_U4 * WMMA_M_U4)       
#define TILE_N_U4 (WARP_TILE_N_U4 * WMMA_N_U4)       
#define WARPS_PER_BLOCK_U4 (WARP_TILE_M_U4 * WARP_TILE_N_U4)
#define THREADS_PER_BLOCK_U4 (WARPS_PER_BLOCK_U4 * WARP_SIZE_U4)

#define SHMEM_K_BYTES_U4 16
#define PAD_K_BYTES_U4 16
#define SHMEM_STRIDE_BYTES_U4 (SHMEM_K_BYTES_U4 + PAD_K_BYTES_U4)
#define SHMEM_STRIDE_ELEMENTS_U4 (SHMEM_STRIDE_BYTES_U4 * 2) 
#define SHMEM_C_N_U4 (TILE_N_U4 + 8)

// ====================================================================================
// CONFIGURACIÓN ESPECÍFICA PARA U2 (MANUAL __dp4a)
// ====================================================================================
#define VALUES_PER_BYTE_U2 4
#define BITS_PER_VALUE_U2 2
#define PACKED_VALUE_MASK_U2 0x03u

#define TILE_M_U2 16
#define TILE_N_U2 16
#define TILE_K_U2 32

__host__ __device__ static inline size_t get_packed_index_U2(int row, int col, int m) {
    if (row > col) {
        int tmp = row; 
        row = col; 
        col = tmp;
    }
    return (size_t)row * m - ((size_t)row * (row + 1)) / 2 + col;
}

static inline void set_packed_genotype_U2(uint8_t *matrix, int packed_cols, int row, int col, uint8_t value) {
    size_t byte_index = (size_t)row * (size_t)packed_cols + (size_t)(col / VALUES_PER_BYTE_U2);
    int shift = (col % VALUES_PER_BYTE_U2) * BITS_PER_VALUE_U2;
    uint8_t mask = (uint8_t)(PACKED_VALUE_MASK_U2 << shift);
    uint8_t encoded = (uint8_t)((value & PACKED_VALUE_MASK_U2) << shift);
    matrix[byte_index] = (uint8_t)((matrix[byte_index] & ~mask) | encoded);
}

__host__ __device__ static inline uint32_t unpack_2bit_to_4x8_U2(uint8_t packed_byte) {
    uint32_t g0 = packed_byte & 0x03u;
    uint32_t g1 = (packed_byte >> 2) & 0x03u;
    uint32_t g2 = (packed_byte >> 4) & 0x03u;
    uint32_t g3 = (packed_byte >> 6) & 0x03u;
    return g0 | (g1 << 8) | (g2 << 16) | (g3 << 24);
}


// ====================================================================================
// KERNELS AUXILIARES - REFERENCIA cuBLAS (Ssyrk, float32) Y VERIFICACION NUMERICA
// ====================================================================================

// Convierte la matriz sin empaquetar (uint8_t, valores {0,1,2}) a float, requerido
// por cublasSsyrk. Es la MISMA matriz base (mismo srand(42)) que usan las secciones
// U8/U4/U2 (ver generate_genomic_matrix* mas abajo: todas consumen rand() en el mismo
// orden logico por elemento, asi que representan exactamente los mismos genotipos aunque
// esten empaquetados distinto), por lo que sirve como referencia de "verdad" comun.
__global__ void u8ToFloatKernel(const uint8_t* in, float* out, size_t count) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) out[idx] = (float)in[idx];
}

// cublasSsyrk con CUBLAS_FILL_MODE_UPPER solo llena, en terminos row-major, la mitad
// con fila >= columna (ver detalle discutido: la convencion column-major de cuBLAS
// mapea al reves de lo intuitivo cuando se lee el buffer como row-major). Este kernel
// copia esa mitad valida hacia la mitad sin inicializar antes de usar C.
__global__ void completarMatrizKernel(float* C, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N && i < j) {
        C[i * N + j] = C[j * N + i];
    }
}

// La diagonal de XXT es exactamente la norma al cuadrado de cada individuo.
__global__ void extractDiagKernel(const float* C, float* diag, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) diag[i] = C[i * N + i];
}

// Arma D en el MISMO formato empaquetado triangular (i<=j) que usan las variantes
// cuantizadas, para poder comparar elemento a elemento sin tener que "desempaquetar"
// nada del lado de las otras variantes.
__global__ void buildDPackedKernel(const float* C, const float* diag, float* Dpacked, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N && i <= j) {
        float d2 = diag[i] + diag[j] - 2.0f * C[i * N + j];
        if (d2 < 0.0f) d2 = 0.0f;  // por posibles errores de redondeo en float
        size_t idx = get_packed_index((size_t)i, (size_t)j, (size_t)N);
        Dpacked[idx] = sqrtf(d2);
    }
}

// Imprime en consola: ||D_variante - D_cuBLAS||_2 usando cublasSaxpy + cublasSnrm2,
// tal como pide la letra. d_distances_float es el resultado en float de cada
// variante (U8/U4/U2); d_Dpacked_ref es la referencia float32 de cuBLAS.
void verificar_contra_cublas(cublasHandle_t handle, const char* nombre_variante,
                              const float* d_distances_float, const float* d_Dpacked_ref,
                              size_t tri_elems) {
    if (tri_elems > (size_t)INT_MAX) {
        fprintf(stderr, "[WARN] tri_elems excede INT_MAX, cublasSaxpy/Snrm2 no lo soportan tal cual.\n");
        return;
    }

    float *d_variant_f = NULL;
    CUDA_CHK(cudaMalloc(&d_variant_f, tri_elems * sizeof(float)));
    CUDA_CHK(cudaMemcpy(d_variant_f, d_distances_float, tri_elems * sizeof(float), cudaMemcpyDeviceToDevice));

    float neg_one = -1.0f;
    // d_variant_f = d_variant_f - d_Dpacked_ref  =  D_variante - D_cuBLAS
    CUBLAS_CHK(cublasSaxpy(handle, (int)tri_elems, &neg_one, d_Dpacked_ref, 1, d_variant_f, 1));

    float norm_diff = 0.0f;
    CUBLAS_CHK(cublasSnrm2(handle, (int)tri_elems, d_variant_f, 1, &norm_diff));

    printf("[VERIFICACION] ||D_%s - D_cuBLAS||_2 = %e\n", nombre_variante, norm_diff);

    CUDA_CHK(cudaFree(d_variant_f));
}

// ====================================================================================
// KERNELS - VERSION U8
// ====================================================================================
__global__ void CalculateDistanceU8(const uint32_t *XXT, const uint32_t *norms, float *distances, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

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

__global__ void XXT_WMMA_Shared_U8(const uint8_t *X, uint32_t *C, int m, int n) {
    int row_base = blockIdx.y * TILE_M_U8;
    int col_base = blockIdx.x * TILE_N_U8;

    if (col_base + TILE_N_U8 <= row_base) return;

    int warp_id = threadIdx.x / WARP_SIZE_U8;
    int warp_tile_row = warp_id / WARP_TILE_N_U8;
    int warp_tile_col = warp_id % WARP_TILE_N_U8;

    int subtile_row = row_base + warp_tile_row * WMMA_M_U8;
    int subtile_col = col_base + warp_tile_col * WMMA_N_U8;

    bool compute_subtile = (subtile_col + WMMA_N_U8 > subtile_row);

    __shared__ __align__(32) uint8_t a_tile[TILE_M_U8][SHMEM_K_U8];
    __shared__ __align__(32) uint8_t b_tile[TILE_N_U8][SHMEM_K_U8];
    __shared__ __align__(32) int32_t c_tile[TILE_M_U8][SHMEM_C_N_U8];

    wmma::fragment<wmma::matrix_a, WMMA_M_U8, WMMA_N_U8, WMMA_K_U8, unsigned char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M_U8, WMMA_N_U8, WMMA_K_U8, unsigned char, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M_U8, WMMA_N_U8, WMMA_K_U8, int> c_frag;

    wmma::fill_fragment(c_frag, 0);

    for (int k0 = 0; k0 < n; k0 += WMMA_K_U8) {
        for (int idx = threadIdx.x; idx < TILE_M_U8 * WMMA_K_U8; idx += blockDim.x) {
            int local_row = idx / WMMA_K_U8;
            int local_k = idx % WMMA_K_U8;
            int global_k = k0 + local_k;
            int a_row = row_base + local_row;

            a_tile[local_row][local_k] = (a_row < m && global_k < n) ? X[a_row * n + global_k] : 0;
        }

        for (int idx = threadIdx.x; idx < TILE_N_U8 * WMMA_K_U8; idx += blockDim.x) {
            int local_row = idx / WMMA_K_U8;
            int local_k = idx % WMMA_K_U8;
            int global_k = k0 + local_k;
            int b_row = col_base + local_row;

            b_tile[local_row][local_k] = (b_row < m && global_k < n) ? X[b_row * n + global_k] : 0;
        }

        __syncthreads();

        if (compute_subtile) {
            const uint8_t *a_ptr = &a_tile[warp_tile_row * WMMA_M_U8][0];
            const uint8_t *b_ptr = &b_tile[warp_tile_col * WMMA_N_U8][0];

            wmma::load_matrix_sync(a_frag, a_ptr, SHMEM_K_U8);
            wmma::load_matrix_sync(b_frag, b_ptr, SHMEM_K_U8);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    if (compute_subtile) {
        int local_row = warp_tile_row * WMMA_M_U8;
        int local_col = warp_tile_col * WMMA_N_U8;
        wmma::store_matrix_sync(&c_tile[local_row][local_col], c_frag, SHMEM_C_N_U8, wmma::mem_row_major);
    }

    __syncthreads();

    for (int idx = threadIdx.x; idx < TILE_M_U8 * TILE_N_U8; idx += blockDim.x) {
        int local_row = idx / TILE_N_U8;
        int local_col = idx % TILE_N_U8;
        int global_row = row_base + local_row;
        int global_col = col_base + local_col;

        if (global_row < m && global_col < m && global_row <= global_col) {
            size_t packed_idx = get_packed_index(global_row, global_col, m);
            C[packed_idx] = (uint32_t)c_tile[local_row][local_col];
        }
    }
}

__inline__ __device__ uint32_t warpReduceSumU8(uint32_t val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void CalculateNormVectorU8(const uint8_t *matrix, uint32_t *norms, int m, int n) {
    int row = blockIdx.x;
    if (row >= m) return;

    uint32_t local_norm = 0;
    for (int col = threadIdx.x; col < n; col += blockDim.x) {
        uint32_t v = matrix[row * n + col];
        local_norm += v * v;
    }

    local_norm = warpReduceSumU8(local_norm);

    __shared__ uint32_t warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = (blockDim.x + 32 - 1) / 32;

    if (lane == 0) warp_sums[warp_id] = local_norm;

    __syncthreads();

    if (warp_id == 0) {
        uint32_t sum = (lane < warps_per_block) ? warp_sums[lane] : 0;
        sum = warpReduceSumU8(sum);
        if (lane == 0) norms[row] = sum;
    }
}

// ====================================================================================
// KERNELS - VERSION U4
// ====================================================================================
__global__ void CalculateDistanceU4(const uint32_t *XXT, const uint32_t *norms, float *distances, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < m && row <= col) {
        size_t idx = get_packed_index_U4(row, col, m);
        uint32_t xxt = XXT[idx];
        uint32_t norm_row = norms[row];
        uint32_t norm_col = norms[col];

        uint32_t sum_norms = norm_row + norm_col;
        float dist_f = 0.0f;
        
        if (sum_norms > 2 * xxt) {
            dist_f = sqrtf((float)(sum_norms - 2 * xxt));
        }

        distances[idx] = dist_f;
    }
}

__global__ void XXT_WMMA_Shared_U4(const uint8_t *X, uint32_t *C, int m, int n) {
    int row_base = blockIdx.y * TILE_M_U4;
    int col_base = blockIdx.x * TILE_N_U4;

    if (col_base + TILE_N_U4 <= row_base) return;

    int warp_id = threadIdx.x / WARP_SIZE_U4;
    int warp_tile_row = warp_id / WARP_TILE_N_U4;
    int warp_tile_col = warp_id % WARP_TILE_N_U4;

    int subtile_row = row_base + warp_tile_row * WMMA_M_U4;
    int subtile_col = col_base + warp_tile_col * WMMA_N_U4;

    bool compute_subtile = (subtile_col + WMMA_N_U4 > subtile_row);

    __shared__ __align__(32) uint8_t a_tile[TILE_M_U4][SHMEM_STRIDE_BYTES_U4];
    __shared__ __align__(32) uint8_t b_tile[TILE_N_U4][SHMEM_STRIDE_BYTES_U4];
    __shared__ __align__(32) int32_t c_tile[TILE_M_U4][SHMEM_C_N_U4];

    wmma::fragment<wmma::matrix_a, WMMA_M_U4, WMMA_N_U4, WMMA_K_U4, wmma::experimental::precision::u4, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M_U4, WMMA_N_U4, WMMA_K_U4, wmma::experimental::precision::u4, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M_U4, WMMA_N_U4, WMMA_K_U4, int32_t> c_frag; 

    wmma::fill_fragment(c_frag, 0);

    int n_bytes = n / 2; 

    for (int k0 = 0; k0 < n; k0 += WMMA_K_U4) {
        int k0_bytes = k0 / 2;
        
        for (int idx = threadIdx.x; idx < TILE_M_U4 * SHMEM_K_BYTES_U4; idx += blockDim.x) {
            int local_row = idx / SHMEM_K_BYTES_U4;
            int local_k_byte = idx % SHMEM_K_BYTES_U4;
            int global_k_byte = k0_bytes + local_k_byte;
            int a_row = row_base + local_row;

            a_tile[local_row][local_k_byte] = (a_row < m && global_k_byte < n_bytes) ? X[a_row * n_bytes + global_k_byte] : 0;
        }

        for (int idx = threadIdx.x; idx < TILE_N_U4 * SHMEM_K_BYTES_U4; idx += blockDim.x) {
            int local_row = idx / SHMEM_K_BYTES_U4;
            int local_k_byte = idx % SHMEM_K_BYTES_U4;
            int global_k_byte = k0_bytes + local_k_byte;
            int b_row = col_base + local_row;

            b_tile[local_row][local_k_byte] = (b_row < m && global_k_byte < n_bytes) ? X[b_row * n_bytes + global_k_byte] : 0;
        }

        __syncthreads();

        if (compute_subtile) {
            const uint8_t *a_ptr = &a_tile[warp_tile_row * WMMA_M_U4][0];
            const uint8_t *b_ptr = &b_tile[warp_tile_col * WMMA_N_U4][0];

            wmma::load_matrix_sync(a_frag, a_ptr, SHMEM_STRIDE_ELEMENTS_U4);
            wmma::load_matrix_sync(b_frag, b_ptr, SHMEM_STRIDE_ELEMENTS_U4);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    if (compute_subtile) {
        int local_row = warp_tile_row * WMMA_M_U4;
        int local_col = warp_tile_col * WMMA_N_U4;
        wmma::store_matrix_sync(&c_tile[local_row][local_col], c_frag, SHMEM_C_N_U4, wmma::mem_row_major);
    }

    __syncthreads();

    for (int idx = threadIdx.x; idx < TILE_M_U4 * TILE_N_U4; idx += blockDim.x) {
        int local_row = idx / TILE_N_U4;
        int local_col = idx % TILE_N_U4;
        int global_row = row_base + local_row;
        int global_col = col_base + local_col;

        if (global_row < m && global_col < m && global_row <= global_col) {
            size_t packed_idx = get_packed_index_U4(global_row, global_col, m);
            C[packed_idx] = (uint32_t)c_tile[local_row][local_col]; 
        }
    }
}

__inline__ __device__ uint32_t warpReduceSumU4(uint32_t val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void CalculateNormVectorU4(const uint8_t *matrix, uint32_t *norms, int m, int n) {
    int row = blockIdx.x;
    if (row >= m) return;

    uint32_t local_norm = 0;
    int n_bytes = n / 2;

    for (int col_byte = threadIdx.x; col_byte < n_bytes; col_byte += blockDim.x) {
        uint8_t byte = matrix[row * n_bytes + col_byte];
        
        uint32_t v0 = byte & 0x0F;
        uint32_t v1 = (byte >> 4) & 0x0F;
        
        local_norm += (v0 * v0) + (v1 * v1);
    }

    local_norm = warpReduceSumU4(local_norm);

    __shared__ uint32_t warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = (blockDim.x + 32 - 1) / 32;

    if (lane == 0) warp_sums[warp_id] = local_norm;

    __syncthreads();

    if (warp_id == 0) {
        uint32_t sum = (lane < warps_per_block) ? warp_sums[lane] : 0;
        sum = warpReduceSumU4(sum);
        if (lane == 0) norms[row] = sum;
    }
}

// ====================================================================================
// KERNELS - VERSION U2 (MANUAL __dp4a)
// ====================================================================================
__global__ void CalculateDistanceU2(const uint32_t *XXT_packed, const uint32_t *norms, float *distances_packed, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < m && row <= col) {
        size_t idx = get_packed_index_U2(row, col, m);
        uint32_t xxt = XXT_packed[idx];
        uint32_t norm_row = norms[row];
        uint32_t norm_col = norms[col];
        
        uint32_t sum_norms = norm_row + norm_col;
        float dist_f = 0.0f;
        
        if (sum_norms > 2 * xxt) {
            dist_f = sqrtf((float)(sum_norms - 2 * xxt));
        }

        distances_packed[idx] = dist_f;
    }
}

__global__ void XXT_Manual_U2(const uint8_t *X, uint32_t *C_packed, int m, int packed_cols) {
    int row_base = blockIdx.y * TILE_M_U2;
    int col_base = blockIdx.x * TILE_N_U2;

    if (col_base + TILE_N_U2 <= row_base) return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = row_base + ty;
    int col = col_base + tx;

    __shared__ uint32_t As[TILE_M_U2][TILE_K_U2 + 1];
    __shared__ uint32_t Bs[TILE_N_U2][TILE_K_U2 + 1];

    uint32_t acc = 0;

    for (int k0 = 0; k0 < packed_cols; k0 += TILE_K_U2) {
        for (int k = tx; k < TILE_K_U2; k += TILE_N_U2) {
            int idx = k0 + k;

            if (row < m && idx < packed_cols)
                As[ty][k] = unpack_2bit_to_4x8_U2(X[(size_t)row * packed_cols + idx]);
            else
                As[ty][k] = 0;

            int b_row = col_base + ty;
            
            if (b_row < m && idx < packed_cols)
                Bs[ty][k] = unpack_2bit_to_4x8_U2(X[(size_t)b_row * packed_cols + idx]);
            else
                Bs[ty][k] = 0;
        }

        __syncthreads();

    #pragma unroll
        for (int k = 0; k < TILE_K_U2; k++) {
            acc = __dp4a(As[ty][k], Bs[tx][k], acc);
        }

        __syncthreads();
    }

    if (row < m && col < m && row <= col) {
        size_t idx = get_packed_index_U2(row, col, m);
        C_packed[idx] = acc;
    }
}

__inline__ __device__ uint32_t warpReduceSumU2(uint32_t val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void CalculateNormVectorU2(const uint8_t *matrix, uint32_t *norms, int m, int packed_cols) {
    int row = blockIdx.x;
    if (row >= m) return;

    const uint8_t *row_ptr = matrix + (size_t)row * (size_t)packed_cols;
    uint32_t local_norm = 0;

    for (int byte_col = threadIdx.x; byte_col < packed_cols; byte_col += blockDim.x) {
        uint8_t packed_byte = row_ptr[byte_col];
        uint32_t unpacked = unpack_2bit_to_4x8_U2(packed_byte);
        local_norm = __dp4a(unpacked, unpacked, local_norm);
    }

    local_norm = warpReduceSumU2(local_norm);

    __shared__ uint32_t warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = (blockDim.x + warpSize - 1) / warpSize;

    if (lane == 0)
        warp_sums[warp_id] = local_norm;

    __syncthreads();

    if (warp_id == 0) {
        uint32_t sum = (lane < warps_per_block) ? warp_sums[lane] : 0;
        sum = warpReduceSumU2(sum);
        if (lane == 0) norms[row] = sum;
    }
}

// ====================================================================================
// GENERADORES DE DATOS EN HOST
// ====================================================================================
void generate_genomic_matrix(uint8_t *matrix, int m, int n) {
    srand(42);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = rand() % 3;
        }
    }
}

void generate_genomic_matrix_packed_U4(uint8_t *matrix, int m, int n) {
    srand(42);
    int n_bytes = n / 2;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n_bytes; j++) {
            uint8_t v0 = rand() % 3; 
            uint8_t v1 = rand() % 3;
            matrix[i * n_bytes + j] = (v0 & 0x0F) | ((v1 & 0x0F) << 4);
        }
    }
}

void generate_genomic_matrix_packed_U2(uint8_t *matrix, int m, int n, int packed_cols) {
    srand(42);
    memset(matrix, 0, (size_t)m * (size_t)packed_cols * sizeof(uint8_t));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            set_packed_genotype_U2(matrix, packed_cols, i, j, (uint8_t)(rand() % 3));
        }
    }
}

// ====================================================================================
// PROGRAMA PRINCIPAL
// ====================================================================================
int main(int argc, char *argv[]) {
    int m = (1 << 10);  
    int n = (1 << 15);  
    int count = 11;
    if (argc >= 3) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }
    
    if (m <= 0 || n <= 0 || n % 32 != 0) {
        fprintf(stderr, "Uso: %s [individuos m] [SNPs n (multiplo de 32)]\n", argv[0]);
        return 1;
    }

    size_t tri_elems = (size_t)m * (m + 1) / 2;

    // Handle y referencia de cuBLAS: vivos durante todo main() para poder
    // comparar cada variante cuantizada contra esta misma referencia.
    cublasHandle_t handle;
    CUBLAS_CHK(cublasCreate(&handle));
    float *d_Dpacked_ref = NULL;
    CUDA_CHK(cudaMalloc(&d_Dpacked_ref, tri_elems * sizeof(float)));

    // ===================================================================
    // SECCIÓN CUBLAS (referencia float32, obligatoria segun la letra)
    // ===================================================================
    {
        size_t matrix_bytes_ref = (size_t)m * (size_t)n * sizeof(uint8_t);

        uint8_t *h_matrix_ref = (uint8_t*)malloc(matrix_bytes_ref);
        uint8_t *d_matrix_ref = NULL;
        float *d_Xf = NULL, *d_C = NULL, *d_diag = NULL;

        CUDA_CHK(cudaMalloc(&d_matrix_ref, matrix_bytes_ref));
        CUDA_CHK(cudaMalloc(&d_Xf, (size_t)m * (size_t)n * sizeof(float)));
        CUDA_CHK(cudaMalloc(&d_C, (size_t)m * (size_t)m * sizeof(float)));
        CUDA_CHK(cudaMalloc(&d_diag, (size_t)m * sizeof(float)));

        // Mismo generador y misma semilla que usa la seccion U8: garantiza
        // que la referencia compara contra los MISMOS genotipos que ven
        // las variantes cuantizadas (ver nota en generate_genomic_matrix).
        generate_genomic_matrix(h_matrix_ref, m, n);
        CUDA_CHK(cudaMemcpy(d_matrix_ref, h_matrix_ref, matrix_bytes_ref, cudaMemcpyHostToDevice));
        for (size_t i = 0; i < count; i++)
        {    
            int threads = 256;
            int blocksConv = (int)(((size_t)m * (size_t)n + threads - 1) / threads);
            u8ToFloatKernel<<<blocksConv, threads>>>(d_matrix_ref, d_Xf, (size_t)m * (size_t)n);
            CUDA_CHK(cudaGetLastError());

            float alpha = 1.0f, beta = 0.0f;

            // Mismo truco row-major/column-major discutido antes: nuestro buffer
            // (m x n row-major) leido en column-major es X^T; con trans=OP_T,
            // Ssyrk calcula X * X^T directamente.
            CUBLAS_CHK(cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                    m, n, &alpha, d_Xf, n, &beta, d_C, m));

            dim3 blockSym(32, 32);
            dim3 gridSym((m + 31) / 32, (m + 31) / 32);
            completarMatrizKernel<<<gridSym, blockSym>>>(d_C, m);
            CUDA_CHK(cudaGetLastError());

            int blocksDiag = (m + threads - 1) / threads;
            extractDiagKernel<<<blocksDiag, threads>>>(d_C, d_diag, m);
            CUDA_CHK(cudaGetLastError());

            dim3 blockD(32, 32);
            dim3 gridD((m + 31) / 32, (m + 31) / 32);
            buildDPackedKernel<<<gridD, blockD>>>(d_C, d_diag, d_Dpacked_ref, m);
            CUDA_CHK(cudaGetLastError());
            CUDA_CHK(cudaDeviceSynchronize());
        }
        
        printf("[cuBLAS] Referencia D (float32, Ssyrk) calculada.\n");

        cudaFree(d_matrix_ref);
        cudaFree(d_Xf);
        cudaFree(d_C);
        cudaFree(d_diag);
        free(h_matrix_ref);
        // d_Dpacked_ref se mantiene vivo: es la referencia para todas las secciones siguientes
    }

    // ===================================================================
    // SECCIÓN WMMA_U8
    // ===================================================================
    {
        size_t matrix_bytes = (size_t)m * (size_t)n * sizeof(uint8_t);
        size_t tri_u32_bytes = tri_elems * sizeof(uint32_t); 
        size_t tri_float_bytes = tri_elems * sizeof(float);

        uint8_t *h_matrix = (uint8_t*)malloc(matrix_bytes);
        uint8_t *d_matrix = NULL;
        uint32_t *d_norms = NULL;      
        uint32_t *d_XXT = NULL;        
        float *d_distances_u8 = NULL; 

        CUDA_CHK(cudaMalloc(&d_matrix, matrix_bytes));
        CUDA_CHK(cudaMalloc(&d_norms, (size_t)m * sizeof(uint32_t))); 
        CUDA_CHK(cudaMalloc(&d_XXT, tri_u32_bytes));
        CUDA_CHK(cudaMalloc(&d_distances_u8, tri_float_bytes));

        generate_genomic_matrix(h_matrix, m, n);
        CUDA_CHK(cudaMemcpy(d_matrix, h_matrix, matrix_bytes, cudaMemcpyHostToDevice));
        for (size_t i = 0; i < count; i++)
        {
            // 1. Calcular Normas
            dim3 block_norms(256);
            dim3 grid_norms(m);
            CalculateNormVectorU8<<<grid_norms, block_norms>>>(d_matrix, d_norms, m, n);

            // 2. Calcular Producto XX^T
            CUDA_CHK(cudaMemset(d_XXT, 0, tri_u32_bytes));
            dim3 block_syrk(THREADS_PER_BLOCK_U8);
            dim3 grid_syrk(div_up(m, TILE_N_U8), div_up(m, TILE_M_U8));
            XXT_WMMA_Shared_U8<<<grid_syrk, block_syrk>>>(d_matrix, d_XXT, m, n);

            // 3. Calcular Matriz de Distancias de la sección
            CUDA_CHK(cudaMemset(d_distances_u8, 0, tri_float_bytes)); 
            dim3 block_dist(16, 16);
            dim3 grid_dist(div_up(m, block_dist.x), div_up(m, block_dist.y));
            CalculateDistanceU8<<<grid_dist, block_dist>>>(d_XXT, d_norms, d_distances_u8, m);
            CUDA_CHK(cudaDeviceSynchronize());
        }
        verificar_contra_cublas(handle, "U8", d_distances_u8, d_Dpacked_ref, tri_elems);

        cudaFree(d_matrix);
        cudaFree(d_norms);
        cudaFree(d_XXT);
        free(h_matrix);
        
        cudaFree(d_distances_u8); 
    }

    // ===================================================================
    // SECCIÓN WMMA_U4
    // ===================================================================
    {
        size_t matrix_bytes_u4 = ((size_t)m * (size_t)n) / 2; 
        size_t tri_u32_bytes_u4 = tri_elems * sizeof(uint32_t);
        size_t tri_float_bytes_u4 = tri_elems * sizeof(float);

        uint8_t *h_matrix_u4 = (uint8_t*)malloc(matrix_bytes_u4);
        uint8_t *d_matrix_u4 = NULL;
        uint32_t *d_norms_u4 = NULL;
        uint32_t *d_XXT_u4 = NULL;
        float *d_distances_u4 = NULL;

        CUDA_CHK(cudaMalloc(&d_matrix_u4, matrix_bytes_u4));
        CUDA_CHK(cudaMalloc(&d_norms_u4, (size_t)m * sizeof(uint32_t)));
        CUDA_CHK(cudaMalloc(&d_XXT_u4, tri_u32_bytes_u4));
        CUDA_CHK(cudaMalloc(&d_distances_u4, tri_float_bytes_u4));

        generate_genomic_matrix_packed_U4(h_matrix_u4, m, n);
        CUDA_CHK(cudaMemcpy(d_matrix_u4, h_matrix_u4, matrix_bytes_u4, cudaMemcpyHostToDevice));
        for (size_t i = 0; i < count; i++)
        {
            // 1. Calcular Normas
            dim3 block_norms_u4(256);
            dim3 grid_norms_u4(m);
            CalculateNormVectorU4<<<grid_norms_u4, block_norms_u4>>>(d_matrix_u4, d_norms_u4, m, n);

            // 2. Calcular Producto XX^T
            CUDA_CHK(cudaMemset(d_XXT_u4, 0, tri_u32_bytes_u4));
            dim3 block_syrk_u4(THREADS_PER_BLOCK_U4);
            dim3 grid_syrk_u4(div_up(m, TILE_N_U4), div_up(m, TILE_M_U4));
            XXT_WMMA_Shared_U4<<<grid_syrk_u4, block_syrk_u4>>>(d_matrix_u4, d_XXT_u4, m, n);

            // 3. Calcular Matriz de Distancias de la sección
            CUDA_CHK(cudaMemset(d_distances_u4, 0, tri_float_bytes_u4));
            dim3 block_dist_u4(16, 16);
            dim3 grid_dist_u4(div_up(m, block_dist_u4.x), div_up(m, block_dist_u4.y));
            CalculateDistanceU4<<<grid_dist_u4, block_dist_u4>>>(d_XXT_u4, d_norms_u4, d_distances_u4, m);
            CUDA_CHK(cudaDeviceSynchronize());
        }
        verificar_contra_cublas(handle, "U4", d_distances_u4, d_Dpacked_ref, tri_elems);

        cudaFree(d_matrix_u4);
        cudaFree(d_norms_u4);
        cudaFree(d_XXT_u4);
        free(h_matrix_u4);

        cudaFree(d_distances_u4);
    }

    // ===================================================================
    // SECCIÓN MANUAL_U2 (__dp4a)
    // ===================================================================
    {
        int packed_cols_u2 = div_up(n, VALUES_PER_BYTE_U2);
        size_t matrix_bytes_u2 = (size_t)m * (size_t)packed_cols_u2 * sizeof(uint8_t);
        size_t tri_u32_bytes_u2 = tri_elems * sizeof(uint32_t);
        size_t tri_float_bytes_u2 = tri_elems * sizeof(float);

        uint8_t *h_matrix_u2 = (uint8_t*)malloc(matrix_bytes_u2);
        uint8_t *d_matrix_u2 = NULL;
        uint32_t *d_norms_u2 = NULL;
        uint32_t *d_XXT_u2 = NULL;
        float *d_distances_u2 = NULL; 

        CUDA_CHK(cudaMalloc(&d_matrix_u2, matrix_bytes_u2));
        CUDA_CHK(cudaMalloc(&d_norms_u2, (size_t)m * sizeof(uint32_t)));
        CUDA_CHK(cudaMalloc(&d_XXT_u2, tri_u32_bytes_u2));
        CUDA_CHK(cudaMalloc(&d_distances_u2, tri_float_bytes_u2));

        generate_genomic_matrix_packed_U2(h_matrix_u2, m, n, packed_cols_u2);
        CUDA_CHK(cudaMemcpy(d_matrix_u2, h_matrix_u2, matrix_bytes_u2, cudaMemcpyHostToDevice));
        for (size_t i = 0; i < count; i++)
        {
            // 1. Calcular Normas
            dim3 block_norms_u2(256);
            dim3 grid_norms_u2(m);
            CalculateNormVectorU2<<<grid_norms_u2, block_norms_u2>>>(d_matrix_u2, d_norms_u2, m, packed_cols_u2);

            // 2. Calcular Producto XX^T
            CUDA_CHK(cudaMemset(d_XXT_u2, 0, tri_u32_bytes_u2));
            dim3 block_syrk_u2(TILE_N_U2, TILE_M_U2);
            dim3 grid_syrk_u2(div_up(m, block_syrk_u2.x), div_up(m, block_syrk_u2.y));
            XXT_Manual_U2<<<grid_syrk_u2, block_syrk_u2>>>(d_matrix_u2, d_XXT_u2, m, packed_cols_u2);

            // 3. Calcular Matriz de Distancias de la sección
            CUDA_CHK(cudaMemset(d_distances_u2, 0, tri_float_bytes_u2));
            dim3 block_dist_u2(16, 16);
            dim3 grid_dist_u2(div_up(m, block_dist_u2.x), div_up(m, block_dist_u2.y));
            CalculateDistanceU2<<<grid_dist_u2, block_dist_u2>>>(d_XXT_u2, d_norms_u2, d_distances_u2, m);
            CUDA_CHK(cudaDeviceSynchronize());
        }
        verificar_contra_cublas(handle, "U2", d_distances_u2, d_Dpacked_ref, tri_elems);

        cudaFree(d_matrix_u2);
        cudaFree(d_norms_u2);
        cudaFree(d_XXT_u2);
        free(h_matrix_u2);

        // Conservar 'd_distances_u2' para futura validación general
        cudaFree(d_distances_u2);
    }

    // ===================================================================
    // SECCIÓN OTROS, NAIVE , ETC. (HERE)
    // ===================================================================
    CUDA_CHK(cudaFree(d_Dpacked_ref));
    CUBLAS_CHK(cublasDestroy(handle));

    return 0;
}
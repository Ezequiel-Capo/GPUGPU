#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <nvtx3/nvToolsExt.h>

using namespace nvcuda;

// Convencion de dimensiones:
//   X tiene forma m x n
//     m = individuos / filas
//     n = SNPs / columnas
//
// SYMRK: el resultado es m x m y simetrico, porque:
//   XXT[i,j] = dot(X[i,*], X[j,*]) = XXT[j,i]
//
// Por esa simetria calculamos y guardamos solo el triangulo superior en formato 
// "empaquetado" 1D, ahorrando casi un 50% de memoria global.
//
// Cada warp ejecuta una operacion WMMA 16x16x16:
//   X_frag = A_frag  = 16 filas de X por 16 SNPs
//   XT_frag = B_frag  = 16 filas de X por 16 SNPs, leido como transpuesto
//   C_frag += A_frag * B_frag^T
//
// WARP_TILE_M y WARP_TILE_N eligen cuantos subtiles 16x16 hay por bloque.
#define WARP_TILE_M 2
#define WARP_TILE_N 2
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32
#define TILE_M (WARP_TILE_M * WMMA_M)
#define TILE_N (WARP_TILE_N * WMMA_N)
#define WARPS_PER_BLOCK (WARP_TILE_M * WARP_TILE_N)
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * WARP_SIZE)

#define SHMEM_K 32
#define SHMEM_C_N (TILE_N + 8)

#if (WARP_TILE_M < 1) || (WARP_TILE_N < 1)
#error "WARP_TILE_M y WARP_TILE_N deben ser al menos 1."
#endif

#if THREADS_PER_BLOCK > 1024
#error "Demasiados warps por bloque: THREADS_PER_BLOCK no puede superar 1024."
#endif

#if SHMEM_K < WMMA_K
#error "SHMEM_K debe ser al menos WMMA_K."
#endif

#if SHMEM_C_N < TILE_N
#error "SHMEM_C_N debe ser al menos TILE_N."
#endif

#define CUDA_CHK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while (0)

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

static inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// -------------------------------------------------------------------------
// Función para mapear índices (r, c) de una matriz simétrica (r <= c)
// a un índice 1D que guarda exclusivamente el triángulo superior.
static __device__ __host__ inline size_t get_packed_index(size_t r, size_t c, size_t m) {
    return r * m - (r * (r + 1)) / 2 + c;
}

// EUCLIDEAN DISTANCE
// D^2(i,j) = ||X_i||^2 + ||X_j||^2 - 2 * dot(X_i, X_j)
// XXT ya contiene dot(X_i, X_j). 
__global__ void CalculateDistance(const int32_t *XXT, const int *norms,
                                  int *distances, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < m && row <= col) {
        size_t packed_idx = get_packed_index(row, col, m);
        int xxt = XXT[packed_idx];
        distances[packed_idx] = norms[row] + norms[col] - 2 * xxt;
    }
}

// XXT = X * X^T usando Tensor Cores.
__global__ void XXT_WMMA_Shared(const uint8_t *X, int32_t *C, int m, int n) {

    int row_base = blockIdx.y * TILE_M;
    int col_base = blockIdx.x * TILE_N;

    // col < row. Como solo guardamos triangulo superior
    if (col_base + TILE_N <= row_base) return;

    int warp_id = threadIdx.x / WARP_SIZE;
    int warp_tile_row = warp_id / WARP_TILE_N;
    int warp_tile_col = warp_id % WARP_TILE_N;

    int subtile_row = row_base + warp_tile_row * WMMA_M;
    int subtile_col = col_base + warp_tile_col * WMMA_N;

    bool compute_subtile = (subtile_col + WMMA_N > subtile_row);

    __shared__ __align__(32) uint8_t a_tile[TILE_M][SHMEM_K];
    __shared__ __align__(32) uint8_t b_tile[TILE_N][SHMEM_K];
    __shared__ __align__(32) int32_t c_tile[TILE_M][SHMEM_C_N];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, unsigned char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, unsigned char, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_frag;

    wmma::fill_fragment(c_frag, 0);

    for (int k0 = 0; k0 < n; k0 += WMMA_K) {
        for (int idx = threadIdx.x; idx < TILE_M * WMMA_K; idx += blockDim.x) {
            int local_row = idx / WMMA_K;
            int local_k = idx % WMMA_K;
            int global_k = k0 + local_k;
            int a_row = row_base + local_row;

            a_tile[local_row][local_k] = (a_row < m && global_k < n) ? X[a_row * n + global_k] : 0;
        }

        for (int idx = threadIdx.x; idx < TILE_N * WMMA_K; idx += blockDim.x) {
            int local_row = idx / WMMA_K;
            int local_k = idx % WMMA_K;
            int global_k = k0 + local_k;
            int b_row = col_base + local_row;

            b_tile[local_row][local_k] = (b_row < m && global_k < n) ? X[b_row * n + global_k] : 0;
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
        int local_row = warp_tile_row * WMMA_M;
        int local_col = warp_tile_col * WMMA_N;
        wmma::store_matrix_sync(&c_tile[local_row][local_col], c_frag, SHMEM_C_N, wmma::mem_row_major);
    }

    __syncthreads();

    // Volcado a memoria global (modificado para usar formato empaquetado 1D)
    for (int idx = threadIdx.x; idx < TILE_M * TILE_N; idx += blockDim.x) {
        int local_row = idx / TILE_N;
        int local_col = idx % TILE_N;
        int global_row = row_base + local_row;
        int global_col = col_base + local_col;

        if (global_row < m && global_col < m && global_row <= global_col) {
            size_t packed_idx = get_packed_index(global_row, global_col, m);
            C[packed_idx] = c_tile[local_row][local_col];
        }
    }
}

// NORMAS
__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void CalculateNormVector(const uint8_t *matrix, int *norms,
                                    int m, int n) {
    int row = blockIdx.x;
    if (row >= m) return;

    int local_norm = 0;
    for (int col = threadIdx.x; col < n; col += blockDim.x) {
        int v = matrix[row * n + col];
        local_norm += v * v;
    }

    local_norm = warpReduceSum(local_norm);

    __shared__ int warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = (blockDim.x + warpSize - 1) / warpSize;

    if (lane == 0)
        warp_sums[warp_id] = local_norm;

    __syncthreads();

    if (warp_id == 0) {
        int sum = (lane < warps_per_block) ? warp_sums[lane] : 0;
        sum = warpReduceSum(sum);

        if (lane == 0)
            norms[row] = sum;
    }
}

void generate_genomic_matrix(uint8_t *matrix, int m, int n) {
    srand(42);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = rand() % 3; // 0, 1, 2
        }
    }
}

void print_matrix_u8(const uint8_t *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%u ", (unsigned)matrix[i * cols + j]);
        }
        printf("\n");
    }
}

bool validate_small_case(const uint8_t *X, const int32_t *XXT, const int *norms, int m, int n) {
    for (int i = 0; i < m; i++) {
        int ref_norm = 0;
        for (int k = 0; k < n; k++) {
            int v = X[i * n + k];
            ref_norm += v * v;
        }
        if (norms[i] != ref_norm) {
            printf("Error norma fila %d: GPU=%d CPU=%d\n", i, norms[i], ref_norm);
            return false;
        }

        for (int j = i; j < m; j++) {
            int32_t ref = 0;
            for (int k = 0; k < n; k++) 
                ref += (int32_t)X[i * n + k] * (int32_t)X[j * n + k];
            
            // Evaluamos con el índice empaquetado
            size_t packed_idx = get_packed_index(i, j, m);
            if (XXT[packed_idx] != ref) {
                printf("Error XXT(%d,%d): GPU=%d CPU=%d\n", i, j, XXT[packed_idx], ref);
                return false;
            }
        }
    }
    return true;
}

// NUEVO: Imprimir reconstruyendo la matriz cuadrada desde el formato empaquetado
void print_packed_matrix_i32(const int32_t *matrix, int m) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            if (i <= j) {
                size_t packed_idx = get_packed_index(i, j, m);
                printf("%d ", matrix[packed_idx]);
            } else {
                printf("0 "); // Triángulo inferior es 0
            }
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int m = (1 << 10);  // individuos
    int n = (1 << 15);  // SNPs

    if (argc >= 3) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }

    if (m <= 0 || n <= 0) {
        fprintf(stderr, "Uso: %s [individuos m] [SNPs n]\n", argv[0]);
        return 1;
    }

    size_t matrix_elems = (size_t)m * (size_t)n;
    size_t matrix_bytes = matrix_elems * sizeof(uint8_t);
    
    // NUEVO: Cantidad de elementos del triángulo superior (empaquetado 1D)
    size_t tri_elems = (size_t)m * (m + 1) / 2;
    size_t tri_i32_bytes = tri_elems * sizeof(int32_t);

    uint8_t *h_matrix = (uint8_t*)malloc(matrix_bytes);

    if (!h_matrix) { 
        fprintf(stderr, "No hay memoria de host suficiente.\n");
        return 1;
    }

    uint8_t *d_matrix = NULL;
    int *d_norms = NULL;
    int32_t *d_XXT = NULL;
    int *d_distances = NULL;

    // Device: d_XXT y d_distances ahora reservan solo para tri_i32_bytes
    CUDA_CHK(cudaMalloc(&d_matrix, matrix_bytes));
    CUDA_CHK(cudaMalloc(&d_norms, (size_t)m * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_XXT, tri_i32_bytes));
    CUDA_CHK(cudaMalloc(&d_distances, tri_i32_bytes));

    printf("Generando matriz genomica X (%d individuos x %d SNPs)...\n", m, n);
    generate_genomic_matrix(h_matrix, m, n);

    // WARM UP:
    CUDA_CHK(cudaMemcpy(d_matrix, h_matrix, matrix_bytes, cudaMemcpyHostToDevice));
    if (m <= 64 && n <= 128) {
        printf("\nX (debug):\n");
        print_matrix_u8(h_matrix, m, n);
    }

    // NORMAS -----------------------------------------------------
    dim3 block_norms(256);
    dim3 grid_norms(m);
    CalculateNormVector<<<grid_norms, block_norms>>>(d_matrix, d_norms, m, n);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    printf("Normas finalizado.\n");
    CUDA_CHK(cudaMemset(d_XXT, 0, tri_i32_bytes));

    // X*X^T -----------------------------------------------------
    dim3 block_syrk(THREADS_PER_BLOCK);
    dim3 grid_syrk(div_up(m, TILE_N), div_up(m, TILE_M));
    XXT_WMMA_Shared<<<grid_syrk, block_syrk>>>(d_matrix, d_XXT, m, n);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaMemset(d_distances, 0, tri_i32_bytes));

    printf("SYRK con tensor cores finalizado\n");

    // DISTANCIAS EUCLIDEAS -----------------------------------------------------
    dim3 block_dist(16, 16);
    dim3 grid_dist(div_up(m, block_dist.x), div_up(m, block_dist.y));
    CalculateDistance<<<grid_dist, block_dist>>>(d_XXT, d_norms, d_distances, m);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    
    for (int i = 0; i < 10; i++) {
        CUDA_CHK(cudaMemcpy(d_matrix, h_matrix, matrix_bytes, cudaMemcpyHostToDevice));

        // NORMAS -----------------------------------------------------
        nvtxRangePushA("Norms");
        CalculateNormVector<<<grid_norms, block_norms>>>(d_matrix, d_norms, m, n);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();

        CUDA_CHK(cudaMemset(d_XXT, 0, tri_i32_bytes));

        // X*X^T -----------------------------------------------------
        nvtxRangePushA("XXT");
        XXT_WMMA_Shared<<<grid_syrk, block_syrk>>>(d_matrix, d_XXT, m, n);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();

        CUDA_CHK(cudaMemset(d_distances, 0, tri_i32_bytes));
        
        // DISTANCIAS EUCLIDEAS -----------------------------------------------------
        nvtxRangePushA("CalculateDistance");
        CalculateDistance<<<grid_dist, block_dist>>>(d_XXT, d_norms, d_distances, m);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop(); 
    }
    
    // DEBUGS, casos pequeños
    if (m <= 64 && n <= 1024) {
        int *h_norms = (int*)malloc((size_t)m * sizeof(int));
        int32_t *h_XXT = NULL;
        CUDA_CHK(cudaMemcpy(h_norms, d_norms, (size_t)m * sizeof(int), cudaMemcpyDeviceToHost));
        if (!h_norms) {
            fprintf(stderr, "No hay memoria de host para validar normas.\n");
        } else {
            printf("Normas (debug):\n");
            for (int i = 0; i < m; i++) {
                printf("%d ", h_norms[i]);
            }
            printf("\n");
        }
        
        h_XXT = (int32_t*)malloc(tri_i32_bytes); // Cambiado
        if (!h_XXT) {
            fprintf(stderr, "No hay memoria de host para validar XXT.\n");
        } else {
            CUDA_CHK(cudaMemcpy(h_XXT, d_XXT, tri_i32_bytes, cudaMemcpyDeviceToHost)); // Cambiado

            bool ok = validate_small_case(h_matrix, h_XXT, h_norms, m, n);
            printf("Validacion CPU/GPU: %s\n", ok ? "OK" : "FALLO");
        }
        
        int *h_distances = (int*)malloc(tri_i32_bytes); // Cambiado
        if (!h_distances) {
            fprintf(stderr, "No hay memoria de host para validar distancias.\n");
        } else {
            fprintf(stderr, "\nDistancias euclideas al cuadrado (debug):\n");
            CUDA_CHK(cudaMemcpy(h_distances, d_distances, tri_i32_bytes, cudaMemcpyDeviceToHost)); // Cambiado
            
            // Usamos la nueva función para imprimir reconstruyendo el 2D
            print_packed_matrix_i32(h_distances, m); 
        }
        free(h_norms);
        if (h_XXT) free(h_XXT);
        if (h_distances) free(h_distances);
    }

    cudaFree(d_matrix);
    cudaFree(d_norms);
    cudaFree(d_XXT);
    cudaFree(d_distances);
    free(h_matrix);

    return 0;
}
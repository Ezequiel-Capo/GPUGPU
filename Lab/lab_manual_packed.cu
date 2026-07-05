#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

// Convencion de dimensiones:
//   X tiene forma m x n
//     m = individuos / filas
//     n = SNPs / columnas
//
// Formato empaquetado de X:
//   Cada genotipo usa 2 bits:
//     0 -> 00
//     1 -> 01
//     2 -> 10
//
//   Cada byte guarda 4 genotipos de la misma fila:
//     bits 1:0 -> SNP 4*b + 0
//     bits 3:2 -> SNP 4*b + 1
//     bits 5:4 -> SNP 4*b + 2
//     bits 7:6 -> SNP 4*b + 3

#define VALUES_PER_BYTE 4
#define BITS_PER_VALUE 2
#define PACKED_VALUE_MASK 0x03u

#define SYMRK_BLOCK_X 16
#define SYMRK_BLOCK_Y 16
#define TILE_M 16
#define TILE_N 16
#define TILE_K 32

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

__host__ __device__ static inline uint8_t unpack_genotype_from_byte(uint8_t packed_byte, int offset_in_byte) {
    return (packed_byte >> (offset_in_byte * BITS_PER_VALUE)) & PACKED_VALUE_MASK;
}

__host__ __device__ static inline uint8_t get_packed_genotype(const uint8_t *matrix, int packed_cols, int row, int col) {
    size_t byte_index = (size_t)row * (size_t)packed_cols + (size_t)(col / VALUES_PER_BYTE);
    int offset = col % VALUES_PER_BYTE;
    return unpack_genotype_from_byte(matrix[byte_index], offset);
}

static inline void set_packed_genotype(uint8_t *matrix, int packed_cols, int row, int col, uint8_t value) {
    size_t byte_index = (size_t)row * (size_t)packed_cols + (size_t)(col / VALUES_PER_BYTE);
    int shift = (col % VALUES_PER_BYTE) * BITS_PER_VALUE;
    uint8_t mask = (uint8_t)(PACKED_VALUE_MASK << shift);
    uint8_t encoded = (uint8_t)((value & PACKED_VALUE_MASK) << shift);
    matrix[byte_index] = (uint8_t)((matrix[byte_index] & ~mask) | encoded);
}

// Convierte 1 byte (4 genotipos de 2 bits) en un entero de 32 bits (4 componentes de 8 bits) listo para __dp4a
__host__ __device__ static inline int unpack_2bit_to_4x8(uint8_t packed_byte) {
    int g0 = packed_byte & 0x03u;
    int g1 = (packed_byte >> 2) & 0x03u;
    int g2 = (packed_byte >> 4) & 0x03u;
    int g3 = (packed_byte >> 6) & 0x03u;
    return g0 | (g1 << 8) | (g2 << 16) | (g3 << 24);
}

// EUCLIDEAN DISTANCE
// D^2(i,j) = ||X_i||^2 + ||X_j||^2 - 2 * dot(X_i, X_j)
__global__ void CalculateDistance(const int32_t *XXT, const int *norms,
                                  int *distances, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < m && row <= col) {
        int xxt = XXT[(size_t)row * (size_t)m + (size_t)col];
        distances[(size_t)row * (size_t)m + (size_t)col] = norms[row] + norms[col] - 2 * xxt;
    }
}

// XXT = X * X^T manual sobre datos empaquetados usando __dp4a intrínseco.
__global__ void XXTManualPacked(const uint8_t *X, int32_t *C, int m, int n, int packed_cols)
{
    int row_base = blockIdx.y * TILE_M;
    int col_base = blockIdx.x * TILE_N;

    // Saltar los tiles estrictamente debajo de la diagonal principal
    if (col_base + TILE_N <= row_base)
        return;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = row_base + ty;
    int col = col_base + tx;

    // Se cambia a 'int' (32 bits). El +1 evita Bank Conflicts perfectamente al mapear palabras completas.
    __shared__ int As[TILE_M][TILE_K + 1];
    __shared__ int Bs[TILE_N][TILE_K + 1];

    int32_t acc = 0;

    for (int k0 = 0; k0 < packed_cols; k0 += TILE_K) {
        // Cada hilo carga y expande simultáneamente a memoria compartida
        for (int k = tx; k < TILE_K; k += TILE_N) {

            int idx = k0 + k;

            // Tile A: Desempaquetado al vuelo durante la carga
            if (row < m && idx < packed_cols)
                As[ty][k] = unpack_2bit_to_4x8(X[(size_t)row * packed_cols + idx]);
            else
                As[ty][k] = 0;

            // Tile B: Desempaquetado al vuelo durante la carga
            int b_row = col_base + ty;
            
            if (b_row < m && idx < packed_cols)
                Bs[ty][k] = unpack_2bit_to_4x8(X[(size_t)b_row * packed_cols + idx]);
            else
                Bs[ty][k] = 0;
        }

        __syncthreads();

    #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            // El hardware calcula los 4 productos internos y los acumula en una sola instrucción
            acc = __dp4a(As[ty][k], Bs[tx][k], acc);
        }

        __syncthreads();
    }

    // Escribir solo el triángulo superior
    if (row < m && col < m && row <= col)
        C[(size_t)row * m + col] = acc;
}

// NORMAS OPTIMIZADAS CON __dp4a
__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void CalculateNormVectorPacked(const uint8_t *matrix, int *norms, int m, int n, int packed_cols) {
    int row = blockIdx.x;
    if (row >= m) return;

    const uint8_t *row_ptr = matrix + (size_t)row * (size_t)packed_cols;
    int local_norm = 0;

    for (int byte_col = threadIdx.x; byte_col < packed_cols; byte_col += blockDim.x) {
        uint8_t packed_byte = row_ptr[byte_col];
        
        // Convertimos el byte e incrementamos la norma con __dp4a
        int unpacked = unpack_2bit_to_4x8(packed_byte);
        local_norm = __dp4a(unpacked, unpacked, local_norm);
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

void generate_genomic_matrix_packed(uint8_t *matrix, int m, int n, int packed_cols) {
    srand(42);
    memset(matrix, 0, (size_t)m * (size_t)packed_cols * sizeof(uint8_t));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            set_packed_genotype(matrix, packed_cols, i, j, (uint8_t)(rand() % 3));
        }
    }
}

void print_matrix_packed_u8(const uint8_t *matrix, int rows, int cols, int packed_cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%u ", (unsigned)get_packed_genotype(matrix, packed_cols, i, j));
        }
        printf("\n");
    }
}

bool validate_small_case_packed(const uint8_t *X, const int32_t *XXT, const int *norms, int m, int n, int packed_cols) {
    for (int i = 0; i < i; i++) { /* Corrección semántica loop interna de validación */ }
        for (int i = 0; i < m; i++) {
            int ref_norm = 0;
            for (int k = 0; k < n; k++) {
                int v = get_packed_genotype(X, packed_cols, i, k);
                ref_norm += v * v;
            }

            if (norms[i] != ref_norm) {
                printf("Error norma fila %d: GPU=%d CPU=%d\n", i, norms[i], ref_norm);
                return false;
            }

            for (int j = i; j < m; j++) {
                int32_t ref = 0;
                for (int k = 0; k < n; k++) {
                    int a = get_packed_genotype(X, packed_cols, i, k);
                    int b = get_packed_genotype(X, packed_cols, j, k);
                    ref += (int32_t)a * (int32_t)b;
                }

                if (XXT[(size_t)i * (size_t)m + (size_t)j] != ref) {
                    printf("Error XXT(%d,%d): GPU=%d CPU=%d\n",
                        i, j, XXT[(size_t)i * (size_t)m + (size_t)j], ref);
                    return false;
                }
            }
    }
    return true;
}

void print_matrix_i32(const int32_t *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[(size_t)i * (size_t)cols + (size_t)j]);
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

    int packed_cols = div_up(n, VALUES_PER_BYTE);
    size_t unpacked_matrix_bytes = (size_t)m * (size_t)n * sizeof(uint8_t);
    size_t packed_matrix_bytes = (size_t)m * (size_t)packed_cols * sizeof(uint8_t);
    size_t square_elems = (size_t)m * (size_t)m;
    size_t square_i32_bytes = square_elems * sizeof(int32_t);
    size_t square_int_bytes = square_elems * sizeof(int);

    uint8_t *h_matrix = (uint8_t*)malloc(packed_matrix_bytes);

    if (!h_matrix) {
        fprintf(stderr, "No hay memoria de host suficiente.\n");
        return 1;
    }

    uint8_t *d_matrix = NULL;
    int *d_norms = NULL;
    int32_t *d_XXT = NULL;
    int *d_distances = NULL;

    CUDA_CHK(cudaMalloc(&d_matrix, packed_matrix_bytes));
    CUDA_CHK(cudaMalloc(&d_norms, (size_t)m * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_XXT, square_i32_bytes));
    CUDA_CHK(cudaMalloc(&d_distances, square_int_bytes));

    printf("Generando matriz genomica X (%d individuos x %d SNPs)...\n", m, n);
    printf("Formato empaquetado: %d bytes por fila, %zu bytes total (sin empaquetar: %zu bytes).\n", packed_cols, packed_matrix_bytes, unpacked_matrix_bytes);
    generate_genomic_matrix_packed(h_matrix, m, n, packed_cols);

    CUDA_CHK(cudaMemcpy(d_matrix, h_matrix, packed_matrix_bytes, cudaMemcpyHostToDevice));

    if (m <= 64 && n <= 128) {
        printf("\nX desempaquetada (debug):\n");
        print_matrix_packed_u8(h_matrix, m, n, packed_cols);
    }

    dim3 block_norms(256);
    dim3 grid_norms(m);
    CalculateNormVectorPacked<<<grid_norms, block_norms>>>(d_matrix, d_norms, m, n, packed_cols);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaMemset(d_XXT, 0, square_i32_bytes));

    dim3 block_syrk(SYMRK_BLOCK_X, SYMRK_BLOCK_Y);
    dim3 grid_syrk(div_up(m, block_syrk.x), div_up(m, block_syrk.y));
    XXTManualPacked<<<grid_syrk, block_syrk>>>(d_matrix, d_XXT, m, n, packed_cols);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaMemset(d_distances, 0, square_int_bytes));

    dim3 block_dist(16, 16);
    dim3 grid_dist(div_up(m, block_dist.x), div_up(m, block_dist.y));
    CalculateDistance<<<grid_dist, block_dist>>>(d_XXT, d_norms, d_distances, m);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());
    // Ejecuciones de prueba y  Benchmarking loop

    for(int i = 0; i < 10; i++) {
        dim3 block_norms(256);
        dim3 grid_norms(m);
        nvtxRangePushA("Norms");
        CalculateNormVectorPacked<<<grid_norms, block_norms>>>(d_matrix, d_norms, m, n, packed_cols);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();
        CUDA_CHK(cudaMemset(d_XXT, 0, square_i32_bytes));

        dim3 block_syrk(SYMRK_BLOCK_X, SYMRK_BLOCK_Y);
        dim3 grid_syrk(div_up(m, block_syrk.x), div_up(m, block_syrk.y));
        nvtxRangePushA("XXT");
        XXTManualPacked<<<grid_syrk, block_syrk>>>(d_matrix, d_XXT, m, n, packed_cols);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();
        CUDA_CHK(cudaMemset(d_distances, 0, square_int_bytes));

        dim3 block_dist(16, 16);
        dim3 grid_dist(div_up(m, block_dist.x), div_up(m, block_dist.y));
        nvtxRangePushA("CalculateDistance");
        CalculateDistance<<<grid_dist, block_dist>>>(d_XXT, d_norms, d_distances, m);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();
    }

    printf("Cálculos completados con éxito.\n");

    // DEBUGS, casos pequenos
    if (m <= 64 && n <= 1024) {
        int *h_norms = (int*)malloc((size_t)m * sizeof(int));
        int32_t *h_XXT = (int32_t*)malloc(square_i32_bytes);
        CUDA_CHK(cudaMemcpy(h_norms, d_norms, (size_t)m * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHK(cudaMemcpy(h_XXT, d_XXT, square_i32_bytes, cudaMemcpyDeviceToHost));

        printf("Normas (debug):\n");
        for (int i = 0; i < m; i++) printf("%d ", h_norms[i]);
        printf("\n");

        bool ok = validate_small_case_packed(h_matrix, h_XXT, h_norms, m, n, packed_cols);
        printf("Validacion CPU/GPU: %s\n", ok ? "OK" : "FALLO");

        int *h_distances = (int*)malloc(square_int_bytes);
        if (h_distances) {
            printf("\nDistancias euclideas al cuadrado (debug):\n");
            CUDA_CHK(cudaMemcpy(h_distances, d_distances, square_int_bytes, cudaMemcpyDeviceToHost));
            print_matrix_i32((const int32_t*)h_distances, m, m);
            free(h_distances);
        }
        free(h_norms);
        free(h_XXT);
    }

    cudaFree(d_matrix);
    cudaFree(d_norms);
    cudaFree(d_XXT);
    cudaFree(d_distances);
    free(h_matrix);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

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
//
// SYMRK manual:
//   C = X * X^T, solo triangulo superior.
//   Un hilo calcula un elemento C[row,col] y recorre la fila empaquetada.
//   No usa WMMA ni Tensor Cores.

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

__device__ static inline int dot4_packed_bytes(uint8_t a_byte, uint8_t b_byte) {
    int acc = 0;

#pragma unroll
    for (int offset = 0; offset < VALUES_PER_BYTE; offset++) {
        int a = unpack_genotype_from_byte(a_byte, offset);
        int b = unpack_genotype_from_byte(b_byte, offset);
        acc += a * b;
    }

    return acc;
}


// EUCLIDEAN DISTANCE
// D^2(i,j) = ||X_i||^2 + ||X_j||^2 - 2 * dot(X_i, X_j)
// XXT ya contiene dot(X_i, X_j). Como las distancias tambien son simetricas,
// solo escribimos row <= col. d_distances se inicializa con cudaMemset, asi que
// el triangulo inferior queda en cero.
__global__ void CalculateDistance(const int32_t *XXT, const int *norms,
                                  int *distances, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < m && row <= col) {
        int xxt = XXT[(size_t)row * (size_t)m + (size_t)col];
        distances[(size_t)row * (size_t)m + (size_t)col] = norms[row] + norms[col] - 2 * xxt;
    }
}

// XXT = X * X^T manual sobre datos empaquetados.
// Cada hilo calcula una celda del triangulo superior. En cada iteracion lee un
// byte de la fila row y un byte de la fila col, desempaqueta 4 exclusivamente
// y acumula sus productos.
// XXT = X * X^T manual sobre datos empaquetados.
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

    // +1 para evitar bank conflicts
    __shared__ uint8_t As[TILE_M][TILE_K + 1];
    __shared__ uint8_t Bs[TILE_N][TILE_K + 1];

    int32_t acc = 0;

    for (int k0 = 0; k0 < packed_cols; k0 += TILE_K) {

        // Cada hilo carga 2 bytes
        for (int k = tx; k < TILE_K; k += TILE_N) {

            int idx = k0 + k;

            // Tile A: usa 'ty' para la fila y 'k' (derivado de tx) para la columna
            if (row < m && idx < packed_cols)
                As[ty][k] = X[(size_t)row * packed_cols + idx];
            else
                As[ty][k] = 0;

            // Tile B: CORRECCIÓN AQUÍ
            // Usamos 'ty' para iterar cooperativamente sobre las filas del Tile B,
            // garantizando que se cargue la cuadrícula completa de 16x32.
            int b_row = col_base + ty;
            
            if (b_row < m && idx < packed_cols)
                Bs[ty][k] = X[(size_t)b_row * packed_cols + idx];
            else
                Bs[ty][k] = 0;
        }

        __syncthreads();

    #pragma unroll
        for (int k = 0; k < TILE_K; k++)
            acc += dot4_packed_bytes(As[ty][k], Bs[tx][k]);

        __syncthreads();
    }

    // Escribir solo el triángulo superior
    if (row < m && col < m && row <= col)
        C[(size_t)row * m + col] = acc;
}

// NORMAS
// norms[row] = ||X[row,*]||^2 = sum_k X[row,k]^2
//
// Cada bloque calcula una fila de X empaquetada. Los hilos recorren bytes con
// stride blockDim.x, suman hasta 4 cuadrados por byte, y luego reducen:
//   1. dentro de cada warp con shuffle instructions;
//   2. entre warps usando shared memory;
//   3. el lane 0 del warp 0 escribe el resultado final.
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

        local_norm += dot4_packed_bytes(packed_byte, packed_byte);
        
    }

    local_norm = warpReduceSum(local_norm);

    // Maximo 1024 hilos por bloque => maximo 32 warps.
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

    // Datos sinteticos: cada genotipo toma valores 0, 1 o 2 y ocupa 2 bits.
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
    // Validacion CPU para casos chicos. Solo revisa el triangulo superior porque
    // esa es la parte que el kernel define como salida valida.
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
        free(h_matrix);
        return 1;
    }

    uint8_t *d_matrix = NULL;
    int *d_norms = NULL;
    int32_t *d_XXT = NULL;
    int *d_distances = NULL;

    // Device: d_XXT y d_distances se reservan como m x m completas.
    // Solo se define el triangulo superior.
    CUDA_CHK(cudaMalloc(&d_matrix, packed_matrix_bytes));
    CUDA_CHK(cudaMalloc(&d_norms, (size_t)m * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_XXT, square_i32_bytes));
    CUDA_CHK(cudaMalloc(&d_distances, square_int_bytes));

    printf("Generando matriz genomica X (%d individuos x %d SNPs)...\n", m, n);
    printf("Formato empaquetado: %d bytes por fila, %zu bytes total (sin empaquetar: %zu bytes).\n",packed_cols, packed_matrix_bytes, unpacked_matrix_bytes);
    generate_genomic_matrix_packed(h_matrix, m, n, packed_cols);

    CUDA_CHK(cudaMemcpy(d_matrix, h_matrix, packed_matrix_bytes, cudaMemcpyHostToDevice));

    if (m <= 64 && n <= 128) {
        printf("\nX desempaquetada (debug):\n");
        print_matrix_packed_u8(h_matrix, m, n, packed_cols);
    }

    // NORMAS -----------------------------------------------------
    dim3 block_norms(256);
    dim3 grid_norms(m);
    CalculateNormVectorPacked<<<grid_norms, block_norms>>>(d_matrix, d_norms, m, n, packed_cols);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    printf("Normas finalizado.\n");
    CUDA_CHK(cudaMemset(d_XXT, 0, square_i32_bytes));

    // X*X^T -----------------------------------------------------
    // XXT manual:
    //   grid.x recorre columnas de C;
    //   grid.y recorre filas de C;
    //   cada hilo calcula un C[row,col] leyendo X empaquetada.
    dim3 block_syrk(SYMRK_BLOCK_X, SYMRK_BLOCK_Y);
    dim3 grid_syrk(div_up(m, block_syrk.x), div_up(m, block_syrk.y));
    XXTManualPacked<<<grid_syrk, block_syrk>>>(d_matrix, d_XXT, m, n, packed_cols);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaMemset(d_distances, 0, square_int_bytes));

    printf("XXT manual con datos empaquetados finalizado.\n");

    // DISTANCIAS EUCLIDEAS -----------------------------------------------------
    dim3 block_dist(16, 16);
    dim3 grid_dist(div_up(m, block_dist.x), div_up(m, block_dist.y));
    CalculateDistance<<<grid_dist, block_dist>>>(d_XXT, d_norms, d_distances, m);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    // DEBUGS, casos pequenos
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

        h_XXT = (int32_t*)malloc(square_i32_bytes);
        if (!h_XXT) {
            fprintf(stderr, "No hay memoria de host para validar XXT.\n");
        } else {
            CUDA_CHK(cudaMemcpy(h_XXT, d_XXT, square_i32_bytes, cudaMemcpyDeviceToHost));

            bool ok = validate_small_case_packed(h_matrix, h_XXT, h_norms, m, n, packed_cols);
            printf("Validacion CPU/GPU: %s\n", ok ? "OK" : "FALLO");
        }

        int *h_distances = (int*)malloc(square_int_bytes);
        if (!h_distances) {
            fprintf(stderr, "No hay memoria de host para validar distancias.\n");
        } else {
            fprintf(stderr, "\nDistancias euclideas al cuadrado (debug):\n");
            CUDA_CHK(cudaMemcpy(h_distances, d_distances, square_int_bytes, cudaMemcpyDeviceToHost));
            print_matrix_i32((const int32_t*)h_distances, m, m);
        }

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

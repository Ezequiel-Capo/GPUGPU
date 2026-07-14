#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // Incluido para el tipo half
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

// Mapeo de 2D (row, col) a 1D Vector (Triangular Superior)
__host__ __device__ static inline size_t get_triangular_index(int row, int col, int m) {
    if (row > col) {
        int tmp = row; 
        row = col; 
        col = tmp;
    }
    return (size_t)row * m - ((size_t)row * (row + 1)) / 2 + col;
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

__device__ static inline uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__global__ void GenerateGenomicMatrixPackedKernel(uint8_t *matrix, int m, int n, int packed_cols, uint32_t seed) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int byte_col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || byte_col >= packed_cols) return;

    int base_col = byte_col * VALUES_PER_BYTE;
    uint8_t packed = 0;

#pragma unroll
    for (int lane = 0; lane < VALUES_PER_BYTE; lane++) {
        int col = base_col + lane;
        uint8_t value = 0;

        if (col < n) {
            uint32_t key = seed ^ ((uint32_t)row * 0x9e3779b9u) ^ ((uint32_t)col * 0x85ebca6bu);
            value = (uint8_t)(hash_u32(key) % 3u);
        }

        packed |= (uint8_t)((value & PACKED_VALUE_MASK) << (lane * BITS_PER_VALUE));
    }

    matrix[(size_t)row * (size_t)packed_cols + (size_t)byte_col] = packed;
}

void generate_genomic_matrix_packed_device(uint8_t *d_matrix, int m, int n, int packed_cols) {
    dim3 block(32, 8);
    dim3 grid(div_up(packed_cols, block.x), div_up(m, block.y));

    GenerateGenomicMatrixPackedKernel<<<grid, block>>>(d_matrix, m, n, packed_cols, 42u);
    CUDA_CHK(cudaGetLastError());
}

// Convierte 1 byte (4 genotipos de 2 bits) en un entero sin signo de 32 bits listo para __dp4a
__host__ __device__ static inline uint32_t unpack_2bit_to_4x8(uint8_t packed_byte) {
    uint32_t g0 = packed_byte & 0x03u;
    uint32_t g1 = (packed_byte >> 2) & 0x03u;
    uint32_t g2 = (packed_byte >> 4) & 0x03u;
    uint32_t g3 = (packed_byte >> 6) & 0x03u;
    return g0 | (g1 << 8) | (g2 << 16) | (g3 << 24);
}

// EUCLIDEAN DISTANCE (Float16 y Raíz Cuadrada)
__global__ void CalculateDistance(const uint32_t *XXT_packed, const uint32_t *norms,
                                  half *distances_packed, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < m && row <= col) {
        size_t idx = get_triangular_index(row, col, m);
        uint32_t xxt = XXT_packed[idx];
        uint32_t norm_row = norms[row];
        uint32_t norm_col = norms[col];
        
        uint32_t sum_norms = norm_row + norm_col;
        float dist_f = 0.0f;
        
        if (sum_norms > 2 * xxt) {
            dist_f = sqrtf((float)(sum_norms - 2 * xxt));
        }

        distances_packed[idx] = __float2half(dist_f);
    }
}

// XXT manual sobre datos empaquetados usando intrínsecas (guardado como uint32_t)
__global__ void XXTManualPacked(const uint8_t *X, uint32_t *C_packed, int m, int n, int packed_cols)
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

    __shared__ uint32_t As[TILE_M][TILE_K + 1];
    __shared__ uint32_t Bs[TILE_N][TILE_K + 1];

    uint32_t acc = 0;

    for (int k0 = 0; k0 < packed_cols; k0 += TILE_K) {
        for (int k = tx; k < TILE_K; k += TILE_N) {
            int idx = k0 + k;

            if (row < m && idx < packed_cols)
                As[ty][k] = unpack_2bit_to_4x8(X[(size_t)row * packed_cols + idx]);
            else
                As[ty][k] = 0;

            int b_row = col_base + ty;
            
            if (b_row < m && idx < packed_cols)
                Bs[ty][k] = unpack_2bit_to_4x8(X[(size_t)b_row * packed_cols + idx]);
            else
                Bs[ty][k] = 0;
        }

        __syncthreads();

    #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            // Utilizamos la sobrecarga para unsigned int que requiere arquitectura sm_61 o superior
            acc = __dp4a(As[ty][k], Bs[tx][k], acc);
        }

        __syncthreads();
    }

    if (row < m && col < m && row <= col) {
        size_t idx = get_triangular_index(row, col, m);
        C_packed[idx] = acc;
    }
}

__inline__ __device__ uint32_t warpReduceSum(uint32_t val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void CalculateNormVectorPacked(const uint8_t *matrix, uint32_t *norms, int m, int n, int packed_cols) {
    int row = blockIdx.x;
    if (row >= m) return;

    const uint8_t *row_ptr = matrix + (size_t)row * (size_t)packed_cols;
    uint32_t local_norm = 0;

    for (int byte_col = threadIdx.x; byte_col < packed_cols; byte_col += blockDim.x) {
        uint8_t packed_byte = row_ptr[byte_col];
        uint32_t unpacked = unpack_2bit_to_4x8(packed_byte);
        local_norm = __dp4a(unpacked, unpacked, local_norm);
    }

    local_norm = warpReduceSum(local_norm);

    __shared__ uint32_t warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = (blockDim.x + warpSize - 1) / warpSize;

    if (lane == 0)
        warp_sums[warp_id] = local_norm;

    __syncthreads();

    if (warp_id == 0) {
        uint32_t sum = (lane < warps_per_block) ? warp_sums[lane] : 0;
        sum = warpReduceSum(sum);

        if (lane == 0)
            norms[row] = sum;
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

// Adaptado para uint32_t
bool validate_small_case_packed(const uint8_t *X, const uint32_t *XXT_packed, const uint32_t *norms, int m, int n, int packed_cols) {
    for (int i = 0; i < m; i++) {
        uint32_t ref_norm = 0;
        for (int k = 0; k < n; k++) {
            uint32_t v = get_packed_genotype(X, packed_cols, i, k);
            ref_norm += v * v;
        }

        if (norms[i] != ref_norm) {
            printf("Error norma fila %d: GPU=%u CPU=%u\n", i, norms[i], ref_norm);
            return false;
        }

        for (int j = i; j < m; j++) {
            uint32_t ref = 0;
            for (int k = 0; k < n; k++) {
                uint32_t a = get_packed_genotype(X, packed_cols, i, k);
                uint32_t b = get_packed_genotype(X, packed_cols, j, k);
                ref += a * b;
            }

            size_t idx = get_triangular_index(i, j, m);
            if (XXT_packed[idx] != ref) {
                printf("Error XXT(%d,%d): GPU=%u CPU=%u\n", i, j, XXT_packed[idx], ref);
                return false;
            }
        }
    }
    return true;
}

// Reconstruye visualmente en float desde el arreglo comprimido half
void print_matrix_half(const half *matrix_packed, int m) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            size_t idx = get_triangular_index(i, j, m);
            printf("%6.2f ", __half2float(matrix_packed[idx]));
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
    
    // Tamaños comprimidos m x (m+1)/2 en vez de m x m
    size_t triangular_elems = (size_t)m * (m + 1) / 2;
    size_t tri_u32_bytes = triangular_elems * sizeof(uint32_t);
    size_t tri_half_bytes = triangular_elems * sizeof(half);

    uint8_t *h_matrix = NULL;

    uint8_t *d_matrix = NULL;
    uint32_t *d_norms = NULL;
    uint32_t *d_XXT_packed = NULL;
    half *d_distances_packed = NULL;

    CUDA_CHK(cudaMalloc(&d_matrix, packed_matrix_bytes));
    CUDA_CHK(cudaMalloc(&d_norms, (size_t)m * sizeof(uint32_t)));
    CUDA_CHK(cudaMalloc(&d_XXT_packed, tri_u32_bytes));
    CUDA_CHK(cudaMalloc(&d_distances_packed, tri_half_bytes));

    printf("Generando matriz genomica X (%d individuos x %d SNPs)...\n", m, n);
    printf("Matriz X empaquetada: %zu bytes (sin empaquetar: %zu bytes).\n", packed_matrix_bytes, unpacked_matrix_bytes);
    printf("Matriz Salida Comprimida Triangular: %zu elementos.\n", triangular_elems);
    
    nvtxRangePushA("GenX");
    generate_genomic_matrix_packed_device(d_matrix, m, n, packed_cols);
    CUDA_CHK(cudaDeviceSynchronize());
    nvtxRangePop();
    
    bool need_host_matrix = (m <= 64 && n <= 1024);
    if (need_host_matrix) {
        h_matrix = (uint8_t*)malloc(packed_matrix_bytes);
        if (!h_matrix) {
            fprintf(stderr, "No hay memoria de host suficiente para depuracion/validacion.\n");
            cudaFree(d_matrix);
            cudaFree(d_norms);
            cudaFree(d_XXT_packed);
            cudaFree(d_distances_packed);
            return 1;
        }
        CUDA_CHK(cudaMemcpy(h_matrix, d_matrix, packed_matrix_bytes, cudaMemcpyDeviceToHost));
    }

    if (m <= 64 && n <= 128 && h_matrix) {
        printf("\nX desempaquetada (debug):\n");
        print_matrix_packed_u8(h_matrix, m, n, packed_cols);
    }

    // Ejecuciones de prueba y Benchmarking loop
    for(int i = 0; i < 10; i++) {
        dim3 block_norms(256);
        dim3 grid_norms(m);
        nvtxRangePushA("Norms");
        CalculateNormVectorPacked<<<grid_norms, block_norms>>>(d_matrix, d_norms, m, n, packed_cols);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();
        
        CUDA_CHK(cudaMemset(d_XXT_packed, 0, tri_u32_bytes));

        dim3 block_syrk(SYMRK_BLOCK_X, SYMRK_BLOCK_Y);
        dim3 grid_syrk(div_up(m, block_syrk.x), div_up(m, block_syrk.y));
        nvtxRangePushA("XXT");
        XXTManualPacked<<<grid_syrk, block_syrk>>>(d_matrix, d_XXT_packed, m, n, packed_cols);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();
        
        CUDA_CHK(cudaMemset(d_distances_packed, 0, tri_half_bytes));

        dim3 block_dist(16, 16);
        dim3 grid_dist(div_up(m, block_dist.x), div_up(m, block_dist.y));
        nvtxRangePushA("CalculateDistance");
        CalculateDistance<<<grid_dist, block_dist>>>(d_XXT_packed, d_norms, d_distances_packed, m);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();
    }

    printf("Cálculos completados con éxito.\n");

    // DEBUGS, casos pequeños
    if (m <= 64 && n <= 1024 && h_matrix) {
        uint32_t *h_norms = (uint32_t*)malloc((size_t)m * sizeof(uint32_t));
        uint32_t *h_XXT_packed = (uint32_t*)malloc(tri_u32_bytes);
        CUDA_CHK(cudaMemcpy(h_norms, d_norms, (size_t)m * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHK(cudaMemcpy(h_XXT_packed, d_XXT_packed, tri_u32_bytes, cudaMemcpyDeviceToHost));

        printf("Normas (debug):\n");
        for (int i = 0; i < m; i++) printf("%u ", h_norms[i]);
        printf("\n");

        bool ok = validate_small_case_packed(h_matrix, h_XXT_packed, h_norms, m, n, packed_cols);
        printf("Validacion CPU/GPU: %s\n", ok ? "OK" : "FALLO");

        half *h_distances = (half*)malloc(tri_half_bytes);
        if (h_distances) {
            printf("\nDistancias euclideas (debug):\n");
            CUDA_CHK(cudaMemcpy(h_distances, d_distances_packed, tri_half_bytes, cudaMemcpyDeviceToHost));
            // Muestra en pantalla el MxM completo leyendo desde el vector Half
            print_matrix_half(h_distances, m);
            free(h_distances);
        }
        free(h_norms);
        free(h_XXT_packed);
    }

    cudaFree(d_matrix);
    cudaFree(d_norms);
    cudaFree(d_XXT_packed);
    cudaFree(d_distances_packed);
    if (h_matrix) free(h_matrix);

    return 0;
}
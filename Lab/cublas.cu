#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>

// ====================================================================================
// MACROS Y FUNCIONES AUXILIARES
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
        default:                             return "UNKNOWN_CUBLAS_ERROR";
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

static inline size_t div_up_size(size_t a, size_t b) {
    return (a + b - 1) / b;
}

static __device__ __host__ inline size_t get_packed_index(size_t r, size_t c, size_t m) {
    return r * m - (r * (r + 1)) / 2 + c;
}

// ====================================================================================
// FUNCIONES DEBUG
// ====================================================================================
void print_matrix_packed_float(const float *matrix, int m) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            size_t packed_idx = (i <= j) ? get_packed_index(i, j, m) : get_packed_index(j, i, m);
            printf("%6.2f ", matrix[packed_idx]);
        }
        printf("\n");
    }
}

bool validate_small_case(const uint8_t *X, const float *XXT_full, const float *norms, size_t m, size_t n) {
    for (size_t i = 0; i < m; i++) {
        float ref_norm = 0;
        for (size_t k = 0; k < n; k++) ref_norm += (float)X[i * n + k] * (float)X[i * n + k];
        if (fabsf(norms[i] - ref_norm) > 1e-3f) return false;
        for (size_t j = i; j < m; j++) {
            float ref = 0;
            for (size_t k = 0; k < n; k++) ref += (float)X[i * n + k] * (float)X[j * n + k];
            if (fabsf(XXT_full[i * m + j] - ref) > 1e-3f) return false;
        }
    }
    return true;
}

// ====================================================================================
// GENERADOR DE DATOS EN GPU
// ====================================================================================
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
    GenerateGenomicMatrixKernelU8<<<grid, block>>>(d_matrix, m, n, 42u);
    CUDA_CHK(cudaGetLastError());
}

// ====================================================================================
// KERNELS - PIPELINE cuBLAS (float32)
// ====================================================================================
__global__ void u8ToFloatKernel(const uint8_t* in, float* out, size_t count) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) out[idx] = (float)in[idx];
}

__global__ void completarMatrizKernel(float* C, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N && i < j) {
        C[i * N + j] = C[j * N + i];
    }
}

__global__ void extractDiagKernel(const float* C, float* diag, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) diag[i] = C[i * N + i];
}

__global__ void buildDPackedKernel(const float* C, const float* diag, float* Dpacked, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N && i <= j) {
        float d2 = diag[i] + diag[j] - 2.0f * C[i * N + j];
        if (d2 < 0.0f) d2 = 0.0f;  
        size_t idx = get_packed_index((size_t)i, (size_t)j, (size_t)N);
        Dpacked[idx] = sqrtf(d2);
    }
}

// ====================================================================================
// PROGRAMA PRINCIPAL
// ====================================================================================
int main(int argc, char *argv[]) {
    size_t m = (size_t)(1 << 10);  
    size_t n = (size_t)(1 << 15);  
    
    printf("Memoria usada: %.2f MB\n", (float)(m * n * sizeof(float)) / (1024.0f * 1024.0f));
    if (argc >= 3) {
        m = (size_t)atoi(argv[1]);
        n = (size_t)atoi(argv[2]);
    }

    if (m == 0 || n == 0) {
        fprintf(stderr, "Uso:\n %s [individuos m] [SNPs n]\n", argv[0]);
        return 1;
    }

    size_t tri_elems = m * (m + 1) / 2;
    size_t matrix_bytes = m * n * sizeof(uint8_t);

    printf("Iniciando Benchmark cuBLAS (Referencia FP32)\n");
    printf("Matriz: %zu x %zu | Elementos de D: %zu | Iteraciones Nsys: 10\n\n", m, n, tri_elems);

    // 1. Setup Device
    cublasHandle_t handle;
    CUBLAS_CHK(cublasCreate(&handle));

    uint8_t *d_matrix = NULL;
    float *d_Xf = NULL, *d_C = NULL, *d_diag = NULL, *d_Dpacked = NULL;

    CUDA_CHK(cudaMalloc(&d_matrix, matrix_bytes));
    CUDA_CHK(cudaMalloc(&d_Xf, m * n * sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_C, m * m * sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_diag, m * sizeof(float)));
    CUDA_CHK(cudaMalloc(&d_Dpacked, tri_elems * sizeof(float)));

    // Generar matriz genomica en Device con mismo Hash/Semilla
    nvtxRangePushA("GenX");
    generate_genomic_matrix_device(d_matrix, m, n);
    CUDA_CHK(cudaDeviceSynchronize());
    nvtxRangePop();

    // Configuración de grid/blocks
    int threads = 256;
    int blocksConv = (int)((m * n + threads - 1) / threads);
    int blocksDiag = (int)((m + threads - 1) / threads);
    dim3 block2D(32, 32);
    dim3 grid2D((unsigned)div_up_size(m, 32), (unsigned)div_up_size(m, 32));

    float alpha = 1.0f, beta = 0.0f;

    // Warm-up
    u8ToFloatKernel<<<blocksConv, threads>>>(d_matrix, d_Xf, m * n);
    cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, (int)m, (int)n, &alpha, d_Xf, (int)n, &beta, d_C, (int)m);
    cudaDeviceSynchronize();

    // =====================================================================
    // 2. BENCHMARK LOOP (10 Iteraciones con Tags NVTX)
    // =====================================================================
    for (int i = 0; i < 10; i++) {    
        
        nvtxRangePushA("u8ToFloat");
        u8ToFloatKernel<<<blocksConv, threads>>>(d_matrix, d_Xf, m * n);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();
        
        nvtxRangePushA("XXT");
        CUBLAS_CHK(cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                               (int)m, (int)n, &alpha, d_Xf, (int)n, &beta, d_C, (int)m));
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();

        nvtxRangePushA("CompleteMatrix");
        completarMatrizKernel<<<grid2D, block2D>>>(d_C, (int)m);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();

        nvtxRangePushA("Norms");
        extractDiagKernel<<<blocksDiag, threads>>>(d_C, d_diag, (int)m);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();

        nvtxRangePushA("CalculateDistance");
        buildDPackedKernel<<<grid2D, block2D>>>(d_C, d_diag, d_Dpacked, (int)m);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();
    }
    
    printf("[cuBLAS] Profiling finalizado exitosamente.\n");

    // =====================================================================
    // BLOQUE DEBUG CONDICIONAL (m < 32 && n < 128)
    // =====================================================================
    if (m <= 32 && n <= 128) {
        uint8_t *h_X = (uint8_t*)malloc(matrix_bytes);
        float *h_C = (float*)malloc(m * m * sizeof(float));
        float *h_diag = (float*)malloc(m * sizeof(float));
        float *h_D = (float*)malloc(tri_elems * sizeof(float));

        CUDA_CHK(cudaMemcpy(h_X, d_matrix, matrix_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHK(cudaMemcpy(h_C, d_C, m * m * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHK(cudaMemcpy(h_diag, d_diag, m * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHK(cudaMemcpy(h_D, d_Dpacked, tri_elems * sizeof(float), cudaMemcpyDeviceToHost));

        printf("\n--- DISTANCIAS (Packed) ---\n");
        print_matrix_packed_float(h_D, (int)m);

        if (validate_small_case(h_X, h_C, h_diag, m, n)) printf("\nValidacion OK\n");
        else printf("\nValidacion FALLO\n");

        free(h_X); free(h_C); free(h_diag); free(h_D);
    }

    // 3. Limpieza
    CUBLAS_CHK(cublasDestroy(handle));
    
    cudaFree(d_matrix);
    cudaFree(d_Xf);
    cudaFree(d_C);
    cudaFree(d_diag);
    cudaFree(d_Dpacked);

    return 0;
}
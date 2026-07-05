#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_THREADS_PER_BLOCK 1024  // Límite de la GTX 1060

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

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

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t _s = (call); \
    if (_s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "[CUBLAS ERROR] %s:%d: %s (code %d)\n", __FILE__, __LINE__, \
                cublasGetErrorStr(_s), (int)_s); \
        exit(1); \
    } \
} while (0)

// Chequea el ultimo error de un kernel: los kernels no devuelven status,
// asi que hay que preguntarle al runtime explicitamente despues de lanzarlos.
#define KERNEL_CHECK() do { \
    CUDA_CHECK(cudaGetLastError()); \
    CUDA_CHECK(cudaDeviceSynchronize()); \
} while (0)


int* generar_matriz_genomica_prueba(int n, int m, int n_poblaciones, unsigned int semilla) {
    srand(semilla);

    printf("[INFO] Generando matriz de prueba realista (%d individuos x %d SNPs)...\n", n, m);

    int *X = (int *)malloc((size_t)n * m * sizeof(int));
    if (X == NULL) {
        fprintf(stderr, "[ERROR] Memoria insuficiente para la matriz X.\n");
        return NULL;
    }

    double *frecuencias = (double *)malloc((size_t)n_poblaciones * m * sizeof(double));
    if (frecuencias == NULL) {
        fprintf(stderr, "[ERROR] Memoria insuficiente para el vector de frecuencias.\n");
        free(X);
        return NULL;
    }

    for (int pob = 0; pob < n_poblaciones; pob++) {
        for (int j = 0; j < m; j++) {
            double p = 0.05 + ((double)rand() / RAND_MAX) * (0.95 - 0.05);
            frecuencias[pob * m + j] = p;
        }
    }

    int individuos_por_pob = n / n_poblaciones;

    for (int pob = 0; pob < n_poblaciones; pob++) {
        int inicio = pob * individuos_por_pob;
        int fin = (pob == n_poblaciones - 1) ? n : (pob + 1) * individuos_por_pob;

        for (int i = inicio; i < fin; i++) {
            for (int j = 0; j < m; j++) {
                double p = frecuencias[pob * m + j];
                int alelos = 0;
                if ((double)rand() / RAND_MAX < p) alelos++;
                if ((double)rand() / RAND_MAX < p) alelos++;
                X[i * m + j] = alelos;
            }
        }
    }

    free(frecuencias);
    return X;
}

void print_matriz(int *mat, int filas, int cols) {
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < cols; j++) printf("%d,", mat[i * cols + j]);
        printf("\n");
    }
}
void print_matriz(float *mat, int filas, int cols) {
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < cols; j++) printf("%f,", mat[i * cols + j]);
        printf("\n");
    }
}

void escribir_csv_int(const char* path, const int* mat, int filas, int cols) {
    FILE* f = fopen(path, "w");
    if (f == NULL) {
        fprintf(stderr, "[ERROR] No se pudo abrir %s para escritura.\n", path);
        exit(1);
    }

    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < cols; j++) {
            if (j > 0) fprintf(f, ",");
            fprintf(f, "%d", mat[i * cols + j]);
        }
        fprintf(f, "\n");
    }

    fclose(f);
}

void escribir_csv_float(const char* path, const float* mat, int filas, int cols) {
    FILE* f = fopen(path, "w");
    if (f == NULL) {
        fprintf(stderr, "[ERROR] No se pudo abrir %s para escritura.\n", path);
        exit(1);
    }

    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < cols; j++) {
            if (j > 0) fprintf(f, ",");
            fprintf(f, "%.6f", mat[i * cols + j]);
        }
        fprintf(f, "\n");
    }

    fclose(f);
}

// ─────────────────────────────────────────────────────────────────────────
//  Kernels de la versión "estándar" (referencia contra la que comparamos)
// ─────────────────────────────────────────────────────────────────────────
__global__ void normKernel(int* IN, int* norms, int* normsT, int rows, int n) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    extern __shared__ int sdata[];

    int acc = 0;
    for (int k = tid; k < n; k += blockDim.x) {
        int val = IN[row * n + k];
        acc += val * val;
    }
    sdata[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    int norm2 = (int)sdata[0];
    for (int col = tid; col < rows; col += blockDim.x) {
        norms[row * rows + col] = norm2;
        normsT[col * rows + row] = norm2;
    }
}

__global__ void xxtKernel(int* IN, int* xxt, int rows, int n) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    int tid = threadIdx.x;
    if (j < i) return;

    extern __shared__ int sdata[];
    int acc = 0;
    for (int k = tid; k < n; k += blockDim.x) {
        acc += IN[i * n + k] * IN[j * n + k];
    }
    sdata[tid] = acc;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        xxt[i * rows + j] = 2 * sdata[0];
        xxt[j * rows + i] = 2 * sdata[0];
    }
}

__global__ void unionKernel(int* norms, int* normsT, int* xxt, float* D, int rows, int n) {
    int i = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= rows) return;
    int idx = i * rows + col;
    int d2 = norms[idx] + normsT[idx] - xxt[idx];
    if (d2 < 0) d2 = 0;
    D[idx] = sqrtf((float)d2);
}

void launchNormKernel(int* d_IN, int* d_norms, int* d_normsT, int rows, int n) {
    int blockSize = (n > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : n;
    dim3 block(blockSize);
    dim3 grid(rows);
    int sharedMemBytes = blockSize * sizeof(int);
    normKernel<<<grid, block, sharedMemBytes>>>(d_IN, d_norms, d_normsT, rows, n);
    KERNEL_CHECK();
}

void launchXXTKernel(int* d_IN, int* d_xxt, int N, int M) {
    int blockSize = (M > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : M;
    int sharedMem = blockSize * sizeof(int);
    dim3 grid(N, N);
    dim3 block(blockSize);
    xxtKernel<<<grid, block, sharedMem>>>(d_IN, d_xxt, N, M);
    KERNEL_CHECK();
}

// NOTA: rows debe caber en un solo bloque (rows <= MAX_THREADS_PER_BLOCK)
// para que blockIdx.y cubra todas las columnas con grid.y = ceil(rows/blockSize).
// Con rows=N=1024 y blockSize=n=M=64 esto da grid.y = 16 (antes, con
// dim3 grid(rows) simple, grid.y quedaba fijo en 1 y solo se llenaban
// las primeras 64 columnas de D: un bug heredado de la version original).
void launchUnionKernel(float* d_D, int* d_norms, int* d_normsT, int* d_xxt, int rows, int n) {
    int blockSize = (n > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : n;
    dim3 block(blockSize);
    dim3 grid(rows, (rows + blockSize - 1) / blockSize);
    unionKernel<<<grid, block>>>(d_norms, d_normsT, d_xxt, d_D, rows, n);
    KERNEL_CHECK();
}

// ─────────────────────────────────────────────────────────────────────────
//  Kernels auxiliares para la versión cuBLAS
// ─────────────────────────────────────────────────────────────────────────

// Convierte la matriz de entrada (int, {0,1,2}) a float, requerido por
// las rutinas de precisión simple de cuBLAS (Ssyrk, Snrm2, Saxpy).
__global__ void intToFloatKernel(const int* in, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) out[idx] = (float)in[idx];
}

// cublasSsyrk con CUBLAS_FILL_MODE_UPPER llena, en la convencion column-major
// de cuBLAS, las entradas con i<=j. Pero como este buffer se lee en row-major
// (C[fila*N+col]), esa mitad "superior colu b mn-major" corresponde exactamente
// a la mitad con fila >= col en terminos row-major (no fila <= col). Por eso
// se copia DESDE C[j*N+i] (valida, j>=i) HACIA C[i*N+j] (sin inicializar, i<j).
__global__ void symmetrizeKernel(float* C, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && j < N && i < j) {
        C[i * N + j] = C[j * N + i];
    }
}

// La diagonal de XXT es exactamente ||x_i||^2 (norma al cuadrado de la
// fila i), así que no hace falta un kernel de normas aparte: se extrae
// directamente de la diagonal del resultado de cublasSsyrk.
__global__ void extractDiagKernel(const float* C, float* diag, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) diag[i] = C[i * N + i];
}

// D^2_ij = ||x_i||^2 + ||x_j||^2 - 2*XXT_ij   (ecuación (1) de la letra)
__global__ void buildDCublasKernel(const float* C, const float* diag, float* D, int N) {
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float d2 = diag[i] + diag[j] - 2.0f * C[i * N + j];
    if (d2 < 0.0f) d2 = 0.0f;  // por posibles errores de redondeo en float
    D[i * N + j] = sqrtf(d2);
}

int main(int argc, char** argv) {
    int N = 2048;  // individuos (filas)
    int M = 2048;  // SNPs (columnas)
    int escribir_csv = 0;

    // Uso: ./labPensado_cublas [N] [M] [--csv]
    // Sin argumentos usa N=M=2048 y no escribe CSV (para poder barrer
    // muchos tamaños rápido sin llenar el disco con matrices grandes).
    if (argc >= 3) {
        N = atoi(argv[1]);
        M = atoi(argv[2]);
    }
    for (int a = 1; a < argc; a++) {
        if (strcmp(argv[a], "--csv") == 0) escribir_csv = 1;
    }

    if (N <= 0 || M <= 0) {
        fprintf(stderr, "[ERROR] N y M deben ser positivos. Uso: %s [N] [M] [--csv]\n", argv[0]);
        return -1;
    }

    int subpoblaciones = 3;
    unsigned int semilla = 42;

    // ──  Generar matriz en CPU (misma para ambas variantes) ─────
    int *c_in = generar_matriz_genomica_prueba(N, M, subpoblaciones, semilla);
    if (c_in == NULL) return -1;

    // ──  Reservar y copiar a GPU ─────────────────────────────────
    int *d_IN = NULL, *d_norms = NULL, *d_normsT = NULL, *d_xxt = NULL;
    float *d_D = NULL;

    CUDA_CHECK(cudaMalloc((void**)&d_IN, (size_t)N * M * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_norms, (size_t)N * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_normsT, (size_t)N * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_xxt, (size_t)N * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_D, (size_t)N * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_IN, c_in, (size_t)N * M * sizeof(int), cudaMemcpyHostToDevice));

    // ──  Version estandar (kernels propios) + timing ─────────────
    cudaEvent_t t0, t1, t2;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventCreate(&t2));

    CUDA_CHECK(cudaEventRecord(t0));
    launchNormKernel(d_IN, d_norms, d_normsT, N, M);
    launchXXTKernel(d_IN, d_xxt, N, M);
    launchUnionKernel(d_D, d_norms, d_normsT, d_xxt, N, M);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms_estandar = 0.0f;
    cudaEventElapsedTime(&ms_estandar, t0, t1);
    printf("[TIMING] Version estandar: %.3f ms\n", ms_estandar);

    // ── 4. Versión cuBLAS ───────────────────────────────────────────
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float *d_INf = NULL, *d_C = NULL, *d_diag = NULL, *d_Dcublas = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_INf, (size_t)N * M * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_diag, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_Dcublas, (size_t)N * N * sizeof(float)));

    int threads = 256;
    int blocksConv = (N * M + threads - 1) / threads;
    intToFloatKernel<<<blocksConv, threads>>>(d_IN, d_INf, N * M);
    KERNEL_CHECK();

    float alpha = 1.0f, beta = 0.0f;

    CUDA_CHECK(cudaEventRecord(t0));

    // Aqui vamos aprovechar que la matriz es simeytrica para solo calcular uno d elos triangulos
    CUBLAS_CHECK(cublasSsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                              N, M, &alpha, d_INf, M, &beta, d_C, N));

    // Completar el triangulo inferior (syrk solo llena el superior)
    dim3 blockSym(32, 32);
    dim3 gridSym((N + 31) / 32, (N + 31) / 32);
    symmetrizeKernel<<<gridSym, blockSym>>>(d_C, N);
    KERNEL_CHECK();

    // Normas al cuadrado = diagonal de XXT
    int blocksDiag = (N + threads - 1) / threads;
    extractDiagKernel<<<blocksDiag, threads>>>(d_C, d_diag, N);
    KERNEL_CHECK();

    // Ensamblar D a partir de la ecuacion (1): D2 = Ne + Ne^T - 2*XXT
    int blockSizeD = (N > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : N;
    dim3 blockD(blockSizeD);
    dim3 gridD(N, (N + blockSizeD - 1) / blockSizeD);
    buildDCublasKernel<<<gridD, blockD>>>(d_C, d_diag, d_Dcublas, N);
    KERNEL_CHECK();

    CUDA_CHECK(cudaEventRecord(t2));
    CUDA_CHECK(cudaEventSynchronize(t2));

    float ms_cublas = 0.0f;
    cudaEventElapsedTime(&ms_cublas, t0, t2);
    printf("[TIMING] Version cuBLAS (Ssyrk): %.3f ms\n", ms_cublas);

    // ──  Verificacion: ||D - D_cuBLAS||_2 usando cublasSaxpy + cublasSnrm2 ──
    float *d_diff = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_diff, (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_diff, d_D, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToDevice));

    float neg_one = -1.0f;
    // d_diff = d_diff - d_Dcublas  =  D_estandar - D_cuBLAS
    CUBLAS_CHECK(cublasSaxpy(handle, N * N, &neg_one, d_Dcublas, 1, d_diff, 1));

    float norm_diff = 0.0f;
    CUBLAS_CHECK(cublasSnrm2(handle, N * N, d_diff, 1, &norm_diff));

    printf("||D - D_cuBLAS||_2 = %f\n", norm_diff);

    // Linea de resumen facil de parsear (usada por el script de benchmark)
    printf("[RESULT] N=%d M=%d t_estandar_ms=%.4f t_cublas_ms=%.4f speedup=%.4f diff=%e\n",
           N, M, ms_estandar, ms_cublas,
           (ms_cublas > 0.0f) ? (ms_estandar / ms_cublas) : 0.0f, norm_diff);

    // ── Salidas CSV para chequeo externo (opcional, con --csv) ─────
    if (escribir_csv) {
        float *h_D = (float *)malloc((size_t)N * N * sizeof(float));
        float *h_Dcublas = (float *)malloc((size_t)N * N * sizeof(float));
        float *h_diff = (float *)malloc((size_t)N * N * sizeof(float));
        if (h_D == NULL || h_Dcublas == NULL || h_diff == NULL) {
            fprintf(stderr, "[ERROR] Memoria insuficiente para copiar matrices de salida.\n");
            free(h_D);
            free(h_Dcublas);
            free(h_diff);
            exit(1);
        }

        CUDA_CHECK(cudaMemcpy(h_D, d_D, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_Dcublas, d_Dcublas, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_diff, d_diff, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToHost));

        escribir_csv_int("X.csv", c_in, N, M);
        escribir_csv_float("D_estandar.csv", h_D, N, N);
        escribir_csv_float("D_cublas.csv", h_Dcublas, N, N);
        escribir_csv_float("D_diff.csv", h_diff, N, N);
        printf("[INFO] CSV escritos: X.csv, D_estandar.csv, D_cublas.csv, D_diff.csv\n");

        free(h_D);
        free(h_Dcublas);
        free(h_diff);
    }

    // ──  Liberar memoria ──────────────────────────────────────────
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaEventDestroy(t2));

    free(c_in);
    CUDA_CHECK(cudaFree(d_IN));
    CUDA_CHECK(cudaFree(d_norms));
    CUDA_CHECK(cudaFree(d_normsT));
    CUDA_CHECK(cudaFree(d_xxt));
    CUDA_CHECK(cudaFree(d_D));
    CUDA_CHECK(cudaFree(d_INf));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_diag));
    CUDA_CHECK(cudaFree(d_Dcublas));
    CUDA_CHECK(cudaFree(d_diff));

    return 0;
}

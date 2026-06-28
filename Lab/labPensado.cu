#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_THREADS_PER_BLOCK 1024  // Límite de la GTX 1060

int* generar_matriz_genomica_prueba(int n, int m, int n_poblaciones, unsigned int semilla) {
    // Configurar la semilla localmente para esta generación
    srand(semilla);

    printf("[INFO] Generando matriz de prueba realista (%d individuos x %d SNPs)...\n", n, m);

    // 1. Reservar memoria para la matriz final (linealizada)
    int *X = (int *)malloc((size_t)n * m * sizeof(int));
    if (X == NULL) {
        fprintf(stderr, "[ERROR] Memoria insuficiente para la matriz X.\n");
        return NULL;
    }

    // 2. Reservar memoria temporal para las frecuencias de los SNPs por cada población
    double *frecuencias = (double *)malloc((size_t)n_poblaciones * m * sizeof(double));
    if (frecuencias == NULL) {
        fprintf(stderr, "[ERROR] Memoria insuficiente para el vector de frecuencias.\n");
        free(X);
        return NULL;
    }

    // 3. Definir frecuencias alélicas aleatorias (entre 0.05 y 0.95) características de cada población
    for (int pob = 0; pob < n_poblaciones; pob++) {
        for (int j = 0; j < m; j++) {
            double p = 0.05 + ((double)rand() / RAND_MAX) * (0.95 - 0.05);
            frecuencias[pob * m + j] = p;
        }
    }

    // 4. Muestrear los datos usando una distribución Binomial B(2, p) para simular organismos diploides {0, 1, 2}
    int individuos_por_pob = n / n_poblaciones;

    for (int pob = 0; pob < n_poblaciones; pob++) {
        int inicio = pob * individuos_por_pob;
        // La última población absorbe el residuo si n no es divisible de forma exacta
        int fin = (pob == n_poblaciones - 1) ? n : (pob + 1) * individuos_por_pob;

        for (int i = inicio; i < fin; i++) {
            for (int j = 0; j < m; j++) {
                double p = frecuencias[pob * m + j];
                
                // Simulación del conteo de alelos (2 lanzamientos de moneda sesgada por p)
                int alelos = 0;
                if ((double)rand() / RAND_MAX < p) alelos++;
                if ((double)rand() / RAND_MAX < p) alelos++;

                // Guardar en formato linealizado [fila * total_columnas + columna]
                X[i * m + j] = alelos;
            }
        }
    }

    // Liberar la estructura auxiliar de frecuencias
    free(frecuencias);

    return X;
}


void print_matriz(int *mat, int filas, int cols) {
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d,", mat[i * cols + j]);
        }
        printf("\n");
    }
}
void print_matriz(float *mat, int filas, int cols) {
    for (int i = 0; i < filas; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f,", mat[i * cols + j]);
        }
        printf("\n");
    }
}
// ── Kernel: cada bloque calcula la norma al cuadrado de una fila ──
__global__ void normKernel(int* IN, int* norms, int* normsT, int rows, int n) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ int sdata[];

    // Cada hilo acumula su porción de la fila
    // (necesario si blockDim.x < n porque se clampeó)
    int sum = 0;
    int val = IN[row * n + tid];
    sdata[tid] =  val * val;
    __syncthreads();

    // Reducción en shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
 
    norms[row * rows  + tid] = (int)sdata[0];
    normsT[tid * rows + row] = (int)sdata[0];
}
// ── Kernel: cada bloque calcula la norma al cuadrado de una fila ──
__global__ void xxtKernel(int* IN, int* xxt, int rows, int n) {
    int i = blockIdx.x;
    int j = blockIdx.y;
    int tid = threadIdx.x;

    // Solo triángulo superior
    if (j < i) return;

    extern __shared__ int sdata[];

    sdata[tid] = IN[i * n + tid] * IN[j * n + tid];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // Solo hilo 0 escribe — una sola posición por bloque
    if (tid == 0) {
        xxt[i * rows + j] = 2 * sdata[0];
        xxt[j * rows + i] = 2 * sdata[0];  // simétrico gratis
    }
}

// ── Kernel: cada bloque calcula la norma al cuadrado de una fila ──
__global__ void unionKernel( int* norms, int* normsT,int* xxt,float * D, int rows, int n) {
    int i   = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;  // columna global
    if (col >= rows) return;
    int idx = i * rows + col;  
    int d2 = norms[idx] + normsT[idx] - xxt[idx];
    D[idx] = sqrtf((float)d2);  // distancia euclídea

}
// ── Función host: recibe IN y n, decide el blockSize ──
void launchUnionKernel(float * d_D, int* d_norms, int* d_normsT, int* d_xxt, int rows, int n) {

    int blockSize = (n > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : n;

    dim3 block(blockSize);
    dim3 grid(rows);        // un bloque por individuo (fila)

    int sharedMemBytes = blockSize * sizeof(int);

    printf("n=%d → blockSize=%d, grid=%d\n", n, blockSize, rows);

    unionKernel<<<grid, block, sharedMemBytes>>>(d_norms, d_normsT, d_xxt, d_D, rows, n);
    cudaDeviceSynchronize();
}
// ── Función host: recibe IN y n, decide el blockSize ──
void launchNormKernel(int* d_IN, int* d_norms, int* d_normsT, int* d_xxt, int rows, int n) {

    int blockSize = (n > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : n;

    dim3 block(blockSize);
    dim3 grid(rows);        // un bloque por individuo (fila)

    int sharedMemBytes = blockSize * sizeof(int);

    printf("n=%d → blockSize=%d, grid=%d\n", n, blockSize, rows);

    normKernel<<<grid, block, sharedMemBytes>>>(d_IN, d_norms, d_normsT, rows, n);
    cudaDeviceSynchronize();
}
void launchXXTKernel(int* d_IN, int* d_xxt, int N, int M) {
    int blockSize = (M > MAX_THREADS_PER_BLOCK) ? MAX_THREADS_PER_BLOCK : M;
    int sharedMem = blockSize * sizeof(int);

    dim3 grid(N, N);
    dim3 block(blockSize);

    xxtKernel<<<grid, block, sharedMem>>>(d_IN, d_xxt, N, M);
    cudaDeviceSynchronize();
}

int main(){
    int N = 1024;  // individuos (filas)
    int M = 64;    // SNPs (columnas)

    int subpoblaciones = 3;
    unsigned int semilla = 42;

    // ── 1. Generar matriz en CPU ──────────────────────────────────
    int *c_in = generar_matriz_genomica_prueba(N, M, subpoblaciones, semilla);
    if (c_in == NULL) return -1;
    print_matriz(c_in, N, M);
    // ── 2. Reservar memoria en GPU ────────────────────────────────
    int *d_IN    = NULL;
    int *d_norms = NULL;
    int *d_normsT = NULL;
    int *d_xxt = NULL;
    float *d_D = NULL;    
    cudaError_t err;

    err = cudaMalloc((void**)&d_IN, (size_t)N * M * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] cudaMalloc d_IN: %s\n", cudaGetErrorString(err));
        free(c_in); return -1;
    }

    err = cudaMalloc((void**)&d_norms, (size_t)N * N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] cudaMalloc d_norms: %s\n", cudaGetErrorString(err));
        free(c_in); cudaFree(d_IN); return -1;
    }

    err = cudaMalloc((void**)&d_normsT, (size_t)N * N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] cudaMalloc d_normsT: %s\n", cudaGetErrorString(err));
        free(c_in); cudaFree(d_IN); cudaFree(d_norms); return -1;
    }

    err = cudaMalloc((void**)&d_xxt, (size_t)N * N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] cudaMalloc d_xxt: %s\n", cudaGetErrorString(err));
        free(c_in); cudaFree(d_IN); cudaFree(d_norms); cudaFree(d_normsT); return -1;
    }

    err = cudaMalloc((void**)&d_D, (size_t)N * N * sizeof(float ));
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] cudaMalloc d_D: %s\n", cudaGetErrorString(err));
        free(c_in); cudaFree(d_IN); cudaFree(d_norms); cudaFree(d_normsT); cudaFree(d_xxt); return -1;
    }

    // ── 3. Copiar X de CPU → GPU ──────────────────────────────────
    err = cudaMemcpy(d_IN, c_in, (size_t)N * M * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] cudaMemcpy H→D: %s\n", cudaGetErrorString(err));
        free(c_in); cudaFree(d_IN); cudaFree(d_norms); cudaFree(d_normsT); return -1;
    }

    // ── 4. Lanzar kernel ──────────────────────────────────────────
    launchNormKernel(d_IN, d_norms, d_normsT, d_xxt, N, M);
    launchXXTKernel(d_IN, d_xxt, N, M);
    launchUnionKernel(d_D, d_norms, d_normsT, d_xxt, N, M);

    // ── 5. Copiar norms GPU → CPU e imprimir ─────────────────────
    int *c_norms = (int*)malloc((size_t)N * N * sizeof(int));
    cudaMemcpy(c_norms, d_norms, (size_t)N * N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Norms (GPU → CPU):\n");
    print_matriz(c_norms, N, N);

    // ── 6. Copiar normsT GPU → CPU e imprimir ────────────────────
    int *c_normsT = (int*)malloc((size_t)N * N * sizeof(int));
    cudaMemcpy(c_normsT, d_normsT, (size_t)N * N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("NormsT (GPU → CPU):\n");
    print_matriz(c_normsT, N, N);


    // ── 7. Copiar xxt GPU → CPU e imprimir ────────────────────
    int *c_xxt = (int*)malloc((size_t)N * N * sizeof(int));
    cudaMemcpy(c_xxt, d_xxt, (size_t)N * N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Xxt (GPU → CPU):\n");
    print_matriz(c_xxt, N, N);

    // ── 7. Copiar xxt GPU → CPU e imprimir ────────────────────
    float *c_D = (float*)malloc((size_t)N * N * sizeof(float));
    cudaMemcpy(c_D, d_D, (size_t)N * N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("D (GPU → CPU):\n");
    print_matriz(c_D, N, N);

    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] cudaMemcpy D→H: %s\n", cudaGetErrorString(err));
        free(c_in);  cudaFree(d_IN); cudaFree(d_norms); cudaFree(d_D); return -1;
    }

    // ── 5. Liberar memoria ────────────────────────────────────────
    free(c_in);
    cudaFree(d_IN);
    cudaFree(d_norms);
    cudaFree(d_normsT);
    cudaFree(d_xxt);
    cudaFree(d_D);
    free(c_norms);
    free(c_normsT);
    free(c_xxt);
    free(c_D);
    return 0;
}
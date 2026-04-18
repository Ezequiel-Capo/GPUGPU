#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda.h"

#define TILE_DIM 32
#define CUDA_CHK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while (0)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void m_traspose_kernel(float *d_matriz_ini, float *d_matriz_res, int N, int tile_dim)
{
    // Padding fijo de 1 para evitar conflictos de bancos al transponer.
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    (void)tile_dim;

    int idx_x = blockIdx.x * blockDim.x + threadIdx.x; //column_id
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y; //row_id

     // cargar desde global 
    if (idx_x < N && idx_y < N) {
        tile[threadIdx.y][threadIdx.x] = d_matriz_ini[idx_y * N + idx_x];
    }

    __syncthreads(); //esperar a que todos los threads hayan cargado su elemento en shared

    // transponer índices de bloque
    int x = blockIdx.y * TILE_DIM + threadIdx.x;
    int y = blockIdx.x * TILE_DIM + threadIdx.y;

    // escribir desde shared 
    if (x < N && y < N) {
        d_matriz_res[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}

int main(int argc, char *argv[])
{
    float times[11];


    int N = (1<<14);
    int block_x = 32, block_y = 32;
    char tile_dim = 0;
    if (argc > 1) tile_dim = atoi(argv[1]); //pasar string a int

    printf("Tiles de tamaño: %d x %d\n", TILE_DIM, TILE_DIM + tile_dim);
    char v = 0;  // 0: no imprimir matrices, 1: imprimir

    unsigned int size = N * N * sizeof(float);
    
    // Reservar memoria en host
    float * h_matriz_ini = (float *)malloc(size);
    float * h_matriz_res = (float *)malloc(size);

	// Inicializar matriz de entrada
    for (int i = 0; i < N * N; i++) {
        h_matriz_ini[i] = (float)(i + 1);
    }

    // Reservar memoria en device
    float *d_matriz_ini, *d_matriz_res;
	// reservar memoria en la GPU
	cudaMalloc((void **)&d_matriz_ini, size);
	cudaMalloc((void **)&d_matriz_res, size);

	// copiar el de host a device
	cudaMemcpy(d_matriz_ini, h_matriz_ini, size, cudaMemcpyHostToDevice);

    //total_threads = #blocks * #threads_per_block(32*32)
    //total_threads = x*y (matrix size)
	dim3 block_s(block_x, block_y); //, size = x*y*4B (por letra fijo)
	dim3 grid_s((N + block_s.x - 1) / block_s.x, (N + block_s.y - 1) / block_s.y); 

    // Create CUDA events
    cudaEvent_t start, stop;
    CUDA_CHK(cudaEventCreate(&start));
    CUDA_CHK(cudaEventCreate(&stop));

    for (int i = 0; i < 11; i++) {
        // Start measuring time
        CUDA_CHK(cudaEventRecord(start, 0));

        m_traspose_kernel<<<grid_s, block_s>>>(d_matriz_ini, d_matriz_res, N, tile_dim);
        CUDA_CHK(cudaGetLastError());

        // Stop measuring time and compute
        CUDA_CHK(cudaEventRecord(stop, 0));
        CUDA_CHK(cudaEventSynchronize(stop));
        CUDA_CHK(cudaEventElapsedTime(&times[i], start, stop));
        printf("Run %d took: %.6f ms\n", i + 1, times[i]);
    }

    float mean = 0.0f;
    for (int i = 1; i < 11; i++) { //elimino warm up
        mean += times[i];
    }
    mean /= 10.0f;

    float variance = 0.0f;
    for (int i = 1; i < 11; i++) { //elimino warm up
        float diff = times[i] - mean;
        variance += diff * diff;
    }
    variance /= 10.0f;
    float stddev = sqrtf(variance);

    printf("Grid size: (%d, %d), Block size: (%d, %d)\n", grid_s.x, grid_s.y, block_s.x, block_s.y);
    printf("Tiempo promedio (10 runs): %.6f ms\n", mean);
    printf("Desviación estándar: %.6f ms\n", stddev);
    CUDA_CHK(cudaEventDestroy(start));
    CUDA_CHK(cudaEventDestroy(stop));
    //fin medicion tiempo

    // copiar el de device a host
	cudaMemcpy(h_matriz_res, d_matriz_res, size, cudaMemcpyDeviceToHost);

    //liberar mem gpu
	cudaFree(d_matriz_ini);
	cudaFree(d_matriz_res);

    // despliego la matriz resultante
    if (v) {
        printf("Matriz original:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%.1f ", h_matriz_ini[i * N + j]);
            }
            printf("\n");
        }

        printf("Matriz transpuesta:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%.1f ", h_matriz_res[i * N + j]);
            }
            printf("\n");
        }
    }

	// libero la memoria en la CPU
	free(h_matriz_res);
	free(h_matriz_ini);

	return 0;
}
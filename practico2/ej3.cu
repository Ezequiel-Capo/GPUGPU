#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda.h"

#define CUDA_CHK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while (0)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Construir un kernel que reciba una matriz de enteros alojada en memoria global y devuelva la matriz
// transpuesta. Reserve dos espacios de memoria distintos para las matrices de entrada y salida. En este
// ejercicio el kernel no debe utilizar la memoria compartida (todas las lecturas y escrituras deben realizarse
// en memoria global). La grilla debe ser bidimensional y los bloques tambi´en deben ser bidimensionales.

// 1. Ejecute el kernel con un tama˜no de bloque de 32×32 y analice el patr´on de acceso a memoria global
// de cada warp que se da en las lecturas y escrituras. Mida el tiempo de ejecuci´on del kernel.
// 2. Modifique el tama˜no de bloque para reducir los accesos no-coalesced. Justifique adecuadamente la
// elecci´on de tama˜no de bloque. Ejecute el kernel nuevamente y compare el tiempo de ejecuci´on con el
// caso anterior.


__device__ int modulo(int a, int b){
	int r = a % b;
	r = (r < 0) ? r + b : r;
	return r;
}


__global__ void m_traspose_kernel(float *d_matriz_ini, float *d_matriz_res, int Nx, int Ny)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x; // columna
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y; // fila

    if (idx_x < Nx && idx_y < Ny) {
        d_matriz_res[idx_x * Ny + idx_y] = d_matriz_ini[idx_y * Nx + idx_x];
    }
}

int main(int argc, char *argv[])
{

    int Nx = 1024, Ny = 1024;
    int block_x = 32, block_y = 32;
    char v = 0;  // 0: no imprimir matrices, 1: imprimir

    if (argc > 1) Nx = atoi(argv[1]);
    if (argc > 2) Ny = atoi(argv[2]);
    if (argc > 3) block_x = atoi(argv[3]);
    if (argc > 4) block_y = atoi(argv[4]);
    if (argc > 5) v = atoi(argv[5]);

    unsigned int size = Nx * Ny * sizeof(float);
    
    // Reservar memoria en host
    float * h_matriz_ini = (float *)malloc(size);
    float * h_matriz_res = (float *)malloc(size);

	// Inicializar matriz de entrada
    for (int i = 0; i < Nx * Ny; i++) {
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
	dim3 grid_s((Nx + block_s.x - 1) / block_s.x, (Ny + block_s.y - 1) / block_s.y); 

    // medir tiempo del kernel con clock() de C
    clock_t start = clock();

    m_traspose_kernel<<<grid_s, block_s>>>(d_matriz_ini, d_matriz_res, Nx, Ny);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    clock_t stop = clock();
    double elapsed_ms = ((double)(stop - start) * 1000.0) / CLOCKS_PER_SEC;
    printf("Tiempo de ejecucion del kernel: %.6f ms (%.3f us)\n", elapsed_ms, elapsed_ms * 1000.0f);
    //fin medicion tiempo

    // copiar el de device a host
	cudaMemcpy(h_matriz_res, d_matriz_res, size, cudaMemcpyDeviceToHost);

    //liberar mem gpu
	cudaFree(d_matriz_ini);
	cudaFree(d_matriz_res);

    // despliego la matriz resultante
    if (v) {
        printf("Matriz original:\n");
        for (int i = 0; i < Ny; i++) {
            for (int j = 0; j < Nx; j++) {
                printf("%.1f ", h_matriz_ini[i * Nx + j]);
            }
            printf("\n");
        }

        printf("Matriz transpuesta:\n");
        for (int i = 0; i < Ny; i++) {
            for (int j = 0; j < Nx; j++) {
                printf("%.1f ", h_matriz_res[i * Nx + j]);
            }
            printf("\n");
        }
    }

	// libero la memoria en la CPU
	free(h_matriz_res);
	free(h_matriz_ini);

	return 0;
}

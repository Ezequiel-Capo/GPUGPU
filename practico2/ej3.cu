#include <stdio.h>
#include <stdlib.h>
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

    int Nx = 64, Ny = 64;
    int block_x = 32, block_y = 32;
    char v = 0;  // 0: no imprimir matrices, 1: imprimir

    if (argc > 1) Nx = atoi(argv[1]); //pasar string a int
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

    //para medir el tiempo de ejecucion del kernel
    cudaEvent_t start, stop;
    CUDA_CHK(cudaEventCreate(&start));
    CUDA_CHK(cudaEventCreate(&stop));
    CUDA_CHK(cudaEventRecord(start));

    m_traspose_kernel<<<grid_s, block_s>>>(d_matriz_ini, d_matriz_res, Nx, Ny);
    CUDA_CHK(cudaGetLastError());

    CUDA_CHK(cudaEventRecord(stop));
    CUDA_CHK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    printf("Tiempo de ejecucion del kernel: %.6f ms (%.3f us)\n", elapsed_ms, elapsed_ms * 1000.0f);
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
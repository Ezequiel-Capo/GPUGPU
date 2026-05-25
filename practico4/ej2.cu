#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cuda.h"

#define TILE_DIM 32
#define MASK 0xFFFFFFFFu

#define CUDA_CHK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while (0)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__inline__ __device__ int warpReduceSum(int val, unsigned mask) {
    for (int offset = 16; offset > 0; offset /= 2) 
        val += __shfl_down_sync(mask, val, offset);
    return val;
}


__inline__ __device__ int warpReduceMax(int val, unsigned mask){
    for (int offset = 16; offset > 0; offset /= 2){
        int pivot = __shfl_down_sync(mask, val, offset);
        if (pivot > val) 
            val = pivot;
    }
    return val;
}

__global__ void kernel_redux_coop_g(int *d_vector_x, int *d_vector_y)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid; 


}


int main(int argc, char *argv[])
{
    srand((unsigned int)time(NULL));

    int N = (1<<28);
    int block = 32; //COMPLETAMENTE NECESARIO PARA LA SHARED
    int a = 0;
    int b = 10;
    char v = 0;
    if (argc > 1) 
        N = atoi(argv[1]);
    if (argc > 2) 
        v = atoi(argv[2]);
    if (argc > 4) 
        a = atoi(argv[4]);
    if (argc > 5) 
        b = atoi(argv[5]);

    int size = N * sizeof(int);
    int sizeSeg = size / 8; //tamño por segmento

    printf("Vector size: %d, Segment size: %d\n", N, N/8);
    // Reservar memoria en host
    int * h_vector_x = (int *)malloc(size);
    int * h_vector_y = (int *)malloc(sizeSeg);

    for (int i = 0; i <  N; i++) {
        h_vector_x[i] = (a + rand() % (b - a + 1)); //naturales de 0 a 10
    }

    // Reservar memoria en device
    int *d_vector_x, *d_vector_y;
	// reservar memoria en la GPU
	CUDA_CHK(cudaMalloc((void **)&d_vector_x, size));
	CUDA_CHK(cudaMalloc((void **)&d_vector_y, sizeSeg));

	// copiar el de host a device
	CUDA_CHK(cudaMemcpy(d_vector_x, h_vector_x, size, cudaMemcpyHostToDevice));

    //total_threads = #blocks * #threads_per_block
    //total_threads = N
	dim3 block_s(block); //

	dim3 grid_s((N + block - 1) / block); //si N siempre multiplo no importa block_s-1

    printf("N: %d, block: %d, grid: %d\n", N, block_s.x, grid_s.x);

    kernel_redux_coop_g<<<grid_s, block_s>>>(d_vector_x, d_vector_y);

    CUDA_CHK(cudaDeviceSynchronize());

    // copiar el de device a host
	CUDA_CHK(cudaMemcpy(h_vector_y, d_vector_y, sizeSeg, cudaMemcpyDeviceToHost));

    //liberar mem gpu
	CUDA_CHK(cudaFree(d_vector_x));
	CUDA_CHK(cudaFree(d_vector_y));

    // despliego la matriz resultante
    if (v) {
        printf("vector x:\n");
        for (int i = 0; i < N; i++) {
            printf("%d ", h_vector_x[i]);
        }
        printf("\n");

        printf("vector y:\n");
        for (int i = 0; i < N/8; i++) {
            printf("%d ", h_vector_y[i]);
        }
        printf("\n");
    }

	// libero la memoria en la CPU
	free(h_vector_y);
	free(h_vector_x);

	return 0;
}
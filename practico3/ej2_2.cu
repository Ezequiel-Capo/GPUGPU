#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
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

__inline__ __device__ int warpReduceSum(int val, unsigned mask) {
    for (int offset = 16; offset > 0; offset /= 2) 
        val += __shfl_down_sync(mask, val, offset);
    return val;
}


__inline__ __device__ int warpReduceMax(int val, unsigned mask){
    for (int offset = 16; offset > 0; offset /= 2){
        int other = __shfl_down_sync(mask, val, offset);
        if (other > val) 
            val = other;
    }
    return val;
}

__global__ void func_suma_arreglo(int *d_arreglo_ini, int *d_arreglo_res)
{
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid; 

    int val = d_arreglo_ini[gid];
    unsigned mask = 0xFFFFFFFFu;
    unsigned negsMask = __ballot_sync(mask, (val < 0));
    //unsigned posMask = ~negsMask;

    int negVal = 0;
    if (val < 0)
       negVal = val;

    int negSum = warpReduceSum(negVal, mask);
    negSum = __shfl_sync(negsMask, negSum, 0);

    int maxVal = warpReduceMax(val, mask); 
    maxVal = __shfl_sync(negsMask, maxVal, 0);

    if (val < 0)
        d_arreglo_res[gid] = negSum + maxVal;
    else
        d_arreglo_res[gid] = val;
}


int main(int argc, char *argv[])
{
    srand((unsigned int)time(NULL));

    int N = (1<<14);
    int block = 32; //COMPLETAMENTE NECESARIO PARA LA SHARED
    int size = N * sizeof(int);
    int a = 0;
    int b = 100;
    char v = 0;
    if (argc > 1) 
        N = atoi(argv[1]);
    if (argc > 2) 
        v = atoi(argv[2]);
    if (argc > 4) 
        a = atoi(argv[4]);
    if (argc > 5) 
        b = atoi(argv[5]);
    
    // Reservar memoria en host
    int * arreglo_ini = (int *)malloc(size);
    int * arreglo_res = (int *)malloc(size);

    for (int i = 0; i <  N; i++) {
        int signo = (rand() % 2) ? 1 : -1;
        arreglo_ini[i] = (a + rand() % (b - a + 1)) * signo; //enteros -100 a 100
    }

    // Reservar memoria en device
    int *d_arreglo_ini, *d_arreglo_res;
	// reservar memoria en la GPU
	CUDA_CHK(cudaMalloc((void **)&d_arreglo_ini, size));
	CUDA_CHK(cudaMalloc((void **)&d_arreglo_res, size));

	// copiar el de host a device
	CUDA_CHK(cudaMemcpy(d_arreglo_ini, arreglo_ini, size, cudaMemcpyHostToDevice));//cuda chk para verificar que la copia se realizó correctamente

    //total_threads = #blocks * #threads_per_block(128)
    //total_threads = N
	dim3 block_s(block); //, size = x*y*4B (por letra fijo)
	dim3 grid_s((N + block - 1) / block); //si N siempre multiplo no importa block_s-1

    for (int i = 0; i < 10; i++) {
        func_suma_arreglo<<<grid_s, block_s>>>(d_arreglo_ini, d_arreglo_res);
        CUDA_CHK(cudaGetLastError());
        CUDA_CHK(cudaDeviceSynchronize());
    }

    // copiar el de device a host
	CUDA_CHK(cudaMemcpy(arreglo_res, d_arreglo_res, size, cudaMemcpyDeviceToHost));

    //liberar mem gpu
	CUDA_CHK(cudaFree(d_arreglo_ini));
	CUDA_CHK(cudaFree(d_arreglo_res));

    // despliego la matriz resultante
    if (v) {
        printf("arreglo original:\n");
        for (int i = 0; i < N; i++) {
            printf("%d ", arreglo_ini[i]);
        }
        printf("\n");

        printf("arreglo con funcion aplicada:\n");
        for (int i = 0; i < N; i++) {
            printf("%d ", arreglo_res[i]);
        }
        printf("\n");
    }

	// libero la memoria en la CPU
	free(arreglo_res);
	free(arreglo_ini);

	return 0;
}
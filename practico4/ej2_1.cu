#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cuda.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;


#define CUDA_CHK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while (0)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void kernel_redux_coop_g_labeled(const int* x, int* y, int* label, int vectorSize){

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid; 

    // Obtiene el grupo cooperativo del bloque actual
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    if (vectorSize >= gid){
        int segmento =  label[gid];

        cg::coalesced_group labeled_group = cg::labeled_partition(warp, segmento);

        int valor = x[gid];

        int lane = labeled_group.thread_rank(); 
            
        int suma = cg::reduce(labeled_group, valor, cg::plus<int>());

        if (lane == 0) {
            y[segmento] = suma;
        }
    }
    
}


int main(int argc, char *argv[]){
    srand((unsigned int)time(NULL));

    int N = (1<<28);//2^28 
    int cambioLabel = 8; 
    int block = 256; 
    int a = 0;
    int b = 10;
    char v = 0;

    if (argc > 1) 
        N = atoi(argv[1]);
    if (argc > 2) 
        cambioLabel = atoi(argv[2]);
    if (argc > 3) 
        v = atoi(argv[3]);
    if (argc > 4) 
        block = atoi(argv[4]);
    if (argc > 5) 
        a = atoi(argv[5]);
    if (argc > 6) 
        b = atoi(argv[6]);

    int size = N * sizeof(int);
    int Nseg = (N + cambioLabel - 1) / cambioLabel; //tamño por segmento, ceil
    int sizeSeg = Nseg * sizeof(int);
    printf("Vector size: %d, Segment size: %d\n", N, sizeSeg);

    // Reservar memoria en host
    int * h_vector_x = (int *)malloc(size);
    int * h_vector_labels = (int *)malloc(size);
    int * h_vector_y = (int *)malloc(sizeSeg);

    int label = 0; 
    for (int i = 0; i <  N; i++) {
        h_vector_x[i] = (a + rand() % (b - a + 1)); //naturales de 0 a 10
        h_vector_labels[i] = label; 
        if ((i+1) % cambioLabel == 0) //etiqueta constante cada cambioLabel elementos luego aumenta.
            label++;
    }


    // Reservar memoria en device
    int *d_vector_x, *d_vector_y, *d_vector_labels;
	// reservar memoria en la GPU
	CUDA_CHK(cudaMalloc((void **)&d_vector_x, size));
	CUDA_CHK(cudaMalloc((void **)&d_vector_y, sizeSeg));
    CUDA_CHK(cudaMalloc((void **)&d_vector_labels, size));

	// copiar el de host a device
	CUDA_CHK(cudaMemcpy(d_vector_x, h_vector_x, size, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_vector_labels, h_vector_labels, size, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_vector_y, h_vector_y, sizeSeg, cudaMemcpyHostToDevice));
    //total_threads = #blocks * #threads_per_block
    //total_threads = N
	dim3 block_s(block); //

	dim3 grid_s((N + block - 1) / block); //si N siempre multiplo no importa block_s-1

    printf("N: %d, block: %d, grid: %d\n", N, block_s.x, grid_s.x);

    for (int i = 0; i < 10; i++) 
        kernel_redux_coop_g_labeled<<<grid_s, block_s>>>(d_vector_x, d_vector_y, d_vector_labels, N);
    
    CUDA_CHK(cudaDeviceSynchronize());

    // copiar el de device a host
	CUDA_CHK(cudaMemcpy(h_vector_y, d_vector_y, sizeSeg, cudaMemcpyDeviceToHost));

    //liberar mem gpu
	CUDA_CHK(cudaFree(d_vector_x));
	CUDA_CHK(cudaFree(d_vector_y));
    CUDA_CHK(cudaFree(d_vector_labels));
    // despliego la matriz resultante
    if (v) {
        printf("vector x:\n");
        for (int i = 0; i < N; i++) {
            printf("%d ", h_vector_x[i]);
        
        }
        printf("\n");
        printf("vector de indices:\n");
        for (int i = 0; i < N; i++) {
            printf("%d ", h_vector_labels[i]);
        }
        printf("\n");
        printf("vector y:\n");
        for (int i = 0; i < Nseg; i++) {
            printf("%d ", h_vector_y[i]);
        }
        printf("\n");
    }

	// libero la memoria en la CPU
	free(h_vector_y);
	free(h_vector_x);
    free(h_vector_labels);

	return 0;
}
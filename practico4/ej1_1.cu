#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define CUDA_CHK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while (0)

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);

        if (abort)
            exit(code);
    }
}

__global__ void exclusive_scan_block(const int *x, int *y,int *block_sums, int N)
{
    extern __shared__ int temp[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // cargar datos en shared
    if (gid < N)
        temp[tid] = x[gid];
    else
        temp[tid] = 0;

    __syncthreads();

    // Blelloch Scan - upsweep
    for (int offset = 1; offset < blockDim.x; offset *= 2)
    {
        int idx = (tid + 1) * offset * 2 - 1;

        if (idx < blockDim.x)
            temp[idx] += temp[idx - offset];

        __syncthreads();
    }

    // guardar suma total del bloque
    if (tid == 0) {
        block_sums[blockIdx.x] = temp[blockDim.x - 1];
        temp[blockDim.x - 1] = 0;
    }

    __syncthreads();

    // downsweep
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < blockDim.x){
            int t = temp[idx - offset];
            temp[idx - offset] = temp[idx];
            temp[idx] += t;
        }
        __syncthreads();
    }

    // escribir resultado
    if (gid < N)
        y[gid] = temp[tid];
}

__global__ void scan_block_sums(int *block_sums, int numBlocks){
    extern __shared__ int temp[];
    int tid = threadIdx.x;
    if (tid < numBlocks)
        temp[tid] = block_sums[tid];
    else
        temp[tid] = 0;
    __syncthreads();
    // upsweep
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < blockDim.x)
            temp[idx] += temp[idx - offset];
        __syncthreads();
    }
    // exclusive
    if (tid == 0)
        temp[blockDim.x - 1] = 0;
    __syncthreads();

    // downsweep
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < blockDim.x) {
            int t = temp[idx - offset];
            temp[idx - offset] = temp[idx];
            temp[idx] += t;
        }
        __syncthreads();
    }

    if (tid < numBlocks)
        block_sums[tid] = temp[tid];
}

__global__ void add_offsets(int *y,int *block_sums,int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N)
        y[gid] += block_sums[blockIdx.x];
}

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));
    int N = 1 << 20;
    int block = 256;
    int v = 0;
    if (argc > 1)
        N = atoi(argv[1]);
    if (argc > 2)
        block = atoi(argv[2]);
    if (argc > 3)
        v = atoi(argv[3]);
    int size = N * sizeof(int);
    printf("N = %d\n", N);
    // host
    int *h_x = (int*)malloc(size);
    int *h_y = (int*)malloc(size);
    for (int i = 0; i < N; i++)
        h_x[i] = rand() % 10;
    // device
    int *d_x;
    int *d_y;
    CUDA_CHK(cudaMalloc((void**)&d_x, size));
    CUDA_CHK(cudaMalloc((void**)&d_y, size));
    CUDA_CHK(cudaMemcpy(d_x,h_x,size,cudaMemcpyHostToDevice));
    int numBlocks = (N + block - 1) / block;
    int *d_block_sums;
    CUDA_CHK(cudaMalloc((void**)&d_block_sums, numBlocks * sizeof(int)));
    dim3 block_s(block);
    dim3 grid_s(numBlocks);
    // scan parcial por bloque
    exclusive_scan_block<<<grid_s,block_s,block * sizeof(int)>>>( d_x,d_y,d_block_sums,N);
    CUDA_CHK(cudaDeviceSynchronize());
    // scan de sumas de bloques
    int threads_scan = 1;
    while (threads_scan < numBlocks)
        threads_scan *= 2;
    scan_block_sums<<<1,threads_scan,threads_scan * sizeof(int)>>>(d_block_sums,numBlocks);
    CUDA_CHK(cudaDeviceSynchronize());
    // agregar offsets
    add_offsets<<<grid_s, block_s>>>( d_y,d_block_sums,N);
    CUDA_CHK(cudaDeviceSynchronize());
    // copiar resultado
    CUDA_CHK(cudaMemcpy(h_y,d_y,size,cudaMemcpyDeviceToHost));
    // imprimir
    if (v) {
        printf("x:\n");
        for (int i = 0; i < N; i++)
            printf("%d ", h_x[i]);
        printf("\n");
        printf("y:\n");
        for (int i = 0; i < N; i++)
            printf("%d ", h_y[i]);

        printf("\n");
    }

    // liberar
    CUDA_CHK(cudaFree(d_x));
    CUDA_CHK(cudaFree(d_y));
    CUDA_CHK(cudaFree(d_block_sums));

    free(h_x);
    free(h_y);

    return 0;
}
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>
#include <cub/cub.cuh>

#define CUDA_CHK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while (0)

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);

        if (abort)
            exit(code);
    }
}

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));

    int N = 1 << 20;
    int v = 0;

    if (argc > 1)
        N = atoi(argv[1]);

    if (argc > 2)
        v = atoi(argv[2]);

    int size = N * sizeof(int);

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

    CUDA_CHK(cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice));

    // CUB necesita memoria temporal
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    // consulta tamaño necesario
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_x, d_y, N);

    // reservar memoria temporal
    CUDA_CHK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // ejecutar exclusive scan
    cub::DeviceScan::ExclusiveSum( d_temp_storage, temp_storage_bytes, d_x, d_y, N);

    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaMemcpy(h_y, d_y, size,cudaMemcpyDeviceToHost));

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
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_temp_storage);

    free(h_x);
    free(h_y);

    return 0;
}
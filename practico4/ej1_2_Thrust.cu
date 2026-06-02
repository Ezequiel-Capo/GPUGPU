#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

#include <nvtx3/nvToolsExt.h>

#define CUDA_CHK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while (0)

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);

        if (abort)
            exit(code);
    }
}

int main(int argc, char *argv[]){
    srand((unsigned int)time(NULL));

    int N = 1 << 20;
    int v = 0;

    if (argc > 1)
        N = atoi(argv[1]);

    if (argc > 2)
        v = atoi(argv[2]);

    // host
    int *h_x = (int*)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++)
        h_x[i] = rand() % 10;

    // vectores device
    thrust::device_vector<int> d_x(h_x, h_x + N);

    thrust::device_vector<int> d_y(N);

    thrust::exclusive_scan(d_x.begin(), d_x.end(), d_y.begin());//WARM UP
    CUDA_CHK(cudaDeviceSynchronize()); //ns si sacarlo afuera
    // exclusive scan
    for (int i = 0; i < 10; i++){
        nvtxRangePush("ESCAN_THRUST");
        thrust::exclusive_scan(d_x.begin(), d_x.end(), d_y.begin());
        CUDA_CHK(cudaDeviceSynchronize());
        nvtxRangePop();
    }

    // copiar resultado
    thrust::host_vector<int> h_y = d_y;

    if (v){
        printf("x:\n");

        for (int i = 0; i < N; i++)
            printf("%d ", h_x[i]);

        printf("\n");

        printf("y:\n");

        for (int i = 0; i < N; i++)
            printf("%d ", h_y[i]);

        printf("\n");
    }

    free(h_x);

    return 0;
}

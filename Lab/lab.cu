#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define CUDA_CHK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while (0)

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__inline__ __device__
int warpReduceSum(int val)
{
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    return val;
}

__global__ void CalculateNormVector(uint8_t *matrix, int *norms, int n, int m) {
    int row = blockIdx.x;

    if (row >= n)
        return;

    int local_norm = 0;

    // Cada hilo procesa parte de la fila, tid 0, blockDim.x-1, 2.blockDim.x -1
    for (int col = threadIdx.x; col < m; col += blockDim.x){
        int v = matrix[row * m + col];
        local_norm += v * v;
    }

    // Reducción dentro del warp
    local_norm = warpReduceSum(local_norm);

    // Un entero por warp, 1024 / 32 = 32 warps, maximo hilos por bloque = 1024
    __shared__ int warpSums[32]; 

    int lane   = threadIdx.x & 31; // threadIdx.x % warpSize -> 31 = 2^5 - 1 (todos 1s) 
    int warpId = threadIdx.x >> 5; // warpId == 0 si threadIdx.x < 32

    if (lane == 0)
        warpSums[warpId] = local_norm;

    __syncthreads();

    // Primer warp reduce las sumas de los warps
    if (warpId == 0){
        int sum = (lane < (blockDim.x / 32)) ? warpSums[lane] : 0; // lane < warps_per_block, pongo warpSums[lane]  

        sum = warpReduceSum(sum); //ultima reduccion, sumo los resultados de los warps

        if (lane == 0)
            norms[row] = sum;
    }
}


void generate_genomic_matrix(uint8_t *matrix, int n, int m) {
    srand(42);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            matrix[i * m + j] = rand() % 3;   // 0,1,2 con igual probabilidad
        }
    }
}


void print_matrix(uint8_t *matrix, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            printf("%d ", matrix[i * m + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {

    int n = (1<<10);   // individuos
    int m = (1<<15);  // SNPs

    if (argc >= 3) {
        n = atoi(argv[1]);
        m = atoi(argv[2]);
    }

    //TODO: Implementar memoria a nivel de bits, 1 byte (4 SNPs)
    size_t size = n * m * sizeof(uint8_t);
    uint8_t *h_matrix = (uint8_t*)malloc(size);


    uint8_t *d_matrix;
    CUDA_CHK(cudaMalloc(&d_matrix, size));

    int *d_norms;
    CUDA_CHK(cudaMalloc(&d_norms, n * sizeof(int)));

    CalculateNormVector<<<n, 256>>>(d_matrix, d_norms, n, m);
    CUDA_CHK(cudaGetLastError());

    printf("Generando matriz genómica (%d x %d)...\n", n, m);

    generate_genomic_matrix(h_matrix, n, m);

    printf("\nMatriz decodificada (debug):\n");
    print_matrix(h_matrix, n, m);


    free(h_matrix);

    return 0;
}
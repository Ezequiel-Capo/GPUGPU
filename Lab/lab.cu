#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;
// Dimensiones de los tiles en memoria compartida
#define TILE_SIZE 32
// Dimensión nativa de Tensor Cores para int8
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16


#define CUDA_CHK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while (0)

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

//EUCLIDIAN DISTANCE CALCULATION X(mxn)X^T(nxm) = XXT(mxm) -> D^2 = ||N||^2 + ||N^T||^2 - 2*(XX^T)
__global__ void CalculateDistance(int32_t *XXT, int *norms, int *distances, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row <= col && row < m && col < m) {//triang sup        
        int v_normas = norms[row];
        int v_normas_trans = norms[col];
        int x_xt = XXT[row * m + col];

        distances[row * m + col] = v_normas + v_normas_trans - (2 * x_xt);
    //TODO: ver una forma de representar triang superior sin cargar ceros (averiguar si vale la pena formato Coo), o simplemente obviar esta parte, dejar con basura
    }else if (row < m && col < m) //triang inf, sin diag 
        distances[row * m + col] = 0;
        //distances[row * m + col] = distances[col * m + row]; //simetría
}


//MATMUL TENSOR CORES
__global__ void MatMul_WMMA(uint8_t *X, int32_t *C, int m, int n) {
    // Solo calcular la mitad superior de la matriz simétrica
    if (blockIdx.x < blockIdx.y) return; 

    __shared__ uint8_t shared_X[TILE_SIZE][TILE_SIZE + 1]; //PAD 

    // Declaración de fragmentos para los Tensor Cores
    //TODO: probar configs 
    //  X[8x32] * X^T[32x8] = C[8x8], (apunta a ser la mejor ya que |SNPs| >> |individuos| (n >> m))
    //  X[16x16] * X^T[16x16] = C[16x16] 
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_K, uint8_t, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_K, WMMA_N, WMMA_K, uint8_t, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag;

    // Inicializar el acumulador del resultado en 0
    wmma::fill_fragment(c_frag, 0);

    
   
}

//NORMAS
__inline__ __device__
int warpReduceSum(int val){
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

    //Inicialización variables------------
    int m = (1<<10);   // individuos
    int n = (1<<15);  // SNPs

    if (argc >= 3) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }

    size_t size = m * n * sizeof(uint8_t);
    uint8_t *h_matrix = (uint8_t*)malloc(size);
    uint8_t *d_matrix;
    
    CUDA_CHK(cudaMalloc(&d_matrix, size));

    //TODO: Implementar memoria a nivel de bits, 1 byte (4 SNPs)
    //Parte 0---------------------------
    //Simular matriz de individuosxSNPs, nxm, con valores 0,1,2
    printf("Generando matriz genómica (%d x %d)...\n", m, n);
    generate_genomic_matrix(h_matrix, m, n); //hace cpu only
    CUDA_CHK(cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice));
    printf("\nMatriz decodificada (debug):\n");
    print_matrix(h_matrix, m, n);

    //Parte 1---------------------------
    // Calcular normas de cada individuo
    int *d_norms;
    CUDA_CHK(cudaMalloc(&d_norms, m * sizeof(int)));


    dim3 block_s(256); 
	dim3 grid_s((m + block_s.x - 1) / block_s.x); 
    CalculateNormVector<<<grid_s, block_s>>>(d_matrix, d_norms, m, n);
    cuda_device_synchronize();

    //DEBUG normas:
    int *h_norms = (int*)malloc(m * sizeof(int)); //Puede no ser necesario, de momento lo dejo para debug
    cudaMemcpy(h_norms, d_norms, m * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nNormas de cada individuo:\n");
    for (int i = 0; i < m; i++) 
        printf("Norma del individuo %d: %d\n", i, h_norms[i]);
    
    //Parte 2---------------------------
    //DEFINIR XX^T
    dim3 block_s(256); 
    dim3 grid_s((m + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE); //podria ser m/Tsize , n/Tsize multiplos de 32
    MatMul_WMMA<<<grid_s, block_s>>>(d_matrix, d_XXT, m, n);
    cuda_device_synchronize();

    //TODO: DEBUG XX^T:

    //PARTE 3---------------------------
    //Cálculo de distancias usando normas y XX^T
    int *d_distances;
    CUDA_CHK(cudaMalloc(&d_distances, m * n * sizeof(int)));
    dim3 block_dist(16, 16); // 256 hilos en formato 2D
    dim3 grid_dist((m + 15) / 16, (m + 15) / 16); 
    CalculateDistance<<<grid_dist, block_dist>>>(d_XXT, d_norms, d_distances, m);
    cudaDeviceSynchronize();


    //TODO: DEBUG distancias:

    // Liberar memoria
    cudaFree(d_matrix);
    cudaFree(d_norms);
    free(h_matrix);
    free(h_norms);

    return 0;
}
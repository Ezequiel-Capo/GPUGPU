#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// Convencion de dimensiones:
//   X tiene forma m x n
//     m = individuos / filas
//     n = SNPs / columnas
//
// Queremos calcular:
//   XXT = X * X^T
//
// Eso es una SYRK/SYMRK: el resultado es m x m y simetrico, porque:
//   XXT[i,j] = dot(X[i,*], X[j,*]) = XXT[j,i]
//
// Por esa simetria calculamos y guardamos solo el triangulo superior.
// La matriz se reserva como m x m completa para mantener indices simples y
// escrituras regulares; el triangulo inferior queda en cero.
//
// Cada warp ejecuta una operacion WMMA 16x16x16:
//   A_frag  = 16 filas de X por 16 SNPs
//   B_frag  = 16 filas de X por 16 SNPs, leido como transpuesto
//   C_frag += A_frag * B_frag^T
//
// WARP_TILE_M y WARP_TILE_N eligen cuantos subtiles 16x16 hay por bloque.
// Cambiarlos cambia automaticamente cantidad de warps, hilos y tamano del tile:
//
//   WARP_TILE_M x WARP_TILE_N -> warps -> hilos -> tile C
//   2 x 2                     -> 4     -> 128   -> 32x32
//   2 x 4                     -> 8     -> 256   -> 32x64
//   4 x 2                     -> 8     -> 256   -> 64x32
//
// Para una salida triangular superior suele convenir WARP_TILE_N >= WARP_TILE_M:
// hacia la derecha de la diagonal hay mas trabajo util que debajo de ella.
#define WARP_TILE_M 2
#define WARP_TILE_N 2
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32
#define TILE_M (WARP_TILE_M * WMMA_M)
#define TILE_N (WARP_TILE_N * WMMA_N)
#define WARPS_PER_BLOCK (WARP_TILE_M * WARP_TILE_N)
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * WARP_SIZE)

// Anchos fisicos en shared memory.
//
// Los tiles A/B tienen ancho logico WMMA_K=16, pero se almacenan con stride 32.
// Ese padding cumple dos objetivos:
//   1. Mantener el leading dimension alineado para wmma::load_matrix_sync.
//   2. Evitar strides demasiado "redondos" respecto de los bancos de shared.
//
// c_tile tiene ancho logico TILE_N, pero se guarda con algunas columnas extra.
// WMMA escribe fragments completos 16x16; despues copiamos a global filtrando
// bordes y triangulo superior.
#define SHMEM_K 32
#define SHMEM_C_N (TILE_N + 8)

#if (WARP_TILE_M < 1) || (WARP_TILE_N < 1)
#error "WARP_TILE_M y WARP_TILE_N deben ser al menos 1."
#endif

#if THREADS_PER_BLOCK > 1024
#error "Demasiados warps por bloque: THREADS_PER_BLOCK no puede superar 1024."
#endif

#if SHMEM_K < WMMA_K
#error "SHMEM_K debe ser al menos WMMA_K."
#endif

#if SHMEM_C_N < TILE_N
#error "SHMEM_C_N debe ser al menos TILE_N."
#endif

#define CUDA_CHK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while (0)

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

static inline int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// EUCLIDEAN DISTANCE
// D^2(i,j) = ||X_i||^2 + ||X_j||^2 - 2 * dot(X_i, X_j)
// XXT ya contiene dot(X_i, X_j). Como las distancias tambien son simetricas,
// solo escribimos row <= col. d_distances se inicializa con cudaMemset, asi que
// el triangulo inferior queda en cero.
__global__ void CalculateDistance(const int32_t *XXT, const int *norms,
                                  int *distances, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < m && row <= col) {
        int xxt = XXT[row * m + col];
        distances[row * m + col] = norms[row] + norms[col] - 2 * xxt;
    }
}

// XXT = X * X^T usando Tensor Cores.
// Cada bloque calcula un tile TILE_M x TILE_N de C.
// Para cada bloque de SNPs k0:k0+15 cargamos en shared:
//   A = X[row_base:row_base+TILE_M, k0:k0+15] -> TILE_M x 16
//   B = X[col_base:col_base+TILE_N, k0:k0+15] -> TILE_N x 16
//
//   C_tile += A * B^T = X*X^T
//
// B se carga en WMMA como col_major. Eso equivale a usar B^T sin hacer una
// transposicion explicita: B[j][k] en row-major se ve como B_wmma[k][j].
__global__ void Symrk_WMMA_Shared(const uint8_t *X, int32_t *C, int m, int n) {
    // blockIdx.y elige el grupo de filas de C; blockIdx.x elige columnas.
    // Con tiles rectangulares, el eje x avanza de a TILE_N y el eje y de a TILE_M.
    int row_base = blockIdx.y * TILE_M;
    int col_base = blockIdx.x * TILE_N;

    // Si todo el tile esta bajo la diagonal, todos sus elementos cumplen
    // col < row. Como solo guardamos triangulo superior, podemos salir.
    if (col_base + TILE_N <= row_base) return;

    // Un warp completo ejecuta una sola operacion WMMA. warp_id identifica que
    // subtile 16x16 de C le toca a este warp dentro del bloque.
    int warp_id = threadIdx.x / WARP_SIZE;

    // Mapeo row-major de warps sobre la grilla de subtiles 16x16.
    // Ejemplo WARP_TILE_M=2, WARP_TILE_N=4:
    //   warp_id:        0      1      2      3
    //               [0,0]  [0,1]  [0,2]  [0,3]
    //   warp_id:        4      5      6      7
    //               [1,0]  [1,1]  [1,2]  [1,3]
    int warp_tile_row = warp_id / WARP_TILE_N;
    int warp_tile_col = warp_id % WARP_TILE_N;

    // Coordenada global del subtile 16x16 que calcula este warp.
    int subtile_row = row_base + warp_tile_row * WMMA_M;
    int subtile_col = col_base + warp_tile_col * WMMA_N;

    // Si el subtile completo esta bajo la diagonal, sus 16 columnas quedan antes
    // de su primera fila. Ese warp no llama a WMMA; igual participa en las
    // barreras para no desincronizar el bloque.
    bool compute_subtile = (subtile_col + WMMA_N > subtile_row);

    // Shared memory:
    //
    // a_tile/b_tile son caches por bloque de los pedazos de X que se reutilizan
    // por todos los warps del tile de salida. El ancho logico es WMMA_K=16, pero
    // el stride fisico es SHMEM_K para padding/alineacion.
    //
    // c_tile recibe los resultados de WMMA antes de escribir a global. Esto
    // permite que WMMA guarde fragments densos 16x16 y que luego filtremos
    // bordes y triangulo superior con codigo escalar simple.
    __shared__ __align__(32) uint8_t a_tile[TILE_M][SHMEM_K];
    __shared__ __align__(32) uint8_t b_tile[TILE_N][SHMEM_K];
    __shared__ __align__(32) int32_t c_tile[TILE_M][SHMEM_C_N];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, unsigned char, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, unsigned char, wmma::col_major> b_frag;
    // Para multiplicandos uint8_t, WMMA expone acumulador entero con tipo int.
    // unsigned int no esta soportado por nvcuda::wmma::fragment<accumulator>.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_frag;

    // Cada warp mantiene su acumulador en registros/fragments durante todo el
    // recorrido en k. No se escribe a shared hasta terminar la suma completa.
    wmma::fill_fragment(c_frag, 0);

    for (int k0 = 0; k0 < n; k0 += WMMA_K) {
        // Carga cooperativa de A = X[row_tile, k0:k0+15].
        //
        // Todos los hilos del bloque cargan elementos escalares. Si el tile cae
        // fuera de m o n, se rellena con cero para que WMMA pueda operar siempre
        // sobre fragments 16x16x16 completos.
        for (int idx = threadIdx.x; idx < TILE_M * WMMA_K; idx += blockDim.x) {
            int local_row = idx / WMMA_K;
            int local_k = idx % WMMA_K;
            int global_k = k0 + local_k;
            int a_row = row_base + local_row;

            a_tile[local_row][local_k] = (a_row < m && global_k < n) ? X[a_row * n + global_k] : 0;
        }

        // Carga cooperativa de B = X[col_tile, k0:k0+15].
        //
        // B tambien viene desde X, no desde una matriz transpuesta materializada.
        // La transposicion se logra en la carga WMMA usando matrix_b col_major.
        for (int idx = threadIdx.x; idx < TILE_N * WMMA_K; idx += blockDim.x) {
            int local_row = idx / WMMA_K;
            int local_k = idx % WMMA_K;
            int global_k = k0 + local_k;
            int b_row = col_base + local_row;

            b_tile[local_row][local_k] = (b_row < m && global_k < n) ? X[b_row * n + global_k] : 0;
        }

        // Asegura que a_tile y b_tile esten completos antes de que cualquier
        // warp haga load_matrix_sync.
        __syncthreads();

        if (compute_subtile) {
            const uint8_t *a_ptr = &a_tile[warp_tile_row * WMMA_M][0];
            const uint8_t *b_ptr = &b_tile[warp_tile_col * WMMA_N][0];

            // SHMEM_K es el leading dimension fisico, no el ancho logico. Si se
            // cambia el padding, este valor debe cambiar junto con el arreglo.
            wmma::load_matrix_sync(a_frag, a_ptr, SHMEM_K);
            wmma::load_matrix_sync(b_frag, b_ptr, SHMEM_K);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        // Ningun hilo puede sobreescribir shared antes de que todos los warps
        // terminen de consumir el k-tile actual.
        __syncthreads();
    }

    if (compute_subtile) {
        int local_row = warp_tile_row * WMMA_M;
        int local_col = warp_tile_col * WMMA_N;
        wmma::store_matrix_sync(&c_tile[local_row][local_col],
                                c_frag, SHMEM_C_N, wmma::mem_row_major);
    }

    // Espera a que todos los warps que calcularon subtiles hayan terminado de
    // guardar sus fragments en c_tile antes de la copia escalar a global.
    __syncthreads();

    // Copia global filtrando:
    //   1. Bordes cuando m no es multiplo de TILE_M/TILE_N.
    //   2. Triangulo inferior dentro de tiles que cruzan la diagonal.
    //
    // Como C se limpia con cudaMemset antes del kernel, todo lo no escrito queda
    // en cero.
    for (int idx = threadIdx.x; idx < TILE_M * TILE_N; idx += blockDim.x) {
        int local_row = idx / TILE_N;
        int local_col = idx % TILE_N;
        int global_row = row_base + local_row;
        int global_col = col_base + local_col;

        if (global_row < m && global_col < m && global_row <= global_col) {
            C[global_row * m + global_col] = c_tile[local_row][local_col];
        }
    }
}

// NORMAS
//
// norms[row] = ||X[row,*]||^2 = sum_k X[row,k]^2
//
// Cada bloque calcula una fila de X. Los hilos recorren la fila con stride
// blockDim.x y acumulan una suma parcial local. Luego reducimos:
//   1. dentro de cada warp con shuffle instructions;
//   2. entre warps usando shared memory;
//   3. el lane 0 del warp 0 escribe el resultado final.
__inline__ __device__ int warpReduceSum(int val) {
    // __shfl_down_sync mueve valores entre lanes del mismo warp sin usar shared.
    // offset: 16, 8, 4, 2, 1 produce una reduccion arbol.
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void CalculateNormVector(const uint8_t *matrix, int *norms,
                                    int m, int n) {
    int row = blockIdx.x;
    if (row >= m) return;

    int local_norm = 0;
    for (int col = threadIdx.x; col < n; col += blockDim.x) {
        int v = matrix[row * n + col];
        local_norm += v * v;
    }

    // Despues de esta llamada, solo lane 0 de cada warp tiene la suma completa
    // de ese warp. Los otros lanes contienen valores que ya no usamos.
    local_norm = warpReduceSum(local_norm);

    // Maximo 1024 hilos por bloque => maximo 32 warps.
    __shared__ int warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warps_per_block = (blockDim.x + warpSize - 1) / warpSize;

    if (lane == 0)
        warp_sums[warp_id] = local_norm;

    __syncthreads();

    if (warp_id == 0) {
        // El primer warp reduce las sumas de todos los warps del bloque. Los
        // lanes que no corresponden a un warp real aportan cero.
        int sum = (lane < warps_per_block) ? warp_sums[lane] : 0;
        sum = warpReduceSum(sum);

        if (lane == 0)
            norms[row] = sum;
    }
}

void generate_genomic_matrix(uint8_t *matrix, int m, int n) {
    srand(42);

    // Datos sinteticos: cada genotipo ocupa un byte con valores 0, 1 o 2.
    //TODO:compactar en bits
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = rand() % 3; // 0, 1, 2
        }
    }
}

void print_matrix_u8(const uint8_t *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%u ", (unsigned)matrix[i * cols + j]);
        }
        printf("\n");
    }
}

bool validate_small_case(const uint8_t *X, const int32_t *XXT, const int *norms, int m, int n) {
    // Validacion CPU para casos chicos. Solo revisa el triangulo superior porque
    // esa es la parte que el kernel define como salida valida.
    for (int i = 0; i < m; i++) {
        int ref_norm = 0;
        for (int k = 0; k < n; k++) {
            int v = X[i * n + k];
            ref_norm += v * v;
        }
        if (norms[i] != ref_norm) {
            printf("Error norma fila %d: GPU=%d CPU=%d\n", i, norms[i], ref_norm);
            return false;
        }

        for (int j = i; j < m; j++) {
            int32_t ref = 0;
            for (int k = 0; k < n; k++) 
                ref += (int32_t)X[i * n + k] * (int32_t)X[j * n + k];
            
            if (XXT[i * m + j] != ref) {
                printf("Error XXT(%d,%d): GPU=%d CPU=%d\n", i, j, XXT[i * m + j], ref);
                return false;
            }
        }
    }

    return true;
}

void print_matrix_i32(const int32_t  *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int m = (1 << 10);  // individuos
    int n = (1 << 15);  // SNPs

    if (argc >= 3) {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
    }

    if (m <= 0 || n <= 0) {
        fprintf(stderr, "Uso: %s [individuos m] [SNPs n]\n", argv[0]);
        return 1;
    }

    size_t matrix_elems = (size_t)m * (size_t)n;
    size_t matrix_bytes = matrix_elems * sizeof(uint8_t);
    size_t square_elems = (size_t)m * (size_t)m;
    size_t square_i32_bytes = square_elems * sizeof(int32_t);

    // Host:
    //   h_matrix: matriz X sintetica
    //   h_norms : copia de normas para validacion/debug
    //   h_XXT   : solo se reserva en casos chicos, para validar contra CPU
    uint8_t *h_matrix = (uint8_t*)malloc(matrix_bytes);
    int *h_norms = (int*)malloc((size_t)m * sizeof(int));
    int32_t *h_XXT = NULL;

    if (!h_matrix || !h_norms) {
        fprintf(stderr, "No hay memoria de host suficiente.\n");
        free(h_matrix);
        free(h_norms);
        return 1;
    }

    uint8_t *d_matrix = NULL;
    int *d_norms = NULL;
    int32_t *d_XXT = NULL;
    int *d_distances = NULL;

    // Device:
    //   d_XXT y d_distances se reservan como m x m completas. Solo el triangulo
    //   superior contiene valores utiles; el inferior se deja en cero.
    CUDA_CHK(cudaMalloc(&d_matrix, matrix_bytes));
    CUDA_CHK(cudaMalloc(&d_norms, (size_t)m * sizeof(int)));
    CUDA_CHK(cudaMalloc(&d_XXT, square_i32_bytes));
    CUDA_CHK(cudaMalloc(&d_distances, square_i32_bytes));

    printf("Generando matriz genomica X (%d individuos x %d SNPs)...\n", m, n);
    generate_genomic_matrix(h_matrix, m, n);


    CUDA_CHK(cudaMemcpy(d_matrix, h_matrix, matrix_bytes, cudaMemcpyHostToDevice));
    if (m <= 64 && n <= 128) {
        printf("\nX (debug):\n");
        print_matrix_u8(h_matrix, m, n);
    }

    dim3 block_norms(256);
    dim3 grid_norms(m);
    CalculateNormVector<<<grid_norms, block_norms>>>(d_matrix, d_norms, m, n);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    printf("Normas finalizado.\n");
    CUDA_CHK(cudaMemset(d_XXT, 0, square_i32_bytes));
    // SYMRK:
    //   grid.x recorre columnas de C en bloques TILE_N;
    //   grid.y recorre filas de C en bloques TILE_M;
    //   block.x tiene WARPS_PER_BLOCK warps, uno por subtile WMMA 16x16.
    dim3 block_syrk(THREADS_PER_BLOCK);
    dim3 grid_syrk(div_up(m, TILE_N), div_up(m, TILE_M));
    Symrk_WMMA_Shared<<<grid_syrk, block_syrk>>>(d_matrix, d_XXT, m, n);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaMemset(d_distances, 0, square_i32_bytes));

    printf("SYMRK con tensor cores finalizado\n");
    // Distancias euclideas al cuadrado usando las normas y XXT. Tambien solo
    // escribimos triangulo superior.
    dim3 block_dist(16, 16);
    dim3 grid_dist(div_up(m, block_dist.x), div_up(m, block_dist.y));
    CalculateDistance<<<grid_dist, block_dist>>>(d_XXT, d_norms, d_distances, m);
    CUDA_CHK(cudaGetLastError());
    CUDA_CHK(cudaDeviceSynchronize());

    CUDA_CHK(cudaMemcpy(h_norms, d_norms,
                        (size_t)m * sizeof(int), cudaMemcpyDeviceToHost));

    if (m <= 64 && n <= 1024) {
        h_XXT = (int32_t*)malloc(square_i32_bytes);
        if (!h_XXT) {
            fprintf(stderr, "No hay memoria de host para validar XXT.\n");
        } else {
            CUDA_CHK(cudaMemcpy(h_XXT, d_XXT, square_i32_bytes, cudaMemcpyDeviceToHost));

            bool ok = validate_small_case(h_matrix, h_XXT, h_norms, m, n);
            printf("Validacion CPU/GPU: %s\n", ok ? "OK" : "FALLO");
        }
    }

    if (m <= 64 && n <= 1024) {
         
        int *h_distances = (int*)malloc(square_i32_bytes); // MxM = Dmxm = X(mxn) * X^T(nxm) + N(m) + N^T(m)
        if (!h_distances) {
            fprintf(stderr, "No hay memoria de host para validar distancias.\n");
        } else {
            fprintf(stderr, "Distancias euclideas al cuadrado (debug):\n");
            CUDA_CHK(cudaMemcpy(h_distances, d_distances, square_i32_bytes, cudaMemcpyDeviceToHost));
            print_matrix_i32(h_distances, m, m);
            free(h_distances);
        }
    } 

    cudaFree(d_matrix);
    cudaFree(d_norms);
    cudaFree(d_XXT);
    cudaFree(d_distances);
    free(h_matrix);
    free(h_norms);
    free(h_XXT);

    return 0;
}

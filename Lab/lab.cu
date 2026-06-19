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

/*
    Codificación:
    0 -> 00
    1 -> 01
    2 -> 10
*/

/* -----------------------------
   Generación pseudo-realista
   -----------------------------
   Idea:
   - frecuencia alélica p aleatoria por SNP
   - genotipo:
       0 ~ (1-p)^2
       1 ~ 2p(1-p)
       2 ~ p^2
*/

__host__ uint8_t sample_genotype(float p, float r) {
    float p0 = (1.0f - p) * (1.0f - p);
    float p1 = 2.0f * p * (1.0f - p);
    // p2 = p^2

    if (r < p0) return 0;
    else if (r < p0 + p1) return 1;
    else return 2;
}

/*
   Empaquetado:
   4 SNPs por byte (2 bits cada uno)
*/

__host__ void generate_genomic_matrix(uint8_t *packed,int n, int m) {
    srand(42);

    int packed_m = m / 4; // 4 SNPs por byte

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < packed_m; j++) {

            uint8_t byte = 0;

            for (int k = 0; k < 4; k++) {
                int snp_idx = j * 4 + k; //indice de SNP a generar

                uint8_t g = 0;

                if (snp_idx < m) {
                    float p = (float)rand() / RAND_MAX; // frecuencia alélica
                    float r = (float)rand() / RAND_MAX; // muestreo

                    g = sample_genotype(p, r);
                }

                // empaquetar en 2 bits: g & 11, luego desplazar según posición en el byte
                byte |= (g & 0x3) << (k * 2);
            }

            packed[i * packed_m + j] = byte;
        }
    }
}

/* Debug opcional: imprimir matriz decodificada */
__host__ void print_matrix(uint8_t *packed, int n, int m) {
    int packed_m = (m + 3) / 4;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            int byte = packed[i * packed_m + (j / 4)];
            int shift = (j % 4) * 2;
            int val = (byte >> shift) & 0x3;

            printf("%d ", val);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {

    int n = 32;   // individuos
    int m = 32;  // SNPs

    if (argc >= 3) {
        n = atoi(argv[1]);
        m = atoi(argv[2]);
    }

    int packed_m = m / 4;

    size_t size = n * packed_m * sizeof(uint8_t);

    uint8_t *h_matrix = (uint8_t*)malloc(size);


    printf("Generando matriz genómica (%d x %d)...\n", n, m);

    generate_genomic_matrix(h_matrix, n, m);

    printf("\nMatriz decodificada (debug):\n");
    print_matrix(h_matrix, n, m);

    // /* GPU allocation (placeholder para pipeline futuro) */
    // uint8_t *d_matrix = nullptr;
    // CUDA_CHK(cudaMalloc(&d_matrix, size));
    // CUDA_CHK(cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice));

    // printf("\nTransferido a GPU correctamente.\n");

    // cudaFree(d_matrix);
    free(h_matrix);

    return 0;
}
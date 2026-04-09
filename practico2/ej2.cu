#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

// KERNEL 1D
__global__ void sumarSubmatriz1D(int* A, int n, int i1, int j1, int i2, int j2, int val) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int filas = i2 - i1 + 1;
    int cols  = j2 - j1 + 1;
    int total = filas * cols;

    if (idx < total) {
        int fila = idx / cols;
        int col  = idx % cols;

        int i = i1 + fila;
        int j = j1 + col;

        A[i * n + j] += val;
    }
}

// KERNEL 2D
__global__ void sumarSubmatriz2D(int* A, int n,int i1, int j1, int i2, int j2, int val) {

    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    int col  = blockIdx.x * blockDim.x + threadIdx.x;

    int filas = i2 - i1 + 1;
    int cols  = j2 - j1 + 1;

    if (fila < filas && col < cols) {
        int i = i1 + fila;
        int j = j1 + col;

        A[i * n + j] += val;
    }
}


// IMPRIMIR MATRIZ
void imprimirMatriz(int* A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%4d ", A[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}


int main() {

    int n = 6;
    int size = n * n * sizeof(int);

    int i1 = 1, j1 = 1;
    int i2 = 4, j2 = 4;
    int val = 10;

    int* A = (int*)malloc(size);

    for (int i = 0; i < n * n; i++) {
        A[i] = i;
    }

    printf("Matriz original:\n");
    imprimirMatriz(A, n);

    int* A_d;
    cudaMalloc((void**)&A_d, size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);

    int filas = i2 - i1 + 1;
    int cols  = j2 - j1 + 1;
    int total = filas * cols;

    int blockSize1D = 256; // múltiplo de 32
    int gridSize1D = (total + blockSize1D - 1) / blockSize1D;

    sumarSubmatriz1D<<<gridSize1D, blockSize1D>>>(
        A_d, n, i1, j1, i2, j2, val
    );

    cudaDeviceSynchronize();

    cudaMemcpy(A, A_d, size, cudaMemcpyDeviceToHost);

    printf("Despues de kernel 1D:\n");
    imprimirMatriz(A, n);

    for (int i = 0; i < n * n; i++) {
        A[i] = i;
    }

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    
    dim3 blockSize2D(32, 8); // 256 hilos (múltiplo de 32)

    dim3 gridSize2D(
        (cols + blockSize2D.x - 1) / blockSize2D.x,
        (filas + blockSize2D.y - 1) / blockSize2D.y
    );

    sumarSubmatriz2D<<<gridSize2D, blockSize2D>>>(
        A_d, n, i1, j1, i2, j2, val
    );

    cudaDeviceSynchronize();

    cudaMemcpy(A, A_d, size, cudaMemcpyDeviceToHost);

    printf("Despues de kernel 2D:\n");
    imprimirMatriz(A, n);

    // =======================
    // CLEANUP
    // =======================
    cudaFree(A_d);
    free(A);

    return 0;
}
#include <iostream>
#include <algorithm>
#include <chrono>
#include <new>

using namespace std;

void inicializarMatriz(float* matrix, int total) {
    for (int i = 0; i < total; i++)
        matrix[i] = 2.0f;
    
}

void inicializarMatrizCeros(float* matrix, int total) {
     for (int i = 0; i < total; i++) {
        matrix[i] = 0.0f;
    }
}

void matrix_mult_convencional(float *A, float *B, float *C, int N){
    int i,j,k;

    auto start = chrono::high_resolution_clock::now();

    for (i = 0; i < N; i++)
        for (k = 0; k < N; k++)
            for (j = 0; j < N; j++)
                C[i*N+j] += A[i*N+k] * B[k*N+j];

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Matrix multiplication time: " << elapsed.count() << " seconds" << endl;
}

void matrix_mult(float *A, float *B, float *C, int N, int BS) {
    int i,j,k,ii,jj,kk;

    auto start = chrono::high_resolution_clock::now();

    for (ii = 0; ii < N; ii += BS)
        for (jj = 0; jj < N; jj += BS)
            for (kk = 0; kk < N; kk += BS)

                for (i = ii; i < ii+BS; i++)
                    for (j = jj; j < jj+BS; j++)
                        for (k = kk; k < kk+BS; k++)
                            C[i*N+j] += A[i*N+k] * B[k*N+j];

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Matrix multiplication time: " << elapsed.count() << " seconds" << endl;
}

int main() {
    //cout << sizeof(float) << std::endl;
    //32 MB = 32 × 1024 × 1024 bytes (LLC) 2^25 -> (2^2) * 2^13 ¨* 2^13
    //16 MB = 16 × 1024 × 1024 bytes (LLC) 2^24 -> (2^2) * 2^12 ¨* 2^12
    const int DIM = (1 << 12); // 2^12 = 4096
    const int TOTAL = DIM * DIM;
    
    float* matrixA = new float[TOTAL];
    float* matrixB = new float[TOTAL];
    float* matrixC = new float[TOTAL];
    cout << "Matrix dimension: " << DIM << "x" << DIM << endl;
    inicializarMatriz(matrixA, TOTAL);
    inicializarMatriz(matrixB, TOTAL);
    
    for (int BS = 8; BS <= 512; BS *= 2) {
        inicializarMatrizCeros(matrixC, TOTAL);
        cout << "BS = " << BS << endl;
        matrix_mult(matrixA, matrixB, matrixC, DIM, BS);
    }
    
    inicializarMatrizCeros(matrixC, TOTAL);
    cout << "Convencional" << endl;
    matrix_mult_convencional(matrixA, matrixB, matrixC, DIM);


    delete[] matrixA;
    delete[] matrixB;
    delete[] matrixC;

    return 0;
    
}


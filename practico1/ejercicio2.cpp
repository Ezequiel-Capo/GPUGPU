#include <iostream>
#include <algorithm>
#include <chrono>

using namespace std;

void matrix_mult(float *A, float *B, float *C, int N, int BS) {
    int i,j,k,ii,jj,kk;

    auto start = chrono::high_resolution_clock::now();

    for (ii = 0; ii < N; ii += BS)
        for (jj = 0; jj < N; jj += BS)
            for (kk = 0; kk < N; kk += BS)

                for (i = ii; i < min(ii + BS, N); i++)
                    for (j = jj; j < min(jj + BS, N); j++)
                        for (k = kk; k < min(kk + BS, N); k++)
                            C[i*N+j] += A[i*N+k] * B[k*N+j];

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Matrix multiplication time: " << elapsed.count() << " seconds" << endl;
}

int main() {
    //cout << sizeof(float) << std::endl;
    //64 MB = 64 × 1024 × 1024 bytes (LLC) 2^26 -> (2^2) * 2^13 ¨* 2^13
    const int DIM = (1 << 13); // 2^13 = 8192
    const size_t TOTAL = static_cast<size_t>(DIM) * DIM;

    float* matrixA = new float[TOTAL]();
    float* matrixB = new float[TOTAL]();
    float* matrixC = new float[TOTAL]();
    
    int BS = 32; // 2^5
    matrix_mult(matrixA, matrixB, matrixC, DIM, BS);
    
    // for (int BS = 32; BS < (1 << 13); BS *= 4 ) {
    //     matrix_mult(matrixA, matrixB, matrixC, DIM, BS);
    // }                  

    delete[] matrixA;
    delete[] matrixB;
    delete[] matrixC;

    return 0;
    
}


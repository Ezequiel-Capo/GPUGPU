#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

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
    cout << sizeof(float) << std::endl;
    //64 MB = 64 × 1024 × 1024 bytes (LLC) 2^26 -> (2^2) * 2^13 ¨* 2^13
    const int DIM = (1 << 13); // 2^13 = 8192
    
    vector<vector<float>> matrixA(DIM, vector<float>(DIM));
    vector<vector<float>> matrixB(DIM, vector<float>(DIM));
    vector<vector<float>> matrixC(DIM, vector<float>(DIM));
    
    int BS = 32; // 2^5
    matrix_mult(&matrixA[0][0], &matrixB[0][0], &matrixC[0][0], DIM, BS);
    
    // for (int BS = 32; BS < (1 << 13); BS *= 4 ) {
    //     matrix_mult(&matrixA[0][0], &matrixB[0][0], &matrixC[0][0], DIM, BS);
    // }                  
    return 0;
    
}


#include <iostream>
#include <chrono>
#include <vector>

using namespace std;

void init(float* M, int total) {
    for (int i = 0; i < total; i++)
        M[i] = 2.0f;
}

void zero(float* M, int total) {
    for (int i = 0; i < total; i++)
        M[i] = 0.0f;
}

double mflops(int N, double seconds) {
    return (double)N * N * N / (seconds * 1e6);
}

double ijk(float *A,float *B,float *C,int N){
    int i,j,k;
    auto start = chrono::high_resolution_clock::now();

    for(i=0;i<N;i++)
        for(j=0;j<N;j++)
            for(k=0;k<N;k++)
                C[i*N+j]+=A[i*N+k]*B[k*N+j];

    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end-start).count();
}

double jik(float *A,float *B,float *C,int N){
    int i,j,k;
    auto start = chrono::high_resolution_clock::now();

    for(j=0;j<N;j++)
        for(i=0;i<N;i++)
            for(k=0;k<N;k++)
                C[i*N+j]+=A[i*N+k]*B[k*N+j];

    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end-start).count();
}

double ikj(float *A,float *B,float *C,int N){
    int i,j,k;
    auto start = chrono::high_resolution_clock::now();

    for(i=0;i<N;i++)
        for(k=0;k<N;k++)
            for(j=0;j<N;j++)
                C[i*N+j]+=A[i*N+k]*B[k*N+j];

    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end-start).count();
}

double kij(float *A,float *B,float *C,int N){
    int i,j,k;
    auto start = chrono::high_resolution_clock::now();

    for(k=0;k<N;k++)
        for(i=0;i<N;i++)
            for(j=0;j<N;j++)
                C[i*N+j]+=A[i*N+k]*B[k*N+j];

    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end-start).count();
}

double jki(float *A,float *B,float *C,int N){
    int i,j,k;
    auto start = chrono::high_resolution_clock::now();

    for(j=0;j<N;j++)
        for(k=0;k<N;k++)
            for(i=0;i<N;i++)
                C[i*N+j]+=A[i*N+k]*B[k*N+j];

    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end-start).count();
}

double kji(float *A,float *B,float *C,int N){
    int i,j,k;
    auto start = chrono::high_resolution_clock::now();

    for(k=0;k<N;k++)
        for(j=0;j<N;j++)
            for(i=0;i<N;i++)
                C[i*N+j]+=A[i*N+k]*B[k*N+j];

    auto end = chrono::high_resolution_clock::now();
    return chrono::duration<double>(end-start).count();
}

int main(){

    vector<int> sizes = {256,260,512,550,1024,1050};

    for(int N : sizes){

        int total = N*N;

        float* A = new float[total];
        float* B = new float[total];
        float* C = new float[total];

        init(A,total);
        init(B,total);

        cout << "\nMatrix " << N << "x" << N << endl;

        zero(C,total);
        double t = ijk(A,B,C,N);
        cout << "ijk  time: " << t << "  MFLOPS: " << mflops(N,t) << endl;

        zero(C,total);
        t = jik(A,B,C,N);
        cout << "jik  time: " << t << "  MFLOPS: " << mflops(N,t) << endl;

        zero(C,total);
        t = ikj(A,B,C,N);
        cout << "ikj  time: " << t << "  MFLOPS: " << mflops(N,t) << endl;

        zero(C,total);
        t = kij(A,B,C,N);
        cout << "kij  time: " << t << "  MFLOPS: " << mflops(N,t) << endl;

        zero(C,total);
        t = jki(A,B,C,N);
        cout << "jki  time: " << t << "  MFLOPS: " << mflops(N,t) << endl;

        zero(C,total);
        t = kji(A,B,C,N);
        cout << "kji  time: " << t << "  MFLOPS: " << mflops(N,t) << endl;

        delete[] A;
        delete[] B;
        delete[] C;
    }
}
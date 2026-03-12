#include <iostream>
#include <vector>

using namespace std;

int main() {
    //cout << sizeof(int) << std::endl;
    //64 MB = 64 × 1024 × 1024 bytes (LLC) 2^26 -> (2^7* 2^2) * (2^7* 2^2) * (2^7* 2^2) = 2^27 
    const int DIM = 128;
    
    vector<vector<vector<int>>> matrix1(DIM, vector<vector<int>>(DIM, vector<int>(DIM)));
    vector<vector<vector<int>>> matrix2(DIM, vector<vector<int>>(DIM, vector<int>(DIM)));
    vector<vector<vector<int>>> matrix3(DIM, vector<vector<int>>(DIM, vector<int>(DIM)));
    
    cout << "3 3D matrixes created successfully" << std::endl;
    
    return 0;
}
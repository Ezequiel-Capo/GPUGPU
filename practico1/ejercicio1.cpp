#include <iostream>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <random>

using namespace std;

const uint32_t SIZE = 100 * 1024 * 1024 / sizeof(char); // 100MB

void initSequentialIndex(uint32_t* index) {
    for (uint32_t i = 0; i < SIZE; i++) {
        index[i] = i;
    }
}

void shuffleIndex(uint32_t* index) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::shuffle(index, index + SIZE, gen);
}

double recorrerArray(char* array, uint32_t* index) {
    auto start = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < SIZE; i++) {
        array[index[i]]++;
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

int main() {

    uint32_t* index = new uint32_t[SIZE];

    // Parte 1 secuencial

    char* largeArraySecuencial = new char[SIZE];

    initSequentialIndex(index);

    std::cout << "Init secuencial\n";

    double tiempoSec = recorrerArray(largeArraySecuencial, index);

    std::cout << "Tiempo secuencial: " << tiempoSec << " s\n";

    delete[] largeArraySecuencial;


    // Parte 2 random

    char* largeArrayRandom = new char[SIZE];

    shuffleIndex(index);

    std::cout << "Init random\n";

    double tiempoRand = recorrerArray(largeArrayRandom, index);

    std::cout << "Tiempo random: " << tiempoRand << " s\n";

    delete[] largeArrayRandom;

    delete[] index;

    return 0;
}
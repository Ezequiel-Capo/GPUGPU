#include <iostream>
#include <chrono>
#include <algorithm>
#include <random>
#include <cstdint>

using namespace std;

const uint32_t SIZE = 100 * 1024 * 1024 / sizeof(char); // 100MB
const int RUNS = 5;

// Inicializa índices secuenciales
void initSequentialIndex(uint32_t* index) {
    for (uint32_t i = 0; i < SIZE; i++) {
        index[i] = i;
    }
}

// Mezcla índices (acceso aleatorio)
void shuffleIndex(uint32_t* index) {
    std::mt19937 gen(42); // semilla fija (reproducible)
    std::shuffle(index, index + SIZE, gen);
}

// Recorre el arreglo usando el patrón de índices
double recorrerArray(char* array, uint32_t* index) {
    auto start = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < SIZE; i++) {
        array[index[i]]++;
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    cout << "Tiempo recorrido: " << elapsed.count() << " s\n";
    return elapsed.count();
}

// Ejecuta varias corridas y promedia (con warm-up)
double medir(char* array, uint32_t* index) {
    // Warm-up (no se mide)
    recorrerArray(array, index);

    double total = 0.0;

    for (int i = 0; i < RUNS; i++) {
        total += recorrerArray(array, index);
    }

    return total / RUNS;
}

int main() {

    uint32_t* index = new uint32_t[SIZE];

    // =========================
    // Parte 1: Secuencial
    // =========================

    char* largeArraySecuencial = new char[SIZE];

    // Inicializar memoria: Asigna copias del valor especificado a todos los elementos dentro del rango [first, last). 
    std::fill(largeArraySecuencial, largeArraySecuencial + SIZE, 0);

    initSequentialIndex(index);

    cout << "Init secuencial\n";

    double tiempoSec = medir(largeArraySecuencial, index);

    cout << "Tiempo secuencial (promedio): " << tiempoSec << " s\n";

    delete[] largeArraySecuencial;

    // =========================
    // Parte 2: Random
    // =========================

    char* largeArrayRandom = new char[SIZE];

    // Inicializar memoria
    std::fill(largeArrayRandom, largeArrayRandom + SIZE, 0);

    shuffleIndex(index);

    cout << "Init random\n";

    double tiempoRand = medir(largeArrayRandom, index);

    cout << "Tiempo random (promedio): " << tiempoRand << " s\n";

    delete[] largeArrayRandom;

    delete[] index;

    return 0;
}
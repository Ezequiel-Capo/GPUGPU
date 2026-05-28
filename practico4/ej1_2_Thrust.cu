#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <thrust/device_vector.h>
#include <thrust/scan.h>

int main(int argc, char *argv[]){
    srand((unsigned int)time(NULL));

    int N = 1 << 20;
    int v = 0;

    if (argc > 1)
        N = atoi(argv[1]);

    if (argc > 2)
        v = atoi(argv[2]);

    // host
    int *h_x = (int*)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++)
        h_x[i] = rand() % 10;

    // vectores device
    thrust::device_vector<int> d_x(h_x, h_x + N);

    thrust::device_vector<int> d_y(N);

    // exclusive scan
    thrust::exclusive_scan(d_x.begin(), d_x.end(), d_y.begin());

    // copiar resultado
    thrust::host_vector<int> h_y = d_y;

    if (v){
        printf("x:\n");

        for (int i = 0; i < N; i++)
            printf("%d ", h_x[i]);

        printf("\n");

        printf("y:\n");

        for (int i = 0; i < N; i++)
            printf("%d ", h_y[i]);

        printf("\n");
    }

    free(h_x);

    return 0;
}
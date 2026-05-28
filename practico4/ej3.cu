#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "cuda.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>

#define K 10


struct Bin_selector { 
    __host__ __device__

    int operator() (int x) const {
        return x / K;
    }
};

#define CUDA_CHK(ans) do { gpuAssert((ans), __FILE__, __LINE__); } while (0)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void imprimir_y_guardar_csv(
    const int *input,
    const int *output,
    int N,
    const int *bin_counts,
    const int *bin_offsets,
    int num_bins,
    const char *csv_filename
) {
    FILE *f = fopen(csv_filename, "w");

    if (f == NULL) {
        printf("Error abriendo archivo CSV\n");
        return;
    }

    // encabezado CSV
    fprintf(f, "input,output,bin_counts,bin_offsets\n");

    printf("-------------------------------------------------------------\n");
    printf("%10s %10s %15s %15s\n",
           "input", "output", "bin_counts", "bin_offsets");
    printf("-------------------------------------------------------------\n");

    int max_rows = N;

    if (num_bins > max_rows) {
        max_rows = num_bins;
    }

    for (int i = 0; i < max_rows; i++) {

        // consola
        if (i < N)
            printf("%10d ", input[i]);
        else
            printf("%10s ", "");

        if (i < N)
            printf("%10d ", output[i]);
        else
            printf("%10s ", "");

        if (i < num_bins)
            printf("%15d ", bin_counts[i]);
        else
            printf("%15s ", "");

        if (i < num_bins)
            printf("%15d ", bin_offsets[i]);
        else
            printf("%15s ", "");

        printf("\n");

        // CSV
        if (i < N)
            fprintf(f, "%d,", input[i]);
        else
            fprintf(f, ",");

        if (i < N)
            fprintf(f, "%d,", output[i]);
        else
            fprintf(f, ",");

        if (i < num_bins)
            fprintf(f, "%d,", bin_counts[i]);
        else
            fprintf(f, ",");

        if (i < num_bins)
            fprintf(f, "%d", bin_offsets[i]);

        fprintf(f, "\n");
    }

    fclose(f);

    printf("-------------------------------------------------------------\n");
    printf("CSV guardado en: %s\n", csv_filename);
}


void inicializar_vector(int **v, int N)
{
    *v = (int*)malloc(N * sizeof(int));

    // 0,1,2,...,N-1
    for (int i = 0; i < N; i++) {
        (*v)[i] = i;
    }

    // shuffle determinístico
    for (int i = 0; i < N; i++) {

        int j = (37 * i + 13) % N;

        int tmp = (*v)[i];
        (*v)[i] = (*v)[j];
        (*v)[j] = tmp;
    }
}

int main(int argc, char *argv[]){

    int N = (1<<20);//2^20 

    char v = 0;

    if (argc > 1) 
        N = atoi(argv[1]);
    if (argc > 2) 
        v = atoi(argv[2]);

    int size = N * sizeof(int);
    

    printf("Vector size: %d, N");

    // Reservar memoria en host
    //int * h_vector_x = (int *)malloc(size);
    int * h_input;
    int * h_output = (int *) malloc(size);

    // se podria  usar thrust::generate(thrust::host, v.begin(), v.end(), rand);
    inicializar_vector(&h_input, N);

    thrust::device_vector<int> d_input(h_input, h_input + N);
    thrust::device_vector<int> d_bins(N);
    thrust::device_vector<int> d_output = d_input;

    // [20,12,2,7] -> [2,1,0,0]
    thrust::transform(d_input.begin(), d_input.end(), d_bins.begin(), Bin_selector()); //aplica la funcion a cada elemento del vector input y lo guarda en bins

    // [2,1,0,0] -> [0,0,1,2] && [20,12,7,2] -> [7,2,12,20], stable mantiene orden relativo. ACA no se si hay una func mejor ver luego
    thrust::stable_sort_by_key(d_bins.begin(), d_bins.end(), d_output.begin()); //ordena bins y reordena output a su vez.

    //ordenadados tomo el último
    int num_bins = d_bins.back() ;// *thrust::max_element(d_input.begin(), d_input.end());
    if (num_bins < N ) 
        num_bins += 1; 
    else
        return -1; 

    thrust::device_vector<int> d_bin_counts(num_bins); //creo bin_count con largo num_bins
    thrust::device_vector<int> d_bin_offsets(num_bins); //creo bin_offset con largo num_bins

    //Tengo que contar cuántos elementos hay en cada bin, para eso uso reduce_by_key
    //input keys -> bins: [0,0,0,0,1,1,2,3]
    //input values -> [1,1,1,1,1,1,1,1], o bueno 1 constante
    thrust::constant_iterator<int> val_iter(1);
    //descarto claves y me quedo con lo que es bin_count
    thrust::reduce_by_key(d_bins.begin(), d_bins.end(), val_iter, thrust::make_discard_iterator(), d_bin_counts.begin());


    //bin_counts = [4, 2, 1, 1] -> es usar excl sccn
    //bin_offsets = [0, 4, 6, 7]
    thrust::exclusive_scan(d_bin_counts.begin(), d_bin_counts.end(), d_bin_offsets.begin());


    //Copiar resultados a  host
    thrust::copy(d_output.begin(), d_output.end(), h_output);
    thrust::host_vector<int> h_bin_counts = d_bin_counts;
    thrust::host_vector<int> h_bin_offsets = d_bin_offsets;


    // despliego la matriz resultante
    if (v) {
        printf("input:\n");
        for (int i = 0; i < N; i++) {
            printf("%d ", h_input[i]);
        }
        printf("\n");

        printf("output:\n");
        for (int i = 0; i < N; i++) {
            printf("%d ", h_output[i]);
        }
        printf("\n");
        
        printf("bin counts:\n");
        for (int i = 0; i < num_bins; i++) {
            printf("%d ", h_bin_counts[i]);
        }
        printf("\n");
        
        printf("bins offset:\n");
        for (int i = 0; i < num_bins; i++) {
            printf("%d ", h_bin_offsets[i]);
        }
        printf("\n");
    }

    imprimir_y_guardar_csv(
        input,
        output,
        N,
        h_bin_counts,
        h_bin_offsets,
        num_bins,
        "resultado.csv"
    );
	// libero la memoria en la CPU
	free(h_output);
	free(h_input);
    free(h_bin_counts);
    free(h_bin_offsets);

	return 0;
}
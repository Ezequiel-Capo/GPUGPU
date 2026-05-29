#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define K 10

void agrupar_serial(
    const int *input,
    int N,
    int *output,
    int **bin_counts,
    int **bin_offsets,
    int *num_bins
) {
    int max_bin = 0;

    for (int i = 0; i < N; i++) {
        int bin = input[i] / K;
        if (bin > max_bin) {
            max_bin = bin;
        }
    }

    *num_bins = max_bin + 1;

    *bin_counts = (int *)calloc(*num_bins, sizeof(int));
    *bin_offsets = (int *)malloc((*num_bins) * sizeof(int));

    for (int i = 0; i < N; i++) {
        int bin = input[i] / K;
        (*bin_counts)[bin]++;
    }

    (*bin_offsets)[0] = 0;
    for (int b = 1; b < *num_bins; b++) {
        (*bin_offsets)[b] = (*bin_offsets)[b - 1] + (*bin_counts)[b - 1];
    }

    int *pos_actual = (int *)malloc((*num_bins) * sizeof(int));

    for (int b = 0; b < *num_bins; b++) {
        pos_actual[b] = (*bin_offsets)[b];
    }

    for (int i = 0; i < N; i++) {
        int bin = input[i] / K;
        int pos = pos_actual[bin]++;
        output[pos] = input[i];
    }

    free(pos_actual);
}

#include <stdio.h>

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



int main() {


    int N = 1000;

    int* input;

    inicializar_vector(&input, N);



    int *output = (int *)malloc(N * sizeof(int));
    int *bin_counts = NULL;
    int *bin_offsets = NULL;
    int num_bins = 0;

    for (int i = 0; i < 10; i++) {
        agrupar_serial(
            input,
            N,
            output,
            &bin_counts,
            &bin_offsets,
            &num_bins);
    }


    imprimir_y_guardar_csv(
        input,
        output,
        N,
        bin_counts,
        bin_offsets,
        num_bins,
        "resultado.csv"
    );

    free(output);
    free(bin_counts);
    free(bin_offsets);

    return 0;
}
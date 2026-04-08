#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__) }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void read_file(const char*, int*);
int get_text_length(const char * fname);

#define A 15
#define B 27
#define M 256
#define A_MMI_M -17

__device__ int modulo(int a, int b){
	int r = a % b;
	r = (r < 0) ? r + b : r;
	return r;
}

__global__ void decrypt_kernel(int *d_message, int length)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < length) {
		d_message[idx] = modulo(A_MMI_M * (d_message[idx] - B), M);
	}
}

int main(int argc, char *argv[])
{
	int *h_message; 
	int *d_message;
	unsigned int size;

	const char * fname = "secreto.txt"; 

	if (argc < 2) printf("Debe ingresar el nombre del archivo\n");
	else
		fname = argv[1];

	int length = get_text_length(fname);

	size = length * sizeof(int);

	// reservar memoria para el mensaje, host
	h_message = (int *)malloc(size);

	// leo el archivo de la entrada
	read_file(fname, h_message);

	// reservar memoria en la GPU
	cudaMalloc((void **)&d_message, size);

	// copiar el mensaje a la GPU
	cudaMemcpy(d_message, h_message, size, cudaMemcpyHostToDevice);


	// lanzar el kernel para desencriptar el mensaje
	dim3 block_s(128); //se toma rtx 2080ti, 1024threads por bloque, tomó 128 threads por bloque
	dim3 grid_s((length + block_s.x - 1) / block_s.x); //cantidad de bloques necesarios para procesar el mensaje, redondeo forzado hacia arriba

	decrypt_kernel<<<grid_s, block_s>>>(d_message, length);

	// copiar el mensaje desencriptado a la CPU
	cudaMemcpy(h_message, d_message, size, cudaMemcpyDeviceToHost);

	// liberar la memoria en la GPU
	cudaFree(d_message);

	// despliego el mensaje
	for (int i = 0; i < length; i++) {
		printf("%c", (char)h_message[i]);
	}
	printf("\n");

	// libero la memoria en la CPU
	free(h_message);

	return 0;
}

	
int get_text_length(const char * fname)
{
	FILE *f = NULL;
	f = fopen(fname, "r"); //read and binary flags

	size_t pos = ftell(f);    
	fseek(f, 0, SEEK_END);    
	size_t length = ftell(f); 
	fseek(f, pos, SEEK_SET);  

	fclose(f);

	return length;
}

void read_file(const char * fname, int* input)
{
	// printf("leyendo archivo %s\n", fname );

	FILE *f = NULL;
	f = fopen(fname, "r"); //read and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find %s file \n", fname);
		exit(1);
	}

	int c; 
	while ((c = getc(f)) != EOF) {
		*(input++) = c;
	}

	fclose(f);
}

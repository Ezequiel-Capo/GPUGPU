#!/bin/bash
set -e

printf "Casos de pruebas, ejecutnado\n"
job_ej1=$(sbatch --parsable --output=ej1.out --error=ej1.err --open-mode=truncate lanzar.sh "./ej1" "secreto.txt")

job_ej2=$(sbatch --parsable --output=ej2.out --error=ej2.err --open-mode=truncate lanzar.sh "./ej2" 32 0 0 4 4 10 4 8)

#mostrar que funciona
job_e3_1=$(sbatch --parsable --output=ej3_caso1.out --error=ej3_caso1.err --open-mode=truncate lanzar.sh "./ej3" 8 8 8 8 1)

job_e3_2=$(sbatch --parsable --output=ej3_caso6.out --error=ej3_caso6.err --open-mode=truncate lanzar.sh "./ej3" 16384 16384 32 32)

job_e3_3=$(sbatch --parsable --output=ej3_caso8.out --error=ej3_caso8.err --open-mode=truncate lanzar.sh "./ej3" 16384 16384 4 64)


printf "Jobs enviados: ej1=%s,  ej2=%s, e3_1=%s, e3_2=%s, e3_3=%s\n" "$job_ej1"  "$job_ej2" "$job_e3_1" "$job_e3_2" "$job_e3_3"

for jid in "$job_ej1" "$job_ej2" "$job_e3_1" "$job_e3_2" "$job_e3_3"; do
	while squeue -h -j "$jid" | grep -q .; do
		sleep 1
	done
done

printf "Resultados de las pruebas\n"

printf "Ejercicio 1)\n\n"
cat ej1.out
printf "\n"

printf "Ejercicio 2)\n\n"
cat ej2.out
printf "\n"


printf "Ejercicio 3)\n\n"
printf "E3-1) Prueba con tamaños Nx=8, Ny=8 para ver transposición correcta\n"
cat ej3_caso1.out
printf "\n"

printf "E3-6) Prueba con tamaños Nx=16384, Ny=16384, blockDim=(32,32) para evaluar rendimiento\n"
cat ej3_caso6.out
printf "\n"

printf "E3-8) Prueba con tamaños Nx=16384, Ny=16384, blockDim=(4,64) para evaluar rendimiento\n"
cat ej3_caso8.out
printf "\n"
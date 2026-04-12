#!/bin/bash
set -e

printf "Casos de pruebas, ejecutnado\n"
job_ej1=$(sbatch --parsable --output=ej1.out --error=ej1.err --open-mode=truncate lanzar.sh "./ej1" "secreto.txt")

job_ej2=$(sbatch --parsable --output=ej2.out --error=ej2.err --open-mode=truncate lanzar.sh "./ej2" 10 1 1 4 4 10)

job_e3_1=$(sbatch --parsable --output=ej3_caso1.out --error=ej3_caso1.err   --open-mode=truncate lanzar.sh "./ej3" 8 8 8 8 1)

job_e3_2=$(sbatch --parsable --output=ej3_caso2.out --error=ej3_caso2.err --open-mode=truncate lanzar.sh "./ej3" 64 64 32 32)
job_e3_3=$(sbatch --parsable --output=ej3_caso3.out --error=ej3_caso3.err --open-mode=truncate lanzar.sh "./ej3" 1024 1024 32 32)
job_e3_4=$(sbatch --parsable --output=ej3_caso4.out --error=ej3_caso4.err --open-mode=truncate lanzar.sh "./ej3" 2048 2048 32 32)
job_e3_5=$(sbatch --parsable --output=ej3_caso5.out --error=ej3_caso5.err --open-mode=truncate lanzar.sh "./ej3" 8192 8192 32 32)
job_e3_6=$(sbatch --parsable --output=ej3_caso6.out --error=ej3_caso6.err --open-mode=truncate lanzar.sh "./ej3" 16384 16384 32 32)

job_e3_7=$(sbatch --parsable --output=ej3_caso7.out --error=ej3_caso7.err --open-mode=truncate lanzar.sh "./ej3" 8192 8192 4 64)
job_e3_8=$(sbatch --parsable --output=ej3_caso8.out --error=ej3_caso8.err --open-mode=truncate lanzar.sh "./ej3" 16384 16384 4 64)


printf "Jobs enviados: ej1=%s, e3_1=%s, ej2=%s, e3_2=%s, e3_3=%s, e3_4=%s, e3_5=%s, e3_6=%s, e3_7=%s, e3_8=%s\n" "$job_ej1"  "$job_ej2" "$job_e3_1" "$job_e3_2" "$job_e3_3" "$job_e3_4" "$job_e3_5" "$job_e3_6" "$job_e3_7" "$job_e3_8"

for jid in "$job_ej1" "$job_ej2" "$job_e3_1" "$job_e3_2" "$job_e3_3" "$job_e3_4" "$job_e3_5" "$job_e3_6" "$job_e3_7" "$job_e3_8"; do
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

printf "E3-1) Prueba con tamaños Nx=64, Ny=64 y bloques de 32x32\n"
cat ej3_caso1.out
printf "\n"

printf "E3-3) Prueba con tamaños Nx=1024, Ny=1024 y bloques de 32x32\n"
cat ej3_caso2.out
printf "\n"

printf "E3-4) Prueba con tamaños Nx=2048, Ny=2048 y bloques de 32x32\n"
cat ej3_caso3.out
printf "\n"

printf "E3-5) Prueba con tamaños Nx=8192, Ny=8192 y bloques de 32x32\n"
cat ej3_caso4.out
printf "\n"

printf "E3-6) Prueba con tamaños Nx=16384, Ny=16384 y bloques de 32x32\n"
cat ej3_caso5.out
printf "\n"

printf "E3-7) Prueba con tamaños Nx=16384, Ny=16384 y bloques de 4x64\n"
cat ej3_caso6.out
printf "\n"

printf "E3-8) Prueba con tamaños Nx=8192, Ny=8192 y bloques de 4x64\n"
cat ej3_caso7.out
printf "\n"

printf "E3-9) Prueba con tamaños Nx=16384, Ny=16384 y bloques de 4x64\n"
cat ej3_caso8.out
printf "\n"


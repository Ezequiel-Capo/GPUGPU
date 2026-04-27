#!/bin/bash
set -e

printf "Casos de pruebas, ejecutnado\n"
#no padding prueba de funcionamiento 10x10
job_e1=$(sbatch --parsable --output=e1.out --error=e1.err --open-mode=truncate lanzar.sh "./ej1" 0 10 1)

#ejecucion timepos no padding 16384x16384
#job_e1_2=$(sbatch --parsable --output=e1_2.out --error=e1_2.err --open-mode=truncate lanzar.sh  nsys profile --stats=true -o transposicion_shared_nopad "./ej1" 0 )

#ejecucion timepos padding 16384x16384
#job_e1_3=$(sbatch --parsable --output=e1_3.out --error=e1_3.err --open-mode=truncate lanzar.sh  nsys profile --stats=true -o transposicion_shared_pad "./ej1" 1)

#mostrar que funciona shared mem
job_e2=$(sbatch --parsable --output=e2_1.out --error=e2_1.err --open-mode=truncate lanzar.sh "./ej2" 64 1)

#ejecucion timepos  16384x16384
#job_e2_1=$(sbatch --parsable --output=e2_2.out --error=e2_1.err --open-mode=truncate lanzar.sh  nsys profile --stats=true -o shared_memory_16384 "./ej2")

#mostrar que funciona warp shuffle
job_e3=$(sbatch --parsable --output=e3_1.out --error=e3_1.err --open-mode=truncate lanzar.sh "./ej2_2" 64 1)

#ejecucion timepos  16384x16384
#job_e3_2=$(sbatch --parsable --output=e3_2.out --error=e3_2.err --open-mode=truncate lanzar.sh  nsys profile --stats=true -o warp_shuffle_16384 "./ej2_2")

job_e1_2=$(sbatch --parsable \
  --output=e1_2.out --error=e1_2.err \
  lanzar.sh "./ej1" --nsys transposicion_shared_nopad 0)

job_e1_3=$(sbatch --parsable \
  --output=e1_3.out --error=e1_3.err \
  lanzar.sh "./ej1" --nsys transposicion_shared_pad 1)

job_e2_1=$(sbatch --parsable \
  --output=e2_2.out --error=e2_2.err \
  lanzar.sh "./ej2" --nsys shared_memory_16384)

job_e3_2=$(sbatch --parsable \
  --output=e3_2.out --error=e3_2.err \
  lanzar.sh "./ej2_2" --nsys warp_shuffle_16384)

printf "Jobs enviados: e1=%s, e1_2=%s, e1_3=%s, e2=%s, e2_1=%s, e3=%s, e3_2=%s\n" "$job_e1" "$job_e1_2" "$job_e1_3" "$job_e2" "$job_e2_1" "$job_e3" "$job_e3_2"

for jid in "$job_e1" "$job_e1_2" "$job_e1_3" "$job_e2" "$job_e2_1" "$job_e3" "$job_e3_2"; do
	while squeue -h -j "$jid" | grep -q .; do
		sleep 1
	done
done

printf "Resultados de las pruebas\n"

printf "Ejercicio 1-1: shared memory 10x10 (trasposicion))\n\n"
cat e1.out
printf "\n"

printf "Ejercicio 1-2: shared memory 2^14x2^14 (sin padding)\n\n"
echo " Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                    Name      \n"
grep kernel e1_2.out
printf "\n"

printf "Ejercicio 1-3: shared memory 2^14x2^14 con padding)\n\n"
echo " Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                    Name      \n"
grep kernel e1_3.out
printf "\n"

printf "Ejercicio 2 parte 1-1: shared memory arreglo de 64 elementos)\n\n"
cat e2_1.out
printf "\n"

printf "Ejercicio 2 parte 1-2: shared memory arreglo de 2^14 elementos)\n\n"
echo " Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                    Name      \n"
grep kernel e2_2.out
printf "\n"

printf "Ejercicio 2 parte 2-1: warp shuffle arreglo de 64 elementos)\n\n"
cat e3_1.out
printf "\n"

printf "Ejercicio 2 parte 2-2: warp shuffle arreglo de 2^14)\n\n"
echo " Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                    Name      \n"
grep kernel e3_2.out
printf "\n"
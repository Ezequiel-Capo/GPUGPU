#!/bin/bash
set -e

K_VALUES=(
	$((1<<11))
	$((1<<12))
	$((1<<13))
	$((1<<14))
	$((1<<15))
	$((1<<16))
	$((1<<17))
	$((1<<18))
	$((1<<19))
	$((1<<20))
)

E3_EXTRA_N_VALUES=(
	10000
	100000
	1000000
	10000000
)

jobs=()

submit_job() {
	local out="$1"
	local err="$2"
	shift 2

	jobs+=("$(sbatch --parsable --output="$out" --error="$err" --open-mode=truncate lanzar.sh "$@")")
}

print_time_header() {
	printf " Time (%%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                    Name\n"
}

grep_or_notice() {
	local pattern="$1"
	local file="$2"

	if ! grep -iE "$pattern" "$file"; then
		printf "No se encontraron entradas para '%s' en %s\n" "$pattern" "$file"
	fi
}

printf "Casos de pruebas, ejecutando\n"

# Casos funcionamiento
submit_job "e1.out" "e1.err" "./ej1_1" 100 256 1
submit_job "e1_2.out" "e1_2.err" "./ej1_2" 100 1
submit_job "e1_3.out" "e1_3.err" "./ej1_3" 100 1

# Casos ejecucion K = 1,..,10
for i in "${!K_VALUES[@]}"; do
	idx=$((i + 1))
	k_val="${K_VALUES[$i]}"

	submit_job "e1_k${idx}.out" "e1_k${idx}.err" "./ej1_1" --nsys "scan_manual_k${idx}" "$k_val"
	submit_job "e1_2_k${idx}.out" "e1_2_k${idx}.err" "./ej1_2" --nsys "scan_cub_k${idx}" "$k_val"
	submit_job "e1_3_k${idx}.out" "e1_3_k${idx}.err" "./ej1_3" --nsys "scan_thrust_k${idx}" "$k_val"
done

# Cooperative Groups funcionamiento
submit_job "e2.out" "e2.err" "./ej2" 64 1 64
submit_job "e2_1.out" "e2_1.err" "./ej2_1" 64 4 1 64

# Bins funcionamiento y tiempos
submit_job "e3.out" "e3.err" "./ej3" 64 1
submit_job "e3_c.out" "e3_c.err" "./ej3" --nsys "bins_paralelo" 1000 0
submit_job "e3sec_c.out" "e3sec_c.err" "./codigoBins" --nsys "bins_secuencial" 1000 0

for n in "${E3_EXTRA_N_VALUES[@]}"; do
	submit_job "e3_c_${n}.out" "e3_c_${n}.err" "./ej3" --nsys "bins_paralelo_${n}" "$n" 0
	submit_job "e3sec_c_${n}.out" "e3sec_c_${n}.err" "./codigoBins" --nsys "bins_secuencial_${n}" "$n" 0
done

printf "Jobs enviados correctamente\n"

for jid in "${jobs[@]}"; do
	while squeue -h -j "$jid" | grep -q .; do
		sleep 1
	done
done

printf "Resultados de las pruebas\n"

printf "Ejercicio 1: Funcionamiento escan manual\n\n"
cat e1.out
printf "\n"

printf "Ejercicio 1: Funcionamiento escan Cub\n\n"
cat e1_2.out
printf "\n"

printf "Ejercicio 1: Funcionamiento escan Thrust\n\n"
cat e1_3.out
printf "\n"

printf "Ejercicio 1: Tiempos de ejecucion =================\n\n"
for i in "${!K_VALUES[@]}"; do
	idx=$((i + 1))
	k_val="${K_VALUES[$i]}"

	printf "Escan manual, k = %s\n" "$k_val"

	print_time_header
	grep_or_notice "ESCAN" "e1_k${idx}.out"
	printf "\n"


done

for i in "${!K_VALUES[@]}"; do
	idx=$((i + 1))
	k_val="${K_VALUES[$i]}"

	printf "Escan Cub, k = %s\n" "$k_val"

	print_time_header
	grep_or_notice 	"ESCAN_CUB" "e1_2_k${idx}.out"
	printf "\n"



done

for i in "${!K_VALUES[@]}"; do
	idx=$((i + 1))
	k_val="${K_VALUES[$i]}"

	printf "Escan Thrust, k = %s\n" "$k_val"

	print_time_header
	grep_or_notice "ESCAN_THRUST" "e1_3_k${idx}.out"
	printf "\n"


done

printf "Ejercicio 2: Cooperative Groups, grupos de 8 por grupo, N = 64\n\n"
cat e2.out
printf "\n"

printf "Ejercicio 2: Cooperative Groups Labeled, cambio Label cada 4 elementos, N = 64\n\n"
cat e2_1.out
printf "\n"

printf "Ejercicio 3: Funcionamiento Bins, N = 64\n\n"
cat e3.out
printf "\n"

printf "Ejercicio 3: Bins Thrust vs Bins Secuencial, N = 1000\n\n"
print_time_header
grep_or_notice "B_PARALELO" "e3_c.out"
printf "\n"

printf "Bins secuenciales CodigoBins, N = 1000\n"
print_time_header
grep_or_notice "B_SECUENCIAL" "e3sec_c.out"
printf "\n"

for n in "${E3_EXTRA_N_VALUES[@]}"; do
	printf "Ejercicio 3: Bins Thrust vs Bins Secuencial, N = %s\n\n" "$n"
	print_time_header
	grep_or_notice "B_PARALELO" "e3_c_${n}.out"
	printf "\n"

	printf "Bins secuenciales CodigoBins, N = %s\n" "$n"
	print_time_header
	grep_or_notice "B_SECUENCIAL" "e3sec_c_${n}.out"
	printf "\n"
done

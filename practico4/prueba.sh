#!/bin/bash
set -e

K1=$((1<<11))
K2=$((1<<12))
K3=$((1<<13))
K4=$((1<<14))
K5=$((1<<15))
K6=$((1<<16))
K7=$((1<<17))
K8=$((1<<18))
K9=$((1<<19))
K10=$((1<<20))


printf "Casos de pruebas, ejecutando\n"

#casos funcionamiento
#scan manual
job_e1=$(sbatch --parsable --output=e1.out --error=e1.err --open-mode=truncate lanzar.sh "./ej1" 100 1)

#scan Cub
job_e2=$(sbatch --parsable --output=e1_2.out --error=e1_2.err --open-mode=truncate lanzar.sh "./ej1_2" 100 1)

#scan Thrust
job_e3=$(sbatch --parsable --output=e1_3.out --error=e1_3.err --open-mode=truncate lanzar.sh "./ej1_3" 100 1)

#casos ejecucion K = 1,..,10 scan a mano
job_e4=$(sbatch --parsable --output=e1_k1.out --error=e1_k1.err lanzar.sh "./ej1" --nsys scan_k $K1)
job_e5=$(sbatch --parsable --output=e1_k2.out --error=e1_k2.err lanzar.sh "./ej1" --nsys scan_k $K2)
job_e6=$(sbatch --parsable --output=e1_k3.out --error=e1_k3.err lanzar.sh "./ej1" --nsys scan_k $K3)
job_e7=$(sbatch --parsable --output=e1_k4.out --error=e1_k4.err lanzar.sh "./ej1" --nsys scan_k $K4)
job_e8=$(sbatch --parsable --output=e1_k5.out --error=e1_k5.err lanzar.sh "./ej1" --nsys scan_k $K5)
job_e9=$(sbatch --parsable --output=e1_k6.out --error=e1_k6.err lanzar.sh "./ej1" --nsys scan_k $K6)
job_e10=$(sbatch --parsable --output=e1_k7.out --error=e1_k7.err lanzar.sh "./ej1" --nsys scan_k $K7)
job_e11=$(sbatch --parsable --output=e1_k8.out --error=e1_k8.err lanzar.sh "./ej1" --nsys scan_k $K8)
job_e12=$(sbatch --parsable --output=e1_k9.out --error=e1_k9.err lanzar.sh "./ej1" --nsys scan_k $K9)
job_e13=$(sbatch --parsable --output=e1_k10.out --error=e1_k10.err lanzar.sh "./ej1" --nsys scan_k $K10)

#casos ejecucion K = 1,..,10 scan con cub
job_e14=$(sbatch --parsable --output=e1_2_k1.out --error=e1_2_k1.err lanzar.sh "./ej1_2" --nsys scan_k $K1)
job_e15=$(sbatch --parsable --output=e1_2_k2.out --error=e1_2_k2.err lanzar.sh "./ej1_2" --nsys scan_k $K2)
job_e16=$(sbatch --parsable --output=e1_2_k3.out --error=e1_2_k3.err lanzar.sh "./ej1_2" --nsys scan_k $K3)
job_e17=$(sbatch --parsable --output=e1_2_k4.out --error=e1_2_k4.err lanzar.sh "./ej1_2" --nsys scan_k $K4)
job_e18=$(sbatch --parsable --output=e1_2_k5.out --error=e1_2_k5.err lanzar.sh "./ej1_2" --nsys scan_k $K5)
job_e19=$(sbatch --parsable --output=e1_2_k6.out --error=e1_2_k6.err lanzar.sh "./ej1_2" --nsys scan_k $K6)
job_e20=$(sbatch --parsable --output=e1_2_k7.out --error=e1_2_k7.err lanzar.sh "./ej1_2" --nsys scan_k $K7)
job_e21=$(sbatch --parsable --output=e1_2_k8.out --error=e1_2_k8.err lanzar.sh "./ej1_2" --nsys scan_k $K8)
job_e22=$(sbatch --parsable --output=e1_2_k9.out --error=e1_2_k9.err lanzar.sh "./ej1_2" --nsys scan_k $K9)
job_e23=$(sbatch --parsable --output=e1_2_k10.out --error=e1_2_k10.err lanzar.sh "./ej1_2" --nsys scan_k $K10)

#casos ejecucion K = 1,..,10 scan con Thrust
job_e24=$(sbatch --parsable --output=e1_3_k1.out --error=e1_3_k1.err lanzar.sh "./ej1_3" --nsys scan_k $K1)
job_e25=$(sbatch --parsable --output=e1_3_k2.out --error=e1_3_k2.err lanzar.sh "./ej1_3" --nsys scan_k $K2)
job_e26=$(sbatch --parsable --output=e1_3_k3.out --error=e1_3_k3.err lanzar.sh "./ej1_3" --nsys scan_k $K3)
job_e27=$(sbatch --parsable --output=e1_3_k4.out --error=e1_3_k4.err lanzar.sh "./ej1_3" --nsys scan_k $K4)
job_e28=$(sbatch --parsable --output=e1_3_k5.out --error=e1_3_k5.err lanzar.sh "./ej1_3" --nsys scan_k $K5)
job_e29=$(sbatch --parsable --output=e1_3_k6.out --error=e1_3_k6.err lanzar.sh "./ej1_3" --nsys scan_k $K6)
job_e30=$(sbatch --parsable --output=e1_3_k7.out --error=e1_3_k7.err lanzar.sh "./ej1_3" --nsys scan_k $K7)
job_e31=$(sbatch --parsable --output=e1_3_k8.out --error=e1_3_k8.err lanzar.sh "./ej1_3" --nsys scan_k $K8)
job_e32=$(sbatch --parsable --output=e1_3_k9.out --error=e1_3_k9.err lanzar.sh "./ej1_3" --nsys scan_k $K9)
job_e33=$(sbatch --parsable --output=e1_3_k10.out --error=e1_3_k10.err lanzar.sh "./ej1_3" --nsys scan_k $K10)


#Coop Group funcionamiento
job_e34=$(sbatch --parsable --output=e2.out --error=e2.err --open-mode=truncate lanzar.sh "./ej2" 64 4 1)

#Coop Group labeled funcionamiento
job_e35=$(sbatch --parsable --output=e2_1.out --error=e2_1.err --open-mode=truncate lanzar.sh "./ej2_1" 64 4 1)

#Bins funcionamiento
job_e36=$(sbatch --parsable --output=e3.out --error=e3.err --open-mode=truncate lanzar.sh "./ej3" 1000 1)

#Bins Thrust vs bins secuencial
job_e38=$(sbatch --parsable --output=e3sec_c.out --error=e3sec_c.err --open-mode=truncate lanzar.sh "./codigoBins" 1000 1)


printf "Jobs enviados correctamente\n"

for jid in \
	"$job_e1" "$job_e2" "$job_e3" \
	"$job_e4" "$job_e5" "$job_e6" "$job_e7" "$job_e8" "$job_e9" "$job_e10" "$job_e11" "$job_e12" "$job_e13" \
	"$job_e14" "$job_e15" "$job_e16" "$job_e17" "$job_e18" "$job_e19" "$job_e20" "$job_e21" "$job_e22" "$job_e23" \
	"$job_e24" "$job_e25" "$job_e26" "$job_e27" "$job_e28" "$job_e29" "$job_e30" "$job_e31" "$job_e32" "$job_e33" \
	"$job_e34" "$job_e35" "$job_e36" "$job_e37" "$job_e38"
do
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


for i in {1..10}; do
	k_var="K${i}"
	k_val=${!k_var}

	printf "Ejercicio 1: Tiempo ejecucion k = %s\n" "$k_val"

	printf "Escan manual\n"
	echo " Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                    Name      \n"
	grep kernel "e1_k${i}.out"
	printf "\n"

	printf "Escan Cub\n"
	echo " Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                    Name      \n"
	grep kernel "e1_2_k${i}.out"
	printf "\n"

	printf "Escan Thrust\n"
	echo " Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                    Name      \n"
	grep kernel "e1_3_k${i}.out"
	printf "\n"
done

printf "Ejercicio 2: Cooperative Groups, grupos de 8 por grupo, N = 64\n\n"
cat e2.out
printf "\n"

printf "Ejercicio 2: Cooperative Groups Labeled, cambio Label cada 4 elementos, N =64\n\n"
cat e2_1.out
printf "\n"

printf "Ejercicio 3: Funcionamiento Bins, N = 64\n\n"
cat e3.out
printf "\n"

printf "Ejercicio 3: Bins Thrust vs Bins Secuencial, N = 1000\n\n"
echo " Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                    Name      \n"
grep kernel "e3_c.out"
printf "\n"

printf "Bins secuenciales CodigoBins, N = 1000\n"
echo " Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                    Name      \n"
grep kernel "e3sec_c.out"
printf "\n"
printf "\n"


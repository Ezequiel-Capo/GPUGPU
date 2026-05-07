#!/bin/bash

mostrar_cat() {
    file=$1
    if [ -s "$file" ]; then
        cat "$file"
    else
        echo "[pendiente] $file no disponible o vacío"
    fi
}

mostrar_kernel() {
    file=$1
    if [ -s "$file" ]; then
        echo " Time(%) | Total(ns) | Instances | Avg(ns) | Med(ns) | Min(ns) | Max(ns) | StdDev(ns) | Kernel"
        grep kernel "$file"
    else
        echo "[pendiente] $file no disponible o vacío"
    fi
}

printf "Resultados de las pruebas\n"

printf "Ejercicio 1-1: shared memory 10x10 (trasposicion)\n\n"
mostrar_cat e1.out
printf "\n"

printf "Ejercicio 1-2: shared memory 2^14x2^14 (sin padding)\n\n"
mostrar_kernel e1_2.out
printf "\n"

printf "Ejercicio 1-3: shared memory 2^14x2^14 con padding\n\n"
mostrar_kernel e1_3.out
printf "\n"

printf "Ejercicio 2 parte 1-1: shared memory arreglo de 64 elementos, alternativa lineal\n\n"
mostrar_cat e2_1.out
printf "\n"

printf "Ejercicio 2 parte 1-2: shared memory arreglo de 2^14 elementos, alternativa lineal\n\n"
mostrar_kernel e2_2.out
printf "\n"

printf "Ejercicio 2 parte 2-1:  shared memory arreglo arreglo de 64 elementos\n\n"
mostrar_cat e3_1.out
printf "\n"

printf "Ejercicio 2 parte 2-2:  shared memory arreglo arreglo de 2^14\n\n"
mostrar_kernel e3_2.out
printf "\n"


printf "Ejercicio 2 parte 2-1: warp shuffle arreglo de 64 elementos\n\n"
mostrar_cat e4_1.out
printf "\n"

printf "Ejercicio 2 parte 2-2: warp shuffle arreglo de 2^14\n\n"
mostrar_kernel e4_2.out
printf "\n"
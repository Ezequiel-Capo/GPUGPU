#!/bin/bash

REPORTES_DIR="reportes"

PROGRAMAS=(
    "lab_manual_packed"
    "lab_wmma_u4"
    "lab_wmma_u8"
    "cublas"
)

CASOS=(
    "1024 524288"  # 2^10x2^19= 512MiB
    "1024 1048576" # 2^10x2^20= 1GiB
    "2048 1048576" # 2^11x2^20= 2GiB
    "4096 1048576" # 2^12x2^20= 4GiB
    "8192 1048576" # 2^13x2^20= 8GiB
)

for prog in "${PROGRAMAS[@]}"; do

    echo "============================================================"
    echo "$prog"
    echo "============================================================"

    # Casos con NSYS
    for caso in "${CASOS[@]}"; do

        read filas columnas <<< "$caso"
        archivo="${REPORTES_DIR}/${prog}_${filas}x${columnas}_profile.out"

        echo
        echo "---------------- $(basename "$archivo") ----------------"

        if [[ ! -f "$archivo" ]]; then
            echo "ERROR: No existe."
        elif [[ ! -s "$archivo" ]]; then
            echo "ERROR: El archivo está vacío."
        else
            awk '
            /Time \(%\).*Range/ {
                print
                getline
                print
                header=1
                next
            }

            header && /PushPop/ {
                if ($NF ~ /:(XXT|Norms|CalculateDistance|GenX)$/)
                    print
            }
            ' "$archivo"
        fi
    done

    # Caso pequeño (sin NSYS)
    archivo="${REPORTES_DIR}/${prog}_32x64.out"

    echo
    echo "---------------- $(basename "$archivo") ----------------"

    if [[ ! -f "$archivo" ]]; then
        echo "ERROR: No existe."
    elif [[ ! -s "$archivo" ]]; then
        echo "ERROR: El archivo está vacío."
    else
        cat "$archivo"
    fi

    echo
done
#!/bin/bash

PROGRAMAS=(
    "./lab_manual_packed"
    "./lab_wmma_u4"
    "./lab_wmma_u8"
    #"./error_numerico"
)

# Crear directorios si no existen
mkdir -p reportes errores

# Casos grandes
CASOS=(
    "1024 262144"  # 2^10x2^18= 256MiB
    "1024 524288"  # 2^10x2^19= 512MiB
    "1024 1048576" # 2^10x2^20= 1GiB
    "4096 1048576" # 2^12x2^20= 4GiB
    "8192 1048576" # 2^13x2^20= 8GiB
)

for prog in "${PROGRAMAS[@]}"; do

    nombre=$(basename "$prog")

    # Casos grandes
    for caso in "${CASOS[@]}"; do

        read filas columnas <<< "$caso"
        reporte="reportes/${nombre}_${filas}x${columnas}"

        if [[ "$prog" == "./error_numerico" ]]; then
            sbatch \
                --output="${reporte}.out" \
                --error="errores/${nombre}_${filas}x${columnas}.err" \
                lanzar.sh \
                "$prog" \
                "$filas" \
                "$columnas"
        else
            sbatch \
                --output="${reporte}_profile.out" \
                --error="errores/${nombre}_${filas}x${columnas}_profile.err" \
                lanzar.sh \
                "$prog" \
                --nsys \
                "$reporte" \
                "$filas" \
                "$columnas"
        fi
    done

    # Caso pequeño (sin NSYS para todos los programas)
    sbatch \
        --output="reportes/${nombre}_32x64.out" \
        --error="errores/${nombre}_32x64.err" \
        lanzar.sh \
        "$prog" \
        32 \
        64

done
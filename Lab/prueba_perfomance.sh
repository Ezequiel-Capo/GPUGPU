#!/bin/bash

PROGRAMAS=(
    "./lab_manual_packed"
    "./lab_wmma_u4"
    "./lab_wmma_u8"
    "./cublas"
)

# Crear directorios si no existen
mkdir -p reportes errores

# Límite de 11 GiB en bytes
LIMIT_BYTES=$((11 * 1024 * 1024 * 1024))

# Casos grandes
CASOS=(
    #para uint8, para float32 es x4
    "1024 524288"  # 2^10x2^19= 512MiB, 2GiB
    "1024 1048576" # 2^10x2^20= 1GiB, 4GiB
    "2048 1048576" # 2^10x2^20= 2GiB, 8GiB
    "4096 1048576" # 2^12x2^20= 4GiB, excede los 11GiB de rtx2080ti
    "8192 1048576" # 2^13x2^20= 8GiB
)

for prog in "${PROGRAMAS[@]}"; do

    nombre=$(basename "$prog")

    # Casos grandes
    for caso in "${CASOS[@]}"; do

        read filas columnas <<< "$caso"
        
        # Calcular memoria requerida en bytes (float = 4 bytes)
        mem_bytes=$((filas * columnas * 4))

        # Restricción: cublas solo se ejecuta si tiene mem disponible
        if [[ "$nombre" == "cublas" ]] && [ "$mem_bytes" -gt "$LIMIT_BYTES" ]; then
            echo "Saltando $nombre para ${filas}x${columnas}: La matriz X de entrada en float32 excede el límite de 11GiB de VRAM disponible."
            continue
        fi

        reporte="reportes/${nombre}_${filas}x${columnas}"

        sbatch \
            --output="${reporte}_profile.out" \
            --error="errores/${nombre}_${filas}x${columnas}_profile.err" \
            lanzar.sh \
            "$prog" \
            --nsys \
            "$reporte" \
            "$filas" \
            "$columnas"
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
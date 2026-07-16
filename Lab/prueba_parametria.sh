#!/bin/bash

# Crear directorios si no existen
mkdir -p reportes errores

PROG_U8="./lab_wmma_u8"
PROG_U4="./lab_wmma_u4"
M="4096"
N="1048576"

# Lista de configuraciones: warp_m warp_n wmma_m wmma_n wmma_k
CONFIGS_U8=(
    "2 2 16 16 16"
    "2 2 32 8 16"
    "2 2 8 32 16"
    "4 4 16 16 16"
    "4 4 32 8 16"
    "4 4 8 32 16"
)

CONFIGS_U4=(
    "2 2"
    "4 4"
)

run_configs () {
    local prog="$1"
    shift
    local -n configs_ref="$1"
    shift

    echo "Lanzando pruebas de parametría para $prog ($M x $N)..."

    for config in "${configs_ref[@]}"; do
        if [[ "$prog" == "$PROG_U4" ]]; then
            read wtm wtn <<< "$config"
        else
            read wtm wtn wm wn wk <<< "$config"
        fi

        if [[ "$prog" == "$PROG_U4" ]]; then
            nombre_base="$(basename "$prog")_${M}x${N}_Warp${wtm}x${wtn}"
        else
            nombre_base="$(basename "$prog")_${M}x${N}_Warp${wtm}x${wtn}_WMMA${wm}x${wn}x${wk}"
        fi
        reporte="reportes/${nombre_base}"

        if [[ "$prog" == "$PROG_U4" ]]; then
            echo " -> Configuración: Warp ${wtm}x${wtn} | WMMA 8x8x32"
        else
            echo " -> Configuración: Warp ${wtm}x${wtn} | WMMA ${wm}x${wn}x${wk}"
        fi

        sbatch \
            --output="${reporte}_profile.out" \
            --error="errores/${nombre_base}_profile.err" \
            lanzar.sh \
            "$prog" \
            --nsys \
            "$reporte" \
            "$M" \
            "$N" \
            "$wtm" "$wtn"

        if [[ "$prog" != "$PROG_U4" ]]; then
            sbatch \
                --output="${reporte}_profile.out" \
                --error="errores/${nombre_base}_profile.err" \
                lanzar.sh \
                "$prog" \
                --nsys \
                "$reporte" \
                "$M" \
                "$N" \
                "$wtm" "$wtn" "$wm" "$wn" "$wk"
        fi
    done
}

run_configs "$PROG_U8" CONFIGS_U8
run_configs "$PROG_U4" CONFIGS_U4
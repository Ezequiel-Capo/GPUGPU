#!/bin/bash

REPORTES_DIR="reportes"
PROGS=("lab_wmma_u8" "lab_wmma_u4")
M="4096"
N="1048576"

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

print_reports () {
    local prog="$1"
    shift
    local -n configs_ref="$1"
    shift

    echo "============================================================"
    echo "Análisis de Parametría: $prog ($M x $N)"
    echo "============================================================"

    for config in "${configs_ref[@]}"; do
        if [[ "$prog" == "lab_wmma_u4" ]]; then
            read wtm wtn <<< "$config"
            nombre_base="${prog}_${M}x${N}_Warp${wtm}x${wtn}"
        else
            read wtm wtn wm wn wk <<< "$config"
            nombre_base="${prog}_${M}x${N}_Warp${wtm}x${wtn}_WMMA${wm}x${wn}x${wk}"
        fi
        archivo="${REPORTES_DIR}/${nombre_base}_profile.out"

        echo
        if [[ "$prog" == "lab_wmma_u4" ]]; then
            echo "---------------- Warp ${wtm}x${wtn} | WMMA 8x8x32 ----------------"
        else
            echo "---------------- Warp ${wtm}x${wtn} | WMMA ${wm}x${wn}x${wk} ----------------"
        fi

        if [[ ! -f "$archivo" ]]; then
            echo "ERROR: El archivo no existe ($archivo)."
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
    echo
}

print_reports "lab_wmma_u8" CONFIGS_U8
print_reports "lab_wmma_u4" CONFIGS_U4
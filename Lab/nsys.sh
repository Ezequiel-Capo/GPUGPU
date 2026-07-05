#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# profile_nsys.sh — Perfila labPensado_cublas con NVIDIA Nsight Systems
# (nsys) para varios tamaños (N,M), y arma:
#   1) Un .nsys-rep por corrida (para abrir en la GUI de Nsight Systems
#      y ver el timeline visual: kernels, copias de memoria, llamadas API).
#   2) Un CSV combinado con el desglose de tiempo POR KERNEL (norm/xxt/union
#      vs cublasSsyrk), que es mas fino que medir con cudaEvent porque nsys
#      te separa cada kernel individualmente y te dice cuanto pesa cada uno
#      sobre el total.
#
# Uso:
#   ./profile_nsys.sh
#   ./profile_nsys.sh labPensado_cublas.cu
#
# Requiere: nsys (Nsight Systems) instalado. Si "nsys --version" no anda:
#   sudo apt install nsight-systems-2023.1.1   (o el paquete que corresponda
#   a tu CUDA toolkit; en Ubuntu/WSL suele venir con nvidia-cuda-toolkit,
#   fijate con: dpkg -l | grep nsight)
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

# En WSL hace falta priorizar el stub de CUDA de WSL por sobre cualquier
# libcuda.so instalada por apt (ver conversacion previa sobre este tema).
#export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

FUENTE="${1:-labPensado_cublas.cu}"
BINARIO="./labPensado_cublas_bench"
DIR_REPORTES="nsys_reportes"
CSV_RESUMEN="nsys_resumen_kernels.csv"

# Tamaños a perfilar: pares (N, M). Con nsys cada corrida tarda un poco
# mas (overhead de tracing), asi que conviene una lista mas corta que la
# del benchmark.sh simple con cudaEvent.

#export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

TAMANIOS=(
    "512 128"
    "1024 256"
    "2048 512"
    "2048 2048"
    "4096 2048"
    "4096 4096"
)
DATETIME=$(date +%d_%H:%M:%S)
echo "[INFO] Moviendo cosas viejas"
mkdir nsys_reportes/viejos_${DATETIME}
mv -f nsys_reportes/*.* nsys_reportes/viejos_${DATETIME}/ || true

if ! command -v nsys &> /dev/null; then
    echo "[ERROR] nsys (NVIDIA Nsight Systems) no esta en el PATH."
    echo "        Instalalo (ej: sudo apt install nsight-systems) o agregalo al PATH,"
    echo "        normalmente en algo como /usr/local/cuda/bin/nsys"
    exit 1
fi

echo "[INFO] Version de nsys detectada:"
nsys --version

mkdir -p "${DIR_REPORTES}"

echo "[INFO] Compilando ${FUENTE}..."
nvcc -O3 -lineinfo "${FUENTE}" -o "${BINARIO}" -lcublas

rm -f "${CSV_RESUMEN}"
ENCABEZADO_ESCRITO=0

for par in "${TAMANIOS[@]}"; do
    read -r N M <<< "${par}"
    NOMBRE="N${N}_M${M}"
    REPORTE="${DIR_REPORTES}/reporte_${NOMBRE}"

    echo ""
    echo "=== Perfilando N=${N} M=${M} con nsys ==="

    # -t cuda,osrt : traza llamadas CUDA (kernels, memcpy) y del sistema operativo
    # --cuda-memory-usage : suma tambien el tracking de cudaMalloc/cudaFree
    nsys profile \
        -o "${REPORTE}" \
        --force-overwrite=true \
        -t cuda,osrt \
        --cuda-memory-usage=true \
        "${BINARIO}" "${N}" "${M}"

    # Resumen de kernels GPU (nombre, tiempo total, instancias, promedio, %)
    # directo a stdout -> archivo, que es lo mas estable entre versiones de nsys.
    STATS_CSV="${DIR_REPORTES}/kernels_${NOMBRE}.csv"

    if nsys stats --report cuda_gpu_kern_sum --format csv "${REPORTE}.nsys-rep" > "${STATS_CSV}" 2>/dev/null; then
        : # ok
    else
        echo "[WARN] 'nsys stats --report cuda_gpu_kern_sum' fallo para ${NOMBRE}."
        echo "       Corre 'nsys stats --help-reports' para ver los nombres de reporte"
        echo "       disponibles en tu version de nsys y ajusta el script si difieren."
        continue
    fi

    # Agregar columnas N,M a cada fila y sumar al CSV combinado.
    # (Los nombres de columna de nsys stats pueden variar segun version;
    #  si el awk siguiente no encuentra encabezado, revisa STATS_CSV a mano.)
    if [ "${ENCABEZADO_ESCRITO}" -eq 0 ]; then
        head -n 1 "${STATS_CSV}" | awk '{print $0",N,M"}' > "${CSV_RESUMEN}"
        ENCABEZADO_ESCRITO=1
    fi
    tail -n +2 "${STATS_CSV}" | awk -v n="${N}" -v m="${M}" '{print $0","n","m}' >> "${CSV_RESUMEN}"

    echo "[INFO] Reporte: ${REPORTE}.nsys-rep  |  Resumen: ${STATS_CSV}"
done

echo ""
echo "[INFO] Listo."
echo "[INFO] Reportes .nsys-rep en '${DIR_REPORTES}/' -> abrilos con la GUI de"
echo "       Nsight Systems (nsys-ui) para ver el timeline visual completo."
echo "[INFO] Resumen combinado de todos los tamaños en: ${CSV_RESUMEN}"
echo ""
if command -v column &> /dev/null; then
    column -s, -t "${CSV_RESUMEN}" 2>/dev/null || cat "${CSV_RESUMEN}"
else
    cat "${CSV_RESUMEN}"
fi
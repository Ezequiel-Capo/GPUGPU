#!/bin/bash
#SBATCH --job-name=lab
#SBATCH --ntasks=1
# AUMENTAMOS LA MEMORIA RAM Y EL TIEMPO PARA EVITAR CORTES EN MATRICES GRANDES
#SBATCH --mem=32G
#SBATCH --time=00:15:00

#SBATCH --gres=gpu:n2080ti:1
# para ejecutar en la gtx1060 ---> #SBATCH --gres=gpu:n1060:1
# para ejecutar en la rtx2080ti ---> #SBATCH --gres=gpu:n2080ti:1

#SBATCH --partition=cursos
#SBATCH --qos=gpgpu

PATH=$PATH:/usr/local/cuda/bin
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
#nvcc --version
set -e

exe="$1"
shift
run_exe="${exe}_${SLURM_JOB_ID:-$$}"
trap 'rm -f "$run_exe"' EXIT

case "$exe" in
    "./lab_manual_packed")
        nvcc -O3 -arch=sm_75 -o "$run_exe" lab_manual_packed.cu
        ;;
    "./lab_wmma_u4")
        nvcc -O3 -arch=sm_75 -o "$run_exe" lab_wmma_u4.cu
        ;;
    "./lab_wmma_u8")
        nvcc -O3 -arch=sm_75 -o "$run_exe" lab_wmma_u8.cu
        ;;
    "./error_numerico")
        nvcc -O3 -arch=sm_75 -o "$run_exe" error_numerico.cu
        ;;
    "./cublas")
        nvcc -O3 -arch=sm_75 -o "$run_exe" cublas.cu -lcublas -lnvToolsExt
        ;;    
    *)
        echo "Error: ejecutable no soportado: $exe"
        exit 1
        ;;
esac

if [ "$1" = "--nsys" ]; then
    shift
    report_name="$1"
    shift
    nsys profile --stats=true --force-overwrite=true -o "$report_name" "$run_exe" "$@"
elif [ "$1" = "--sanitizer" ]; then
    shift
    echo "Ejecutando con compute-sanitizer..."
    compute-sanitizer "$run_exe" "$@"
else
    "$run_exe" "$@"
fi
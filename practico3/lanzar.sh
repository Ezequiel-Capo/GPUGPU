#!/bin/bash
#SBATCH --job-name=practico2_ej1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:05:00

#SBATCH --gres=gpu:n2080ti:1
# para ejecutar en la gtx1060 ---> #SBATCH --gres=gpu:n1060:1
# para ejecutar en la rtx2080ti ---> #SBATCH --gres=gpu:n2080ti:1

#SBATCH --partition=cursos
#SBATCH --qos=gpgpu

PATH=$PATH:/usr/local/cuda/bin
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

set -e

exe="$1"
shift

case "$exe" in
	"./ej1")
		nvcc -O2 -o ej1 ej1.cu
		;;
	"./ej2")
		nvcc -O2 -o ej2 ej2.cu
		;;
	"./ej2_2")
		nvcc -O2 -o ej2_2 ej2_2.cu
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
    nsys profile --stats=true -o "$report_name" "$exe" "$@"
else
    "$exe" "$@"
fi
#sbatch lanzar.sh "./ej1" "secreto.txt"

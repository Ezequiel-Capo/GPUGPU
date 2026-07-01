#!/bin/bash
#SBATCH --job-name=lab
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
#nvcc --version
set -e

exe="$1"
shift
run_exe="${exe}_${SLURM_JOB_ID:-$$}"
trap 'rm -f "$run_exe"' EXIT

case "$exe" in
	"./lab")
		nvcc -O2 -arch=sm_75 -o "$run_exe" lab.cu
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
else
    "$run_exe" "$@"
fi
#sbatch lanzar.sh "./lab" 
#sbatch --parsable --output=labWMMA.out --error=labWMMA.err lanzar.sh "./lab" --nsys tensorShared 1024 1048576
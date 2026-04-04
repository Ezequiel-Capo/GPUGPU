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

$1 $2 $3 $4 $5 $6

# nvcc -O2 -o ej1 ej1.cu
#sbatch practico2/lanzar.sh "./ej1" "secreto.txt"

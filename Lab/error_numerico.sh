#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v sbatch >/dev/null 2>&1; then
	echo "Error: sbatch no esta disponible en este entorno."
	exit 1
fi

SIZES_M=(512 1024 2048)      # 2^9, 2^10, 2^11
SIZES_EXP=(9 10 11)
N=1048576                    # 2^20

echo "Lanzando pruebas de error numerico contra cuBLAS"
echo "Tamano de SNPs fijo: n=${N} (2^20)"
echo

for i in "${!SIZES_M[@]}"; do
	M="${SIZES_M[$i]}"
	M_EXP="${SIZES_EXP[$i]}"
	JOB_NAME="error_numerico_m${M}_n${N}"
	LOG_FILE="e_numerico/${JOB_NAME}.out"

	mkdir -p e_numerico

	echo "============================================================"
	echo "Caso: m=${M} (2^${M_EXP}), n=${N} (2^20)"
	echo "Enviando job: ${JOB_NAME}"

	sbatch  --output="$LOG_FILE" --job-name="$JOB_NAME" \
		lanzar.sh ./error_numerico "$M" "$N"

	echo "Lanzado ||D - Dcublas|| para m=${M}, n=${N}:"

done

echo "Al finalizar. Logs disponibles en e_numerico/*.out"

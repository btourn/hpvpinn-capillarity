#!/bin/bash
#SBATCH --job-name=hpoGS
#SBATCH --partition=principal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Requires MODEL, NOISE_LEVEL, NOISE_LABEL set at submission time. Unlike a
# #SBATCH comment line, $MODEL below IS expanded here -- it's evaluated by
# your interactive shell before sbatch ever sees it, not parsed from the
# script file. Create the model's log dir first (--output/--error must
# already exist when the job starts), then submit, e.g.:
#
#   mkdir -p hp_vpinn/logs
#   sbatch --export=MODEL=hp_vpinn,NOISE_LEVEL=0.02,NOISE_LABEL=2pct \
#       --output=hp_vpinn/logs/%x_%j.out --error=hp_vpinn/logs/%x_%j.err \
#       run_hparam_grenoblesand.sh
#
#   mkdir -p frontx/logs
#   sbatch --export=MODEL=frontx,NOISE_LEVEL=0.02,NOISE_LABEL=2pct \
#       --output=frontx/logs/%x_%j.out --error=frontx/logs/%x_%j.err \
#       run_hparam_grenoblesand.sh

echo "=============================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Nodo       : $(hostname)"
echo "Fecha      : $(date)"
echo "Model      : $MODEL"
echo "Noise level: $NOISE_LEVEL"
echo "=============================="

cd "${HPVPINN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # clone of the repository (submit from its root)

source .venv/bin/activate

python -m hpvpinn_capillarity.hpo.search_grenoblesand \
    --model $MODEL \
    --n-trials 150 \
    --noise-level $NOISE_LEVEL \
    --warmup-epochs 0 \
    --transition-epochs 20000

echo "Finalizado: $(date)"

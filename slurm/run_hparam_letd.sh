#!/bin/bash
#SBATCH --job-name=hpoLETd
#SBATCH --partition=principal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Requires MODEL set at submission time. Unlike a #SBATCH comment line,
# $MODEL below IS expanded here -- it's evaluated by your interactive shell
# before sbatch ever sees it, not parsed from the script file. Create the
# model's log dir first (--output/--error must already exist when the job
# starts), then submit, e.g.:
#
#   mkdir -p hp_vpinn/logs
#   sbatch --export=MODEL=hp_vpinn \
#       --output=hp_vpinn/logs/%x_%j.out --error=hp_vpinn/logs/%x_%j.err \
#       run_hparam_letd.sh
#
#   mkdir -p frontx/logs
#   sbatch --export=MODEL=frontx \
#       --output=frontx/logs/%x_%j.out --error=frontx/logs/%x_%j.err \
#       run_hparam_letd.sh

echo "=============================="
echo "Job ID : $SLURM_JOB_ID"
echo "Nodo   : $(hostname)"
echo "Fecha  : $(date)"
echo "Model  : $MODEL"
echo "=============================="

cd ~/manuscript_review_runs   # repo root: contains hp_vpinn/, frontx/, and hparam_search_letd.py

source "${MODEL}/.venv/bin/activate"

python hparam_search_letd.py \
    --model $MODEL \
    --n-trials 150 \
    --warmup-epochs 200000 \
    --transition-epochs 0 \
    --results-dir results_letd_exp

echo "Finalizado: $(date)"

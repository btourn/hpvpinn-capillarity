#!/bin/bash
# submit_all_brooks_corey_new.sh
# Submits parallel Optuna workers for each (model, noise_level, schedule)
# combination.
#
# Full matrix (when --model is omitted):
#   model      0.01 × warmup=0,      transition=20000
#   hp_vpinn,  0.01 × warmup=200000, transition=0
#   frontx
#   (1% noise only for this case)
#
# The unified Python script (--model flag) handles all internal directory
# and study naming; each model's results land under its own subdirectory
# (hp_vpinn/... or frontx/...), same base results-dir name for both.
#
# Usage:
#   bash slurm/submit_all_brooks_corey_new.sh                       # both models, full matrix
#   bash slurm/submit_all_brooks_corey_new.sh --model hp_vpinn       # one model only, full matrix
#   bash slurm/submit_all_brooks_corey_new.sh --model frontx --dry-run
#   bash slurm/submit_all_brooks_corey_new.sh --dry-run --model hp_vpinn   # order doesn't matter

cd "$(dirname "${BASH_SOURCE[0]}")/.."   # repository root

DRY_RUN=false
MODEL_FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --model)
            MODEL_FILTER="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--model {hp_vpinn}] [--dry-run]" >&2
            exit 1
            ;;
    esac
done

if [[ -n "$MODEL_FILTER" ]]; then
    if [[ "$MODEL_FILTER" != "hp_vpinn" ]]; then
        echo "Error: --model must be hp_vpinn, got: $MODEL_FILTER" >&2
        exit 1
    fi
    MODELS=("$MODEL_FILTER")
else
    MODELS=("hp_vpinn")
fi

N_WORKERS=5

# Total trial BUDGET differs by model: hp_vpinn's weak-form search space is
# 5-dimensional (nn_n_layers, nn_width, activation, n_elements, poly_degree)
# and gets the full 250-trial budget; frontx's strong-form search space is
# only 3-dimensional (no n_elements/poly_degree -- no quadrature to tune),
# so 120 trials is enough. N_WORKERS stays the same for both so the two
# models keep the same degree of SLURM parallelism; only the per-worker
# trial count changes.
declare -A TRIALS_PER_WORKER=( [hp_vpinn]=50 )   # 5 workers * 50 = 250 trials

# ---------------------------------------------------------------------------
# Noise/schedule combination matrix, shared by both models
# Format: "NOISE_LEVEL:WARMUP_EPOCHS:TRANSITION_EPOCHS"
# ---------------------------------------------------------------------------
declare -a CONFIGS=(
    "0.01:0:20000"
    "0.01:200000:0"
)

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
echo "=== Brooks-Corey (new, 2 unknowns) hyperparameter search ==="
echo "    Models:             ${MODELS[*]}"
echo "    Combinations:       ${#CONFIGS[@]}"
echo "    Workers per combo:  ${N_WORKERS}"
for MODEL in "${MODELS[@]}"; do
    echo "    ${MODEL}: ${TRIALS_PER_WORKER[$MODEL]} trials/worker -> $(( N_WORKERS * TRIALS_PER_WORKER[$MODEL] )) trials/combo"
done
TOTAL_JOBS=$(( ${#MODELS[@]} * ${#CONFIGS[@]} * N_WORKERS ))
echo "    Total jobs:         ${TOTAL_JOBS}"
echo ""

for MODEL in "${MODELS[@]}"; do
    LOG_DIR="results/${MODEL}/logs"   # --model's value IS the subdirectory name now
    mkdir -p "$LOG_DIR"   # must exist before sbatch, or --output/--error fail at submission
    TPW="${TRIALS_PER_WORKER[$MODEL]}"

    for config in "${CONFIGS[@]}"; do
        IFS=':' read -r NOISE WARMUP TRANSITION <<< "$config"

        NOISE_TAG="n${NOISE//./}"
        SCHED_TAG="w${WARMUP}_t${TRANSITION}"

        echo "--- model=${MODEL} | noise=${NOISE} | warmup=${WARMUP} | transition=${TRANSITION} | trials/worker=${TPW} ---"

        for i in $(seq 1 $N_WORKERS); do
            CMD="sbatch \
                --job-name=hpo_${MODEL}_${NOISE_TAG}_${SCHED_TAG}_w${i} \
                --output=${LOG_DIR}/%x_%j.out \
                --error=${LOG_DIR}/%x_%j.err \
                --export=MODEL=${MODEL},NOISE_LEVEL=${NOISE},WARMUP_EPOCHS=${WARMUP},TRANSITION_EPOCHS=${TRANSITION},N_TRIALS=${TPW} \
                slurm/slurm_run_hparam_brooks_corey_new.sh"

            if $DRY_RUN; then
                echo "    [DRY RUN] $CMD"
            else
                OUTPUT=$($CMD)
                echo "    Worker ${i}: ${OUTPUT}"
            fi
        done
        echo ""
    done
done

echo "Done. Monitor with: squeue -u $USER -l"

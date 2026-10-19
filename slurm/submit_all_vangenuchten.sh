#!/bin/bash
# submit_all_vangenuchten.sh
# Submits parallel Optuna workers for each (model, schedule) combination
# of the van Genuchten case.
#
# Full matrix (when --model is omitted):
#   model              warmup=0,      transition=20000
#   hp_vpinn, frontx   warmup=200000, transition=0
#
# The unified Python script (--model flag) handles all internal directory
# and study naming; each model's results land under its own subdirectory
# (hp_vpinn/... or frontx/...).
#
# Usage:
#   bash slurm/submit_all_vangenuchten.sh
#   bash slurm/submit_all_vangenuchten.sh --model hp_vpinn
#   bash slurm/submit_all_vangenuchten.sh --model frontx --dry-run
#   bash slurm/submit_all_vangenuchten.sh --dependency afterany:111:222:333

cd "$(dirname "${BASH_SOURCE[0]}")/.."   # repository root

DRY_RUN=false
MODEL_FILTER=""
DEPENDENCY=""

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
        --dependency=*)
            DEPENDENCY="${1#*=}"
            shift
            ;;
        --dependency)
            DEPENDENCY="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--model {hp_vpinn}] [--dry-run] [--dependency SPEC]" >&2
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

DEP_FLAG=""
if [[ -n "$DEPENDENCY" ]]; then
    DEP_FLAG="--dependency=${DEPENDENCY}"
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
# Schedule variants, shared by both models
# Format: "WARMUP_EPOCHS:TRANSITION_EPOCHS"
# ---------------------------------------------------------------------------
declare -a CONFIGS=(
    "0:20000"
    "200000:0"
)

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
echo "=== Van Genuchten hyperparameter search ==="
echo "    Models:              ${MODELS[*]}"
echo "    Variants:            ${#CONFIGS[@]}"
echo "    Workers per variant: ${N_WORKERS}"
for MODEL in "${MODELS[@]}"; do
    echo "    ${MODEL}: ${TRIALS_PER_WORKER[$MODEL]} trials/worker -> $(( N_WORKERS * TRIALS_PER_WORKER[$MODEL] )) trials/variant"
done
TOTAL_JOBS=$(( ${#MODELS[@]} * ${#CONFIGS[@]} * N_WORKERS ))
echo "    Total jobs:          ${TOTAL_JOBS}"
if [[ -n "$DEPENDENCY" ]]; then
    echo "    Dependency:          ${DEPENDENCY}"
fi
echo ""

for MODEL in "${MODELS[@]}"; do
    LOG_DIR="results/${MODEL}/logs"
    mkdir -p "$LOG_DIR"   # must exist before sbatch, or --output/--error fail at submission
    TPW="${TRIALS_PER_WORKER[$MODEL]}"

    for config in "${CONFIGS[@]}"; do
        IFS=':' read -r WARMUP TRANSITION <<< "$config"
        SCHED_TAG="w${WARMUP}_t${TRANSITION}"

        echo "--- model=${MODEL} | warmup=${WARMUP} | transition=${TRANSITION} | trials/worker=${TPW} ---"

        for i in $(seq 1 $N_WORKERS); do
            CMD="sbatch \
                --job-name=hpo_${MODEL}_vg_${SCHED_TAG}_w${i} \
                --output=${LOG_DIR}/%x_%j.out \
                --error=${LOG_DIR}/%x_%j.err \
                $DEP_FLAG \
                --export=MODEL=${MODEL},WARMUP_EPOCHS=${WARMUP},TRANSITION_EPOCHS=${TRANSITION},N_TRIALS=${TPW} \
                slurm/slurm_run_hparam_vangenuchten.sh"

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

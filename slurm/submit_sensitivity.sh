#!/bin/bash
# submit_sensitivity.sh
# Noise-realisation (R2-m08, R2-M03) and data-density (R1-19, R2-m08) study
# on the frozen Case 1 hp-VPINN winner. Default: gb schedule, 1% noise,
# 25 realisations x 4 densities = 100 runs, striped over 20 array tasks
# (5 runs of ~40 s each per task, plus JIT compile).
#
# Prerequisite: the Stage-A search for (SCHED, NOISE) has finished -- the
# frozen configuration is read from its hparam_results_*.jsonl.
#
# Usage:
#   bash submit_sensitivity.sh                       # gb, 1%, stride
#   bash submit_sensitivity.sh --subsample random    # nested random subsets
#   bash submit_sensitivity.sh --noise 0.02          # the 2% case
#   bash submit_sensitivity.sh --dry-run

SCHED="gb"
NOISE="0.01"
NREAL=25
FRACS="1.0 0.5 0.25 0.125"
SUBSAMPLE="stride"
N_TASKS=20
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sched)     SCHED="$2"; shift 2 ;;
        --noise)     NOISE="$2"; shift 2 ;;
        --nreal)     NREAL="$2"; shift 2 ;;
        --fracs)     FRACS="$2"; shift 2 ;;
        --subsample) SUBSAMPLE="$2"; shift 2 ;;
        --n-tasks)   N_TASKS="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=true; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."   # repository root
LOG_DIR="results/hp_vpinn/logs"
mkdir -p "$LOG_DIR"

source .venv/bin/activate
python -m hpvpinn_capillarity.analysis.sensitivity_noise_density --schedule "$SCHED" --noise-level "$NOISE" \
    --n-realisations "$NREAL" --fractions $FRACS --subsample "$SUBSAMPLE" --list
echo "--- Frozen configuration (check it matches the SI identifiability table) ---"
python -m hpvpinn_capillarity.analysis.sensitivity_noise_density --schedule "$SCHED" --noise-level "$NOISE" --show-winner || exit 1
deactivate 2>/dev/null || true

NOISE_TAG=$(python3 -c "print(f'{int(float(\"$NOISE\")*100)}pct')")
JOB="sens_${SCHED}_${NOISE_TAG}_${SUBSAMPLE}"

# FRACS contains spaces, which a comma-separated --export=VAR=... list cannot
# carry; exporting to the environment and using --export=ALL avoids that.
export SCHED NOISE NREAL FRACS SUBSAMPLE N_TASKS

CMD=(sbatch
     --job-name="$JOB"
     --array="0-$((N_TASKS - 1))"
     --output="${LOG_DIR}/%x_%A_%a.out"
     --error="${LOG_DIR}/%x_%A_%a.err"
     --export=ALL
     slurm/slurm_run_sensitivity.sh)

if $DRY_RUN; then
    echo "[DRY RUN] SCHED=$SCHED NOISE=$NOISE NREAL=$NREAL FRACS='$FRACS' SUBSAMPLE=$SUBSAMPLE N_TASKS=$N_TASKS"
    echo "[DRY RUN] ${CMD[*]}"
else
    "${CMD[@]}"
fi

echo "Monitor with: squeue -u $USER -l"

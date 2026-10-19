#!/bin/bash
#SBATCH --job-name=hpoLETd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2000
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --constraint=v3-20cores

# These are set by the submission script via --export:
#   MODEL, WARMUP_EPOCHS, TRANSITION_EPOCHS, N_TRIALS
#
# --output/--error/--job-name are intentionally NOT set here: #SBATCH
# directives are parsed as literal text, so they can't reference $MODEL to
# route each model's logs into its own subdirectory. submit_all_letd.sh
# passes them as sbatch CLI flags instead, which do know $MODEL at
# submission time and override any static directive here.

export OMP_NUM_THREADS=10
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"

cd "${HPVPINN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # clone of the repository (submit from its root)
source .venv/bin/activate

# Stagger workers to avoid SQLite race condition on schema creation.
# Each (model, schedule) combination writes to its own results_dir
# (hp_vpinn/... or frontx/...) and therefore its own study.db, so adding
# MODEL to the matrix does not introduce any new cross-worker contention
# beyond what already existed per schedule combination.
SLEEP_S=$(( RANDOM % 60 ))
echo "Sleeping ${SLEEP_S}s before starting to stagger DB access..."
sleep $SLEEP_S

python -m hpvpinn_capillarity.hpo.search_letd \
    --model              $MODEL \
    --warmup-epochs      $WARMUP_EPOCHS \
    --transition-epochs  $TRANSITION_EPOCHS \
    --n-trials           $N_TRIALS

#!/bin/bash
#SBATCH --job-name=multistart
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2000
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --constraint=v3-20cores
#SBATCH --output=logs/multistart_%A_%a.out
#SBATCH --error=logs/multistart_%A_%a.err

# One array task = one start of multistart/manifest.csv (row id = SLURM_ARRAY_TASK_ID).
#   python -m hpvpinn_capillarity.analysis.multistart manifest
#   mkdir -p logs
#   sbatch --array=0-<N-1>%100 slurm_run_multistart.sh        # N printed by the manifest step
# Re-run only failed tasks with e.g.:  sbatch --array=7,23,41 slurm_run_multistart.sh

export OMP_NUM_THREADS=10
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"

cd "${HPVPINN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # clone of the repository (submit from its root)
source .venv/bin/activate

python -m hpvpinn_capillarity.analysis.multistart run --task-id $SLURM_ARRAY_TASK_ID

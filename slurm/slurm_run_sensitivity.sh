#!/bin/bash
#SBATCH --job-name=sensND
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2000
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --constraint=v3-20cores

# Array job: each task trains the runs with run_index % N_TASKS == SLURM_ARRAY_TASK_ID.
# Set by submit_sensitivity.sh via --export:
#   SCHED, NOISE, NREAL, FRACS, SUBSAMPLE, N_TASKS
# Each run writes its own JSON (atomic rename), so tasks never share a file
# and resubmitting the same array only trains the runs that are missing.

export OMP_NUM_THREADS=10
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"

cd "${HPVPINN_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"   # clone of the repository (submit from its root)
source .venv/bin/activate

python -m hpvpinn_capillarity.analysis.sensitivity_noise_density \
    --schedule       $SCHED \
    --noise-level    $NOISE \
    --n-realisations $NREAL \
    --fractions      $FRACS \
    --subsample      $SUBSAMPLE \
    --task-id        $SLURM_ARRAY_TASK_ID \
    --n-tasks        $N_TASKS

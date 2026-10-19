# Changelog

## Revised manuscript (unreleased)

Changes with respect to `v1.0-submission`. Results obtained with this version
are not comparable number by number with those of the original submission.

### Numerics

- Runs use 64-bit precision (`jax_enable_x64`); the data casts to `float32` and
  the `float32` default dtype of the trainable parameters were removed.
- The boundary terms of the physics loss are the Dirichlet and the derivative
  conditions at the far end, weighted by `lambda_bc` (10 in all searches); the
  original version penalised the derivative only.
- The maximum value of the adaptive physics weight (`lambda_max`) is 1e6
  instead of 1e9.
- The van Genuchten diffusivity is evaluated with a numerically stable form
  (`expm1`/`log1p`) that avoids catastrophic cancellation near saturation, and
  the effective saturation is clipped to `[1e-8, 1 - 1e-8]`.
- New Brooks and Corey diffusivity (`BrooksAndCoreyDiffusivity`).
- `TrainingHistory` is a named tuple with the PDE, Dirichlet and derivative
  components of the physics loss.
- New strong-form residual on a fine grid (`strong_form_residual.py`), used as
  an independent check of the physics enforcement.

### Workflow

- Unified Optuna search drivers for the four cases, with the held-out
  quadrature evaluation of the physics loss and the classification of trials
  (admissible, calibrated, qualified) used to select the winners.
- Multi-seed final runs of the selected configurations
  (`hp_vpinn/final_run_common.py` and one driver per case).
- Early-stopping analysis, multi-start campaign, quadrature check,
  noise-realisation and data-density sensitivity, and conventional-inversion
  benchmark with forward check.
- SLURM launchers of the searches in `slurm/`.
- Case 4 (Brooks and Corey, synthetic).

### Repository

- The layout of the original submission (`src/hpvpinn/`, `experiments/`,
  `postprocessing/`, `figures/`, `data/`, `results/`) is replaced by the
  layout described in `README.md`. It remains available in `v1.0-submission`.
- `PROVENANCE.md` records the source commits and the sha256 of the copied files.

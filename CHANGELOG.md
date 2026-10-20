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
  (`final_runs/common.py` and one driver per case).
- Early-stopping analysis, multi-start campaign, quadrature check,
  noise-realisation and data-density sensitivity, and conventional-inversion
  benchmark with forward check.
- SLURM launchers of the searches in `slurm/`.
- Case 4 (Brooks and Corey, synthetic).

### Repository

- The layout of the original submission (`src/hpvpinn/`, `experiments/`,
  `postprocessing/`, `figures/`, `data/`, `results/`) is replaced. It remains
  available in `v1.0-submission`.
- The code is an installable package, `hpvpinn_capillarity` (`src/` layout), with
  the subpackages `core`, `hpo`, `final_runs`, `analysis` and `conventional`, a
  `paths` module that defines where data and results are read and written, and a
  single environment at the repository root (`uv sync`). Importing the package
  enables 64-bit precision.
- Data files are in `data/`; recorded outputs are in `results/` with the layout
  the scripts write to (`results/hp_vpinn/...`, `results/multistart/`,
  `results/conventional/`). The location of data and results can be set with the
  environment variables `HPVPINN_DATA` and `HPVPINN_RESULTS`.
- The collocation-PINN (`frontx`) backend of the search drivers is not part of
  the package; the search drivers accept only `--model hp_vpinn`. The scripts of
  `conventional/` use the public `frontx` (commit `e04d264`), installed from
  `requirements-conventional.txt`.
- The dependencies that nothing in the package imports are removed (`torch`,
  `torchvision`, `torchaudio`, `flax`, `orbax-checkpoint`, `rich`, `plotly`,
  `optuna-dashboard`); `jaxtyping` remains as a dependency of `equinox`.
- `data/README.md` documents the data files and the origin of the experimental data.
- Figures and tables are written to different directories: `results/hp_vpinn/Figures/`
  holds only figures; LaTeX tables, `winners.*` and the machine-readable summaries are
  in `results/hp_vpinn/tables/` (`--tables-dir`).
- The scripts are run as modules (`python -m hpvpinn_capillarity....`); the SLURM
  launchers in `slurm/` do the same and change to the repository root themselves.
- `PROVENANCE.md` records where every file comes from and how it changed.

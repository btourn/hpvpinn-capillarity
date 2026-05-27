# *hp*-VPINN for inverse hydraulic parameter recovery in unsaturated porous media

This repository contains the implementation of the *hp*-variational
physics-informed neural network (*hp*-VPINN) framework for inverse modeling
of horizontal capillary flow, developed alongside the manuscript:

<!--
> B. A. Tourn, S. Palavecino, P. A. Kler, G. S. Gerlero —
> *Inverse parameter estimation of horizontal capillary flow in porous media with hp-variational physics-informed neural networks* -
> *Transport in Porous Media*, 2025 (under review).
-->

The framework recovers unknown hydraulic parameters from sparse, noisy
moisture-content profiles by embedding the Boltzmann similarity
transformation of the Richards–Richardson equation into a differentiable
neural network architecture trained with JAX.

---

## Repository structure

```
hpvpinn-capillarity/
├── src/hpvpinn/          # installable package
│   ├── vpinn_loss.py     # weak-form physics loss (VPINNLoss)
│   ├── train_vpinn.py    # Adam training loop + L-BFGS refinement
│   ├── nn_model.py       # feed-forward neural network
│   ├── pinn_model.py     # PINN container (network + PDE)
│   ├── pde.py            # capillarity Boltzmann PDE
│   ├── constitutive_model.py  # Van Genuchten / LETd diffusivity
│   ├── datasets.py       # MeasurementSet and data utilities
│   ├── transformations.py     # Boltzmann scaling
│   └── plot_utils.py     # plotting helpers
├── experiments/
│   ├── grenoblesand/     # Case 1 — Grenoble sand (synthetic, Van Genuchten)
│   ├── vangenuchten/     # Case 2 — Whatman paper (experimental, Van Genuchten)
│   └── letd/             # Case 3 — Whatman paper (experimental, LETd)
├── postprocessing/
│   └── hparam_postprocess.py  # Optuna diagnostics and SI figures
├── figures/
│   ├── combine_panels_2x2_history.py   # loss + parameter history figure
│   └── combine_panels_2x2_profile.py  # wetting profile compound figure
├── data/
│   ├── grenoblesand/data_grenoblesand.dat
│   ├── vangenuchten/data_vangenuchten.dat
│   └── letd/data_letd.dat
├── results/              # pre-computed results 
├── pyproject.toml
└── uv.lock
```

---

## Installation

The project uses [`uv`](https://docs.astral.sh/uv/) for environment and
dependency management. Install `uv` if you do not have it:

```bash
pip install uv
```

Then clone the repository and install all dependencies in one step:

```bash
git clone https://github.com/btourn/hpvpinn-capillarity.git
cd hpvpinn-capillarity
uv sync
```

This creates a `.venv` virtual environment, installs all dependencies listed
in `uv.lock`, and installs the `hpvpinn` package in editable mode.
No additional steps are required. All scripts can be run by prefixing
`uv run` as shown in the examples below.

Alternatively, if you prefer to work in an activated shell session:

```bash
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

**VSCode users**: after running `uv sync`, open the Command Palette
(`Ctrl+Shift+P`) → *Python: Select Interpreter* → select
`.venv/bin/python` to enable debugging and IntelliSense.

---

## Pre-computed results

The `results/` directory contains the pre-computed outputs used in the
manuscript, including training histories, wetting profiles, and
hyperparameter search records. Running the driver or search scripts will
regenerate and overwrite these files. To restore the original state,
check out the submission tag:

    git checkout v1.0-submission

## Reproducing the results

### Running the best-trial configuration directly

Each experiment directory contains a driver script that runs the
hyperparameter configuration reported in the paper:

```bash
# Case 1 — Grenoble sand, 1% noise, hpVPINN-gb
uv run python experiments/grenoblesand/grenoblesand_trial226.py

# Case 1 — Grenoble sand, 2% noise, hpVPINN-gb
uv run python experiments/grenoblesand/grenoblesand_trial236.py

# Case 2 — Whatman paper, Van Genuchten, hpVPINN-es
uv run python experiments/vangenuchten/vangenuchten_trial182.py

# Case 3 — Whatman paper, LETd, hpVPINN-gb
uv run python experiments/letd/letd_trial72.py
```

### Running the hyperparameter search

The full Optuna search (250 trials per configuration) can be reproduced with:

```bash
# Case 1, hpVPINN-gb, 1% noise
uv run python experiments/grenoblesand/hparam_search_grenoblesand.py \
    --config gb --noise-level 0.01 --n-trials 250

# Case 3, hpVPINN-gb
uv run python experiments/letd/hparam_search_letd.py \
    --config gb --n-trials 250
```

Results are written to `results/results_<case>_<config>/` as JSONL files.

<!--The full search results reported in the paper are archived on Zenodo at
[`https://doi.org/10.5281/zenodo.TODO`](https://doi.org/10.5281/zenodo.TODO).-->

### Generating the figures

```bash
# Convergence history figure (Fig. 4 in the paper)
uv run python figures/combine_panels_2x2_history.py

# Wetting profile compound figure (Fig. 2)
uv run python figures/combine_panels_2x2_profile.py

# Supplementary Information diagnostics
uv run python postprocessing/hparam_postprocess.py \
    --case grenoblesand --config gb --noise-level 0.01
```

---

## Hardware and reproducibility

Exact bitwise reproducibility across different hardware or JAX/XLA
versions cannot be guaranteed due to floating-point non-determinism in
parallel reductions. Results are reproducible in a statistical sense:
independent runs recover parameters within the error ranges reported in
the paper. See the Supplementary Information for details.

---

<!--
## Citation

If you use this code in your research, please cite:

```bibtex
@article{tourn2025hpvpinn,
  author  = {Tourn, Benjamin and xxxxxxx},
  title   = {xxxx},
  journal = {Transport in Porous Media},
  year    = {2025},
  note    = {Under review}
}
```
-->

---

## Licence

Copyright 2025 Benjamin Tourn.
Licensed under the [Apache License 2.0](LICENSE).
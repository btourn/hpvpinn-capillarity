"""
grenoblesand.py
===============
Final hp-VPINN runs for Case 1 (Grenoble sand, synthetic data, van Genuchten
model with trainable Ks/alpha and m). See final_run_common.py for what a
final run is, the outputs, and the options.

Usage
-----
    # configuration selected by hpvpinn_capillarity.hpo.postprocess --winners, 10 fresh seeds
    python -m hpvpinn_capillarity.final_runs.grenoblesand --noise-level 0.01
    python -m hpvpinn_capillarity.final_runs.grenoblesand --noise-level 0.02 --n-seeds 5

    # the winner of a given schedule instead of the selected one
    python -m hpvpinn_capillarity.final_runs.grenoblesand --noise-level 0.01 --schedule es

    # user-defined configuration
    python -m hpvpinn_capillarity.final_runs.grenoblesand --noise-level 0.01 --schedule gb \\
        --arch "W=16,L=2,act=tanh,Nel=30,p=5"

    # quick smoke test (separate run directory, tag suffix _ep2000)
    python -m hpvpinn_capillarity.final_runs.grenoblesand --noise-level 0.01 --n-seeds 2 --epochs 2000

    # rebuild figures and tables from the saved seeds (no training, no JAX)
    python -m hpvpinn_capillarity.final_runs.grenoblesand --noise-level 0.01 --aggregate-only
    python -m hpvpinn_capillarity.final_runs.grenoblesand --noise-level 0.01 --show-config
"""

from pathlib import Path

import numpy as np

from hpvpinn_capillarity import paths
from hpvpinn_capillarity.final_runs.common import CaseSpec, main, read_constant

SEARCH_MODULE = "hpvpinn_capillarity.hpo.search_grenoblesand"

# Reference (true) parameters, read from the HPO module so they cannot drift.
_SRC = paths.module_source(SEARCH_MODULE)
_KS, _ALPHA, _M = (read_constant(_SRC, n) for n in ("Ks_TRUE", "ALPHA_TRUE", "M_TRUE"))
TRUE_PARAMS = dict(KsAlpha=_KS / _ALPHA, m=_M)


def prepare_shared(m, data_path, noise_level):
    """Same data as the HPO: canonical noise-free curve, SEED_NOISE, SEED_SPLIT."""
    return m.prepare_shared_data(data_path, noise_level)


def reference_diffusivity(m):
    """van Genuchten D(theta) with the true parameters and the HPO's fixed l, theta_r, theta_s."""
    from hpvpinn_capillarity.core.constitutive_model import FixedParam, VanGenuchtenDiffusivity
    return VanGenuchtenDiffusivity(
        KsAlpha=FixedParam(m.TRUE_PARAMS["KsAlpha"]), m=FixedParam(m.TRUE_PARAMS["m"]),
        l=FixedParam(m.L_FIXED), theta_r=FixedParam(m.THETA_R), theta_s=m.THETA_S,
    )


def noise_free_curve(m, data_path):
    raw = np.loadtxt(data_path, delimiter=",")
    return raw[:, 0], raw[:, 1]


SPEC = CaseSpec(
    case="grenoblesand",
    search_module=SEARCH_MODULE,
    label="Case 1 (Grenoble sand)",
    param_labels={"KsAlpha": r"$K_s/\alpha$ (cm$^2$/h)", "m": r"$m$"},
    D_units=r"cm$^2$/h",
    prepare_shared=prepare_shared,
    default_data="data_grenoblesand.dat",
    noise_levels=(0.01, 0.02),
    reference_params=TRUE_PARAMS,
    reference_label="Reference (true parameters)",
    reference_short="true",
    has_ground_truth=True,
    reference_diffusivity=reference_diffusivity,
    noise_free_curve=noise_free_curve,
)


if __name__ == "__main__":
    main(SPEC)

"""
letd.py
=======
Final hp-VPINN runs for Case 3 (Whatman No. 1 filter paper, experimental
data, LETd model with trainable D_wt, L, E, T and theta_r). See
final_run_common.py for what a final run is, the outputs, and the options.

There is no ground truth for this case: the reference parameter set
(TRUE_PARAMS in hparam_search_letd.py, Gerlero et al. 2022) is drawn for
comparison in the parameter panels, the D(theta) figure and the parameter
table ("Ref." column), but no recovery error or bias is computed. The
measurement uncertainty is the per-point std shipped with the data (no
synthetic noise, hence no --noise-level). Same dataset as Case 2.

Usage
-----
    # configuration selected by hparam_postprocess.py --winners, 10 fresh seeds
    uv run python letd.py

    # the winner of a given schedule instead of the selected one
    uv run python letd.py --schedule es

    # user-defined configuration
    uv run python letd.py --schedule gb --arch "W=16,L=2,act=tanh,Nel=30,p=5"

    # smoke test (separate run directory, tag suffix _ep2000)
    uv run python letd.py --n-seeds 2 --epochs 2000

    # rebuild figures and tables from the saved seeds (no training)
    uv run python letd.py --aggregate-only
    uv run python letd.py --show-config
"""

from hpvpinn_capillarity import paths
from hpvpinn_capillarity.final_runs.common import CaseSpec, main, read_constant

SEARCH_MODULE = "hpvpinn_capillarity.hpo.search_letd"

# Reference parameter set, read from the HPO module so it cannot drift.
_SRC = paths.module_source(SEARCH_MODULE)
REFERENCE_PARAMS = read_constant(_SRC, "TRUE_PARAMS")
if not REFERENCE_PARAMS:
    raise ImportError(f"Could not read TRUE_PARAMS from {_SRC}.")


def prepare_shared(m, data_path, noise_level):
    """Same data and train/test split as the HPO (SEED_SPLIT; measured std)."""
    return m.prepare_shared_data(data_path)


def reference_diffusivity(m):
    """LETd D(theta) with the reference parameter set."""
    from hpvpinn_capillarity.core.constitutive_model import FixedParam, LETdDiffusivity
    p = m.TRUE_PARAMS
    return LETdDiffusivity(
        Dwt=FixedParam(p["Dwt"]), L=FixedParam(p["L"]), E=FixedParam(p["E"]),
        T=FixedParam(p["T"]), theta_r=FixedParam(p["theta_r"]), theta_s=m.THETA_S,
    )


SPEC = CaseSpec(
    case="letd",
    search_module=SEARCH_MODULE,
    label="Case 3 (Whatman No. 1, LETd)",
    param_labels={
        "Dwt":     r"$D_{wt}$ (m$^2$/s)",
        "L":       r"$L$",
        "E":       r"$E$",
        "T":       r"$T$",
        "theta_r": r"$\theta_r$",
    },
    D_units=r"m$^2$/s",
    prepare_shared=prepare_shared,
    default_data="data_letd.dat",
    noise_levels=None,
    reference_params=REFERENCE_PARAMS,
    reference_label="Reference parameter set",
    reference_short="ref.",
    has_ground_truth=False,
    reference_diffusivity=reference_diffusivity,
    noise_free_curve=None,
)


if __name__ == "__main__":
    main(SPEC)

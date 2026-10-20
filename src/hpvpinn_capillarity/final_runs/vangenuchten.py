"""Final hp-VPINN runs for Case 2 (Whatman No. 1 filter paper, experimental
data, van Genuchten model with trainable Ks/alpha, m, l and theta_r). See
final_runs/common.py for the outputs and the options.

There is no ground truth for this case: the reference parameter set
(TRUE_PARAMS in hpo/search_vangenuchten.py, the literature values for
this dataset) is drawn for comparison in the parameter panels, the D(theta)
figure and the parameter table ("Ref." column), but no recovery error or
bias is computed. The measurement uncertainty is the per-point std shipped
with the data (no synthetic noise, hence no --noise-level).

Usage
-----
    # configuration selected by hpvpinn_capillarity.hpo.postprocess --winners, 10 fresh seeds
    python -m hpvpinn_capillarity.final_runs.vangenuchten

    # the winner of a given schedule instead of the selected one
    python -m hpvpinn_capillarity.final_runs.vangenuchten --schedule es

    # user-defined configuration
    python -m hpvpinn_capillarity.final_runs.vangenuchten --schedule gb --arch "W=16,L=2,act=tanh,Nel=30,p=5"

    # smoke test (separate run directory, tag suffix _ep2000)
    python -m hpvpinn_capillarity.final_runs.vangenuchten --n-seeds 2 --epochs 2000

    # rebuild figures and tables from the saved seeds (no training)
    python -m hpvpinn_capillarity.final_runs.vangenuchten --aggregate-only
    python -m hpvpinn_capillarity.final_runs.vangenuchten --show-config
"""

from hpvpinn_capillarity import paths
from hpvpinn_capillarity.final_runs.common import CaseSpec, main, read_constant

SEARCH_MODULE = "hpvpinn_capillarity.hpo.search_vangenuchten"

# Reference parameter set, read from the HPO module so it cannot drift.
_SRC = paths.module_source(SEARCH_MODULE)
REFERENCE_PARAMS = read_constant(_SRC, "TRUE_PARAMS")
if not REFERENCE_PARAMS:
    raise ImportError(f"Could not read TRUE_PARAMS from {_SRC}.")


def prepare_shared(m, data_path, noise_level):
    """Same data and train/test split as the HPO (SEED_SPLIT; measured std)."""
    return m.prepare_shared_data(data_path)


def reference_diffusivity(m):
    """van Genuchten D(theta) with the reference parameter set."""
    from hpvpinn_capillarity.core.constitutive_model import FixedParam, VanGenuchtenDiffusivity
    p = m.TRUE_PARAMS
    return VanGenuchtenDiffusivity(
        KsAlpha=FixedParam(p["KsAlpha"]), m=FixedParam(p["m"]),
        l=FixedParam(p["l"]), theta_r=FixedParam(p["theta_r"]), theta_s=m.THETA_S,
    )


SPEC = CaseSpec(
    case="vangenuchten",
    search_module=SEARCH_MODULE,
    label="Case 2 (Whatman No. 1, van Genuchten)",
    param_labels={
        "KsAlpha": r"$K_s/\alpha$ (m$^2$/s)",
        "m":       r"$m$",
        "l":       r"$l$",
        "theta_r": r"$\theta_r$",
    },
    D_units=r"m$^2$/s",
    prepare_shared=prepare_shared,
    default_data="data_vangenuchten.dat",
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

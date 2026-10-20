"""
hpvpinn_capillarity.hpo.search_brooks_corey_new
==============================
Unified Optuna hyperparameter search for the Brooks-Corey inverse problem
(synthetic case derived from Grenoble sand: exact curve generated with frontx,
then corrupted with Gaussian noise), covering BOTH candidate models via a
single --model flag:

    --model hp_vpinn hp-VPINN weak-form model      (hp_vpinn/)

Unknowns (2): KsAlpha = Ks/alpha  and  n.   Fixed: l = 1, theta_r = 0, alpha = 1
(frontx side: only Ks/alpha is identifiable, so alpha is pinned to 1 and the
fitted Ks IS KsAlpha).

This file lives at the repo root, as a sibling of the "hp_vpinn" and
"frontx" directories. Results for each model are written under that
model's own subdirectory.

Usage
-----
    python -m hpvpinn_capillarity.hpo.search_brooks_corey_new --model hp_vpinn --noise-level 0.01
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import jax

# Must run before any local module (hp_vpinn.* or frontx.*) creates a JAX
# array. Setting this twice (frontx.neural also sets it at import time) is
# harmless.
jax.config.update("jax_enable_x64", True)  # noqa: FBT003

from hpvpinn_capillarity import paths

import jax.numpy as jnp
import numpy as np
import optax
import optuna
from optuna.samplers import TPESampler



# ---------------------------------------------------------------------------
# Fixed experimental configuration -- shared by both models
# ---------------------------------------------------------------------------

# True parameters (used only for reporting recovery error, never for fitting).
# KsAlpha := Ks / alpha.
#
# KsAlpha_TRUE is NOT taken from nominal Ks/alpha (4.27 / (1/11.43) ~ 48.8): the
# generator's actual Ks, alpha and units are unknown. It is the value implied
# by the data file itself (Bruce-Klute inversion of data_brooks_corey.dat
# with n=1.2876, l=1, theta_s=0.312 gives a constant 0.017559 at all theta).
# It is used ONLY to report err_KsAlpha / obj_recovery -- never for fitting or
# for the Optuna objective (chi2_train + chi2_test).
KSALPHA_TRUE = 0.017559
N_TRUE       = 1.2876
THETA_S      = 0.312

TRUE_PARAMS = dict(KsAlpha=KSALPHA_TRUE, n=N_TRUE)

# Shared initial values: BOTH branches start from the same point in parameter
# space (frontx.Param(min=0) would otherwise silently start at 1.0).
KSALPHA_INIT = 1.0
N_INIT       = 1.0

# Diffusivity parameters held FIXED throughout all trials, both models
L_FIXED = 1.0
THETA_R = 0.0
I_BC    = 0.0
B_BC    = THETA_S - 1e-7

# Training budget (both models train this many steps during the search
# stage -- no early exit)
MAX_STEPS = 200_000

# hp-VPINN-only fixed settings
N_QUAD = 24      # training-time quadrature degree
Q_EVAL = 40      # held-out quadrature degree for the honest physics-loss eval
LR_INIT, LR_END, LR_STEPS = 1e-3, 1e-4, 100_000

# Shared BC weighting
LAMBDA_BC = 10.0

# Dense grid for the post-hoc strong-form residual diagnostic.
# Nondimensional [0, 1]; shared by both models so strong_form_rms is
# evaluated on literally the same grid in both.
N_GRID = 2000
PHI_DENSE = jnp.linspace(1e-4, 1.0, N_GRID)

# Seeds: exactly one split seed and one noise seed, used by both models via
# prepare_shared_data(), so both see identical train/test indices and
# identical noisy observations.
SEED_SPLIT = 7
SEED_NOISE = 42

TEST_FRACTION = 0.20

# The canonical reference curve has 5000 points; the search uses a 100-point
# ordered (non-random) subsample, evenly spaced in index along o, endpoints
# included -- comparable to the Grenoble sand case (100 points).
N_DATA_POINTS = 100
N_PARAMS = 2  # KsAlpha and n

NOISE_LEVELS = (0.01, 0.02)
PHYSICS_THRESHOLD = 1e-5

STAGE = "search"

ACTIVATIONS_NAMES = ["tanh", "sigmoid", "swish", "gelu"]

# Per-model fixed settings that legitimately differ
N_STARTUP_TRIALS = dict(hp_vpinn=25)
STUDY_NAME_PREFIX = dict(hp_vpinn="vpinn_hparam_search_brooks_corey_new")
BASE_RESULTS_DIR  = "results_brooks_corey_new"  # same base name for both models --
                                                # disambiguated by model subdirectory


# ---------------------------------------------------------------------------
# Lazy per-model backend loading
# ---------------------------------------------------------------------------

def _load_vpinn_backend():

    global FeedForwardNN, CapillarityBoltzmannPDE, PINN
    global FixedParam, LowerBoundedParam, BrooksAndCoreyDiffusivity
    global CapillarityScaling, MeasurementSet, VPINNTrainer, VPINNLoss
    global evaluate_strong_form_rms_vpinn, ACTIVATIONS_VPINN

    from hpvpinn_capillarity.core.nn_model import FeedForwardNN
    from hpvpinn_capillarity.core.pde import CapillarityBoltzmannPDE
    from hpvpinn_capillarity.core.pinn_model import PINN
    from hpvpinn_capillarity.core.constitutive_model import (
        FixedParam, LowerBoundedParam, BrooksAndCoreyDiffusivity,
    )
    from hpvpinn_capillarity.core.transformations import CapillarityScaling
    from hpvpinn_capillarity.core.datasets import MeasurementSet
    from hpvpinn_capillarity.core.train_vpinn import VPINNTrainer
    from hpvpinn_capillarity.core.vpinn_loss import VPINNLoss
    from hpvpinn_capillarity.core.strong_form_residual import evaluate_strong_form_rms as evaluate_strong_form_rms_vpinn

    ACTIVATIONS_VPINN = {
        "tanh": jax.nn.tanh, "sigmoid": jax.nn.sigmoid,
        "swish": jax.nn.swish, "gelu": jax.nn.gelu,
    }


# ---------------------------------------------------------------------------
# Shared data preparation -- the single place split/noise/oi are decided
# ---------------------------------------------------------------------------

def prepare_shared_data(data_path: str, noise_level: float) -> dict:
    """Load the canonical noise-free Brooks-Corey reference curve once, add
    noise once, split once -- and hand identical arrays to both model branches.

    `data_path` must point at the ONE canonical, noise-free, comma-separated
    reference curve (o, theta in physical units), generated once via frontx.

    Returns the split/noised train and test arrays, all in physical units,
    plus `oi` (== phi_ref) computed from the FULL dataset before the split.
    """
    raw = np.loadtxt(data_path, delimiter=",")  # float64 by default
    # Ordered, evenly spaced subsample (deterministic, no RNG): picks
    # N_DATA_POINTS indices spanning the full curve, first and last included.
    sub = np.round(np.linspace(0, len(raw) - 1, N_DATA_POINTS)).astype(int)
    assert len(np.unique(sub)) == N_DATA_POINTS
    o_all, theta_clean = raw[sub, 0], raw[sub, 1]

    # phi_ref (VPINN) / oi (frontx): same quantity, computed once from the
    # full dataset.
    oi = 1.05 * float(o_all[-1])

    rng_noise = np.random.default_rng(SEED_NOISE)
    if noise_level > 0.0:
        theta_noisy = theta_clean + rng_noise.normal(0.0, noise_level, size=theta_clean.shape)
    else:
        theta_noisy = theta_clean.copy()

    rng_split = np.random.default_rng(SEED_SPLIT)
    idx = rng_split.permutation(len(o_all))
    n_test = max(1, int(TEST_FRACTION * len(o_all)))
    test_idx  = np.sort(idx[:n_test])
    train_idx = np.sort(idx[n_test:])

    return dict(
        oi          = oi,
        o_train     = o_all[train_idx],
        theta_train = theta_noisy[train_idx],
        o_test      = o_all[test_idx],
        theta_test  = theta_noisy[test_idx],
        noise_level = float(noise_level),
    )


# ---------------------------------------------------------------------------
# VPINN branch
# ---------------------------------------------------------------------------

def _vpinn_prepare(shared: dict) -> dict:
    """Adapt the shared physical-unit arrays into VPINN's nondimensional
    MeasurementSet representation. Noise/split already happened in
    prepare_shared_data(); this only rescales and wraps."""
    scaling = CapillarityScaling(
        phi_ref = shared["oi"],
        theta_i = I_BC,
        theta_b = float(B_BC),
    )
    sigma_nd = float(scaling.sigma_forward(shared["noise_level"]))

    def _to_measurement_set(o, theta):
        phi_nd   = np.array(scaling.phi_forward(jnp.array(o)))
        theta_nd = np.array(scaling.theta_forward(jnp.array(theta)))
        return MeasurementSet(phi_nd, theta_nd, sigma=sigma_nd, theta_exact=None, true_value=False)

    return dict(
        scaling             = scaling,
        phi_ref             = shared["oi"],
        measurements_train  = _to_measurement_set(shared["o_train"], shared["theta_train"]),
        measurements_test   = _to_measurement_set(shared["o_test"],  shared["theta_test"]),
    )


def _vpinn_build_pinn(n_layers, width, act_name, phi_ref, scaling, init_seed):
    diffusivity = BrooksAndCoreyDiffusivity(
        KsAlpha = LowerBoundedParam(value=KSALPHA_INIT, min=0.0),
        n       = LowerBoundedParam(value=N_INIT, min=0.0),
        l       = FixedParam(value=L_FIXED),
        theta_r = FixedParam(value=THETA_R),
        theta_s = THETA_S,
    )
    nn_model = FeedForwardNN(
        width=width, depth=n_layers, activation=ACTIVATIONS_VPINN[act_name],
        key=jax.random.PRNGKey(init_seed),
    )
    pde = CapillarityBoltzmannPDE(diffusivity=diffusivity, phi_ref=phi_ref, scaling=scaling)
    return PINN(nn_model=nn_model, pde=pde)


def _vpinn_evaluate_pde_loss_held_out(trained_pinn, n_elements, poly_degree, q_eval):
    vpinn_loss_eval = VPINNLoss(
        phi_min=0.0, phi_max=1.0,
        phi_ref=trained_pinn.pde.phi_ref, scaling=trained_pinn.pde.scaling,
        n_elements=n_elements, poly_degree=poly_degree, n_quad=q_eval,
        element_clustering=1.0,
    )
    pde_loss_eval, _, _, _ = vpinn_loss_eval(trained_pinn)
    return float(pde_loss_eval)


def sample_vpinn_hparams(trial: optuna.Trial) -> dict:
    return dict(
        nn_n_layers = trial.suggest_int("nn_n_layers", 1, 3),
        nn_width    = trial.suggest_categorical("nn_width", [8, 12, 16, 24, 32]),
        activation  = trial.suggest_categorical("activation", ACTIVATIONS_NAMES),
        n_elements  = trial.suggest_int("n_elements", 10, 40, step=10),
        poly_degree = trial.suggest_int("poly_degree", 3, 6),
    )


def run_vpinn_trial(trial, sampled, data, noise_level, warmup_epochs, transition_epochs):
    init_seed = trial.number
    hparams = dict(**sampled, noise_level=noise_level, init_seed=init_seed)

    pinn = _vpinn_build_pinn(
        sampled["nn_n_layers"], sampled["nn_width"], sampled["activation"],
        data["phi_ref"], data["scaling"], init_seed,
    )
    lr_schedule = optax.exponential_decay(
        init_value=LR_INIT, transition_steps=LR_STEPS,
        decay_rate=LR_END / LR_INIT, end_value=LR_END,
    )
    # measurements_val=None disables early stopping/gating on the VPINN
    # side too, matching frontx's freeze_on_convergence=False during search.
    trainer = VPINNTrainer(
        pinn               = pinn,
        measurements_train = data["measurements_train"],
        measurements_val   = None,
        n_params           = N_PARAMS,
        lambda_data        = 1.0,
        lambda_bc          = LAMBDA_BC,
        lr_schedule_fn      = lr_schedule,
        n_elements          = sampled["n_elements"],
        poly_degree         = sampled["poly_degree"],
        n_quad              = N_QUAD,
        use_lbfgs           = False,
    )

    t0 = time.perf_counter()
    trained_pinn, history = trainer.fit(
        epochs=MAX_STEPS, warmup_epochs=warmup_epochs, transition_epochs=transition_epochs,
    )
    elapsed = time.perf_counter() - t0

    final_total      = float(np.mean(history.total_loss[-10_000:]))
    final_data       = float(np.mean(history.data_loss[-10_000:]))
    final_physics    = float(np.mean(history.physics_loss[-10_000:]))
    final_pde        = float(np.mean(history.pde_loss[-10_000:]))
    final_dirichlet  = float(np.mean(history.dirichlet_loss[-10_000:]))
    final_derivative = float(np.mean(history.derivative_loss[-10_000:]))

    if np.isnan(final_physics) or np.isnan(final_data):
        return hparams, dict(elapsed_s=elapsed), True, None

    diff = trained_pinn.pde.diffusivity
    n_trainable_params = sum(
        1 for name in diff.__dataclass_fields__ if hasattr(getattr(diff, name), "hat")
    )
    # Guard: the physical problem has exactly N_PARAMS unknowns (KsAlpha, n).
    assert n_trainable_params == N_PARAMS, (
        f"VPINN diffusivity has {n_trainable_params} trainable params, expected {N_PARAMS}"
    )
    chi2_train = float(data["measurements_train"].chi2_nu(trained_pinn.nn_model, n_params=n_trainable_params))
    chi2_test  = float(data["measurements_test"].chi2_nu(trained_pinn.nn_model, n_params=n_trainable_params))

    pde_loss_eval = _vpinn_evaluate_pde_loss_held_out(
        trained_pinn, sampled["n_elements"], sampled["poly_degree"], Q_EVAL
    )
    strong_form_rms = float(evaluate_strong_form_rms_vpinn(trained_pinn, PHI_DENSE))

    recovered = {name: float(getattr(diff, name).value()) for name in TRUE_PARAMS}

    metrics = dict(
        chi2_nu_train=chi2_train, chi2_nu_test=chi2_test,
        final_total=final_total, final_data=final_data, final_physics=final_physics,
        final_pde=final_pde, final_dirichlet=final_dirichlet, final_derivative=final_derivative,
        pde_loss_eval=pde_loss_eval, q_eval=Q_EVAL,
        strong_form_rms=strong_form_rms, n_grid_eval=N_GRID,
        elapsed_s=elapsed,
    )
    return hparams, metrics, False, recovered


# ---------------------------------------------------------------------------
# Shared objective machinery
# ---------------------------------------------------------------------------

BACKENDS = dict(
    hp_vpinn=dict(
        loader=_load_vpinn_backend, prepare=_vpinn_prepare,
        sample=sample_vpinn_hparams, run_trial=run_vpinn_trial,
    ),
)


def make_objective(model: str, data: dict, results_path: str, noise_level: float,
                    warmup_epochs: int, transition_epochs: int) -> callable:
    backend = BACKENDS[model]

    def objective(trial: optuna.Trial) -> float:
        sampled = backend["sample"](trial)
        print(f"\n[Trial {trial.number}] {sampled}")

        try:
            hparams, metrics, diverged, recovered = backend["run_trial"](
                trial, sampled, data, noise_level, warmup_epochs, transition_epochs
            )
        except Exception as exc:
            warnings.warn(f"Trial {trial.number} failed: {exc}")
            _save_result(results_path, trial.number, sampled, {}, failed=True)
            return float("inf")

        if diverged:
            warnings.warn(f"Trial {trial.number}: NaN in history, diverged during training.")
            _save_result(results_path, trial.number, hparams,
                         dict(**metrics), failed=True)
            return float("inf")

        chi2_train = metrics["chi2_nu_train"]
        chi2_test  = metrics["chi2_nu_test"]
        obj = chi2_train + chi2_test

        param_errors = {
            f"err_{name}": abs(recovered[name] - TRUE_PARAMS[name]) / abs(TRUE_PARAMS[name])
            for name in TRUE_PARAMS
        }
        obj_recovery = float(np.mean(list(param_errors.values())))

        metrics = dict(
            **metrics, obj=obj, obj_recovery=obj_recovery,
            **recovered, **param_errors,
        )

        if metrics["final_physics"] > PHYSICS_THRESHOLD:
            _save_result(results_path, trial.number, hparams, metrics, failed=False, pruned=True)
            raise optuna.TrialPruned()

        print(
            f"  chi2_nu_train={chi2_train:.3f}  chi2_nu_test={chi2_test:.3f}"
            f"  obj={obj:.3f}  obj_recovery={obj_recovery:.3f}"
            f"  data={metrics['final_data']:.3e}  phys={metrics['final_physics']:.3e}"
            f"  strong_rms[N={N_GRID}]={metrics['strong_form_rms']:.3e}"
            f"  t={metrics['elapsed_s']:.0f}s"
        )
        for name in TRUE_PARAMS:
            print(
                f"    {name}: {recovered[name]:.4g}  "
                f"(true={TRUE_PARAMS[name]:.4g}, err={param_errors[f'err_{name}']:.1%})"
            )

        _save_result(results_path, trial.number, hparams, metrics, failed=False)
        return obj

    return objective


def _save_result(path, trial_number, hparams, metrics, failed, pruned=False):
    record = dict(trial=trial_number, failed=failed, pruned=pruned, stage=STAGE, **hparams, **metrics)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    valid_noise_strs = [str(s) for s in NOISE_LEVELS]

    parser = argparse.ArgumentParser(
        description="hp-VPINN hyperparameter search — Brooks and Corey (new, noisy)"
    )
    parser.add_argument("--model", required=True, choices=["hp_vpinn"])
    parser.add_argument("--data", default=str(paths.data_file("data_brooks_corey.dat")),
                        help="Path to the single canonical noise-free, comma-separated "
                             "reference curve (o, theta), shared by both models.")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--results-dir", default=None,
                        help="Overrides the per-model default results-dir base name.")
    parser.add_argument("--study-name", default=None,
                        help="Overrides the per-model default study-name prefix.")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--transition-epochs", type=int, default=20_000)
    parser.add_argument(
        "--noise-level", type=str, required=True, choices=valid_noise_strs,
        metavar=f"{{{','.join(valid_noise_strs)}}}",
        help=f"Measurement noise std dev. Must be one of: {valid_noise_strs}",
    )
    args = parser.parse_args()

    model = args.model
    BACKENDS[model]["loader"]()  # pull in only this model's heavy deps

    noise_level = float(args.noise_level)
    sched_tag = "es" if args.warmup_epochs >= MAX_STEPS else "gb"
    noise_tag = f"{int(round(noise_level * 100)):01d}pct_noise"

    base_results_dir = args.results_dir or BASE_RESULTS_DIR
    study_prefix      = args.study_name or STUDY_NAME_PREFIX[model]

    results_dir = paths.model_dir(model) / f"{base_results_dir}_{sched_tag}_{noise_tag}"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f"hparam_results_{noise_tag}.jsonl"
    db_path      = results_dir / "study.db"
    study_name   = f"{study_prefix}_{sched_tag}_sigma{int(round(noise_level * 1000)):04d}"

    print("=" * 60)
    print(f"Unified hyperparameter search — model={model}, Brooks and Corey (new)")
    print(f"  Noise level:   sigma={noise_level}")
    print(f"  Schedule:      {sched_tag} (warmup={args.warmup_epochs}, transition={args.transition_epochs})")
    print(f"  Max steps:     {MAX_STEPS}")
    print(f"  Trials:        {args.n_trials}")
    print(f"  Test fraction: {TEST_FRACTION:.0%}")
    print(f"  Data:          {args.data}")
    print(f"  Results:       {results_path}")
    print(f"  Optuna DB:     {db_path}")
    print(f"  Dashboard:     optuna-dashboard sqlite:///{db_path}")
    print("=" * 60)

    shared = prepare_shared_data(args.data, noise_level)
    data = BACKENDS[model]["prepare"](shared)
    print(f"  Train points: {len(shared['o_train'])}  |  Test points: {len(shared['o_test'])}")

    sampler = TPESampler(
        seed=0, n_startup_trials=N_STARTUP_TRIALS[model], multivariate=True, group=True,
    )
    study = optuna.create_study(
        direction="minimize", study_name=study_name,
        storage=f"sqlite:///{db_path}", sampler=sampler, load_if_exists=True,
    )

    objective = make_objective(model, data, str(results_path), noise_level,
                                args.warmup_epochs, args.transition_epochs)
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1)

    best = study.best_trial
    print("\n" + "=" * 60)
    print("Best trial")
    print("=" * 60)
    print(f"  Objective : {best.value:.4f}")
    print(f"  Params    : {best.params}")
    print("=" * 60)


if __name__ == "__main__":
    main()

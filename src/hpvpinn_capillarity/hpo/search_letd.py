"""
hparam_search_letd.py
=======================
Unified Optuna hyperparameter search for the LETd EXPERIMENTAL inverse
problem, covering both models via a single --model flag:

    --model hp_vpinn hp-VPINN weak-form model      (hp_vpinn/)
    --model frontx   frontx strong-form PINN model (frontx/)

Same architecture as hparam_search_vangenuchten.py (see that file's
docstring for the full rationale) -- lives at the repo root, results land
under each model's own subdirectory, and prepare_shared_data() is the ONLY
place the dataset is loaded and split, consumed by both branches.

Usage
-----
    python hparam_search_letd.py --model hp_vpinn
    python hparam_search_letd.py --model frontx --n-trials 80
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

import jax.numpy as jnp
import numpy as np
import optax
import optuna
from optuna.samplers import TPESampler

REPO_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Fixed experimental configuration -- shared by both models
# ---------------------------------------------------------------------------

# Reference parameter values (Gerlero et al., 2022) -- used only for
# reporting recovery error, never for fitting. Direct 1:1 correspondence on
# both branches, unlike van Genuchten's KsAlpha/alpha translation.
TRUE_PARAMS = dict(
    Dwt     = 1.004e-3,
    L       = 1.356,
    E       = 10_010.0,
    T       = 1.224,
    theta_r = 0.00625,
)

# Fixed physical constants, shared by both models (same dataset as
# hparam_search_vangenuchten.py)
THETA_S = 0.7
THETA_I = 0.025
THETA_B = THETA_S - 1e-7

MAX_STEPS = 200_000

N_QUAD = 24
Q_EVAL = 40
LR_INIT, LR_END, LR_STEPS = 1e-3, 1e-4, 100_000

LAMBDA_BC = 10.0

N_GRID = 2000
PHI_DENSE = jnp.linspace(1e-4, 1.0, N_GRID)

# One split seed, used by both models via prepare_shared_data()
SEED_SPLIT = 7

TEST_FRACTION = 0.20
N_PARAMS = 5   # Dwt, L, E, T, theta_r

PHYSICS_THRESHOLD = 1e-5

STAGE = "search"

ACTIVATIONS_NAMES = ["tanh", "sigmoid", "swish", "gelu"]

# Shared initial values -- both branches must start from the same point in
# parameter space. Dwt's "constant" is a formula, not a literal, since it's
# dataset-dependent (matches hp_vpinn's phi_ref**2 exactly -- frontx's
# original script used o_all[-1]**2 instead, missing the 1.05 margin
# phi_ref/oi includes; see module docstring history for the derivation).
# L, E, T, theta_r are literals, chosen to match hp_vpinn's explicit values
# rather than frontx.Param's derived defaults (which agreed for L/E/T by
# coincidence, but not for theta_r -- see the cross-check discussion).
L_INIT       = 0.0
E_INIT       = 1.0
T_INIT       = 0.0
THETA_R_INIT = 0.01

N_STARTUP_TRIALS = dict(hp_vpinn=25, frontx=15)   # matches search-space dimensionality (5 vs 3)
STUDY_NAME_PREFIX = dict(hp_vpinn="vpinn_hparam_search_letd", frontx="neural_letd_hparam_search")
BASE_RESULTS_DIR = "results_letd"   # same base name for both models, disambiguated
                                     # by which model subdirectory they're written under


# ---------------------------------------------------------------------------
# Lazy per-model backend loading
# ---------------------------------------------------------------------------

def _load_vpinn_backend():
    hp_vpinn_dir = str(REPO_ROOT / "hp_vpinn")
    if hp_vpinn_dir not in sys.path:
        sys.path.insert(0, hp_vpinn_dir)

    global FeedForwardNN, CapillarityBoltzmannPDE, PINN
    global BoundedParam, FreeParam, LowerBoundedParam, LETdDiffusivity
    global CapillarityScaling, MeasurementSet, VPINNTrainer, VPINNLoss
    global evaluate_strong_form_rms_vpinn, ACTIVATIONS_VPINN

    from nn_model import FeedForwardNN
    from pde import CapillarityBoltzmannPDE
    from pinn_model import PINN
    from constitutive_model import (
        BoundedParam, FreeParam, LowerBoundedParam, LETdDiffusivity,
    )
    from transformations import CapillarityScaling
    from datasets import MeasurementSet
    from train_vpinn import VPINNTrainer
    from vpinn_loss import VPINNLoss
    from strong_form_residual import evaluate_strong_form_rms as evaluate_strong_form_rms_vpinn

    ACTIVATIONS_VPINN = {
        "tanh": jax.nn.tanh, "sigmoid": jax.nn.sigmoid,
        "swish": jax.nn.swish, "gelu": jax.nn.gelu,
    }


def _load_frontx_backend():
    global frontx, LETd, ACTIVATIONS_FRONTX

    import frontx
    import frontx.neural
    from frontx.models import LETd

    ACTIVATIONS_FRONTX = {
        "tanh": jax.nn.tanh, "sigmoid": jax.nn.sigmoid,
        "swish": jax.nn.swish, "gelu": jax.nn.gelu,
    }


# ---------------------------------------------------------------------------
# Shared data preparation -- the single place the split is decided
# ---------------------------------------------------------------------------

def prepare_shared_data(data_path: str) -> dict:
    """Load the canonical LETd experimental dataset once, split once -- and
    hand identical arrays to both model branches.

    No noise is added: sigma (column 3) is the real per-point measurement
    std shipped with the data, used as-is by both models.
    """
    raw = np.loadtxt(data_path, delimiter=",")  # float64 by default
    o_all, theta_all, std_all = raw[:, 0], raw[:, 1], raw[:, 2]

    oi = 1.05 * float(o_all[-1])   # == phi_ref; computed from the FULL dataset

    rng_split = np.random.default_rng(SEED_SPLIT)
    idx = rng_split.permutation(len(o_all))
    n_test = max(1, int(TEST_FRACTION * len(o_all)))
    test_idx  = np.sort(idx[:n_test])
    train_idx = np.sort(idx[n_test:])

    return dict(
        oi          = oi,
        o_train     = o_all[train_idx], theta_train = theta_all[train_idx], std_train = std_all[train_idx],
        o_test      = o_all[test_idx],  theta_test  = theta_all[test_idx],  std_test  = std_all[test_idx],
    )


# ---------------------------------------------------------------------------
# hp_vpinn branch
# ---------------------------------------------------------------------------

def _vpinn_prepare(shared: dict) -> dict:
    scaling = CapillarityScaling(phi_ref=shared["oi"], theta_i=THETA_I, theta_b=THETA_B)

    def _to_measurement_set(o, theta, std):
        phi_nd   = np.array(scaling.phi_forward(jnp.array(o)))
        theta_nd = np.array(scaling.theta_forward(jnp.array(theta)))
        sigma_nd = np.array(scaling.sigma_forward(jnp.array(std)))   # per-point, not scalar
        return MeasurementSet(phi_nd, theta_nd, sigma=sigma_nd, theta_exact=None, true_value=False)

    return dict(
        scaling             = scaling,
        phi_ref             = shared["oi"],
        measurements_train  = _to_measurement_set(shared["o_train"], shared["theta_train"], shared["std_train"]),
        measurements_test   = _to_measurement_set(shared["o_test"],  shared["theta_test"],  shared["std_test"]),
    )


def _vpinn_build_pinn(n_layers, width, act_name, phi_ref, scaling, init_seed):
    diffusivity = LETdDiffusivity(
        Dwt     = LowerBoundedParam(value=phi_ref ** 2, min=0.0),
        L       = FreeParam(value=L_INIT),
        E       = LowerBoundedParam(value=E_INIT, min=0.0),
        T       = FreeParam(value=T_INIT),
        theta_r = BoundedParam(value=THETA_R_INIT, min=0.0, max=THETA_I),
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


def run_vpinn_trial(trial, sampled, data, warmup_epochs, transition_epochs):
    init_seed = trial.number
    hparams = dict(**sampled, init_seed=init_seed)

    pinn = _vpinn_build_pinn(
        sampled["nn_n_layers"], sampled["nn_width"], sampled["activation"],
        data["phi_ref"], data["scaling"], init_seed,
    )
    lr_schedule = optax.exponential_decay(
        init_value=LR_INIT, transition_steps=LR_STEPS,
        decay_rate=LR_END / LR_INIT, end_value=LR_END,
    )
    trainer = VPINNTrainer(
        pinn               = pinn,
        measurements_train = data["measurements_train"],
        measurements_val   = None,   # disables early stopping, matches frontx's freeze_on_convergence=False
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
# frontx branch
# ---------------------------------------------------------------------------

def _frontx_chi2_nu(sol, o, theta, std, n_params):
    pred = np.array(sol(jnp.array(o)), dtype=np.float64)
    return float(np.sum(((pred - theta) / std) ** 2) / max(len(o) - n_params, 1))


def _frontx_get_param(p):
    v = p.value
    return float(v() if callable(v) else v)


def sample_frontx_hparams(trial: optuna.Trial) -> dict:
    return dict(
        nn_n_layers = trial.suggest_int("nn_n_layers", 1, 3),
        nn_width    = trial.suggest_categorical("nn_width", [8, 12, 16, 24, 32]),
        activation  = trial.suggest_categorical("activation", ACTIVATIONS_NAMES),
    )


def run_frontx_trial(trial, sampled, data, warmup_epochs, transition_epochs):
    hparams = dict(
        **sampled,
        warmup_epochs=warmup_epochs, transition_epochs=transition_epochs,
        init_seed=trial.number,
    )

    D_train = LETd(
        Dwt         = frontx.Param(value=data["oi"] ** 2, min=0.0),   # matches hp_vpinn's phi_ref**2 exactly
        L           = frontx.Param(value=L_INIT),
        E           = frontx.Param(value=E_INIT, min=0.0),
        T           = frontx.Param(value=T_INIT),
        theta_range = (frontx.Param(value=THETA_R_INIT, min=0.0, max=float(THETA_I)), float(THETA_S)),
    )

    t0 = time.perf_counter()
    sol, history = frontx.neural.fit(
        D_train,
        data["o_train"], data["theta_train"], data["std_train"],
        i=float(THETA_I), b=float(THETA_B),
        oi=data["oi"],   # shared oi, not derived from o_train alone
        width=sampled["nn_width"], depth=sampled["nn_n_layers"],
        activation=ACTIVATIONS_FRONTX[sampled["activation"]],
        lambda_bc=LAMBDA_BC,
        key=jax.random.PRNGKey(trial.number),
        max_steps=MAX_STEPS,
        warmup_epochs=warmup_epochs, transition_epochs=transition_epochs,
        freeze_on_convergence=False,   # always run the full budget during HPO
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

    chi2_train = _frontx_chi2_nu(sol, data["o_train"], data["theta_train"], data["std_train"], N_PARAMS)
    chi2_test  = _frontx_chi2_nu(sol, data["o_test"],  data["theta_test"],  data["std_test"],  N_PARAMS)

    strong_form_rms = frontx.neural.evaluate_strong_form_rms(sol, PHI_DENSE)

    D_fit = sol.D
    recovered = dict(
        Dwt     = _frontx_get_param(D_fit.Dwt),
        L       = _frontx_get_param(D_fit.L),
        E       = _frontx_get_param(D_fit.E),
        T       = _frontx_get_param(D_fit.T),
        theta_r = _frontx_get_param(D_fit.theta_range[0]),
    )

    metrics = dict(
        chi2_nu_train=chi2_train, chi2_nu_test=chi2_test,
        final_total=final_total, final_data=final_data, final_physics=final_physics,
        final_pde=final_pde, final_dirichlet=final_dirichlet, final_derivative=final_derivative,
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
    frontx=dict(
        loader=_load_frontx_backend, prepare=lambda shared: shared,
        sample=sample_frontx_hparams, run_trial=run_frontx_trial,
    ),
)


def make_objective(model: str, data: dict, results_path: str,
                    warmup_epochs: int, transition_epochs: int) -> callable:
    backend = BACKENDS[model]

    def objective(trial: optuna.Trial) -> float:
        sampled = backend["sample"](trial)
        print(f"\n[Trial {trial.number}] {sampled}")

        try:
            hparams, metrics, diverged, recovered = backend["run_trial"](
                trial, sampled, data, warmup_epochs, transition_epochs
            )
        except Exception as exc:
            warnings.warn(f"Trial {trial.number} failed: {exc}")
            _save_result(results_path, trial.number, sampled, {}, failed=True)
            return float("inf")

        if diverged:
            warnings.warn(f"Trial {trial.number}: NaN in history, diverged during training.")
            _save_result(results_path, trial.number, hparams, metrics, failed=True)
            return float("inf")

        chi2_train = metrics["chi2_nu_train"]
        chi2_test  = metrics["chi2_nu_test"]
        #gap          = max(0.0, chi2_test - chi2_train)
        #relative_gap = min(gap / (chi2_train + 1e-8), 10.0)
        #obj = max(0.0, chi2_test - 1.0) + (chi2_test * relative_gap if chi2_test > 1.0 else 0.0)
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
    parser = argparse.ArgumentParser(
        description="Unified hp-VPINN / frontx hyperparameter search — LETd (experimental)"
    )
    parser.add_argument("--model", required=True, choices=["hp_vpinn", "frontx"])
    parser.add_argument("--data", default=str(REPO_ROOT / "data_letd.dat"),
                        help="Path to the single canonical experimental dataset "
                             "(phi, theta, std columns), shared by both models.")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--results-dir", default=None,
                        help="Overrides the per-model default results-dir base name.")
    parser.add_argument("--study-name", default=None,
                        help="Overrides the per-model default study-name prefix.")
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--transition-epochs", type=int, default=20_000)
    args = parser.parse_args()

    model = args.model
    BACKENDS[model]["loader"]()

    sched_tag = "es" if args.warmup_epochs >= MAX_STEPS else "gb"

    base_results_dir = args.results_dir or BASE_RESULTS_DIR
    study_prefix      = args.study_name or STUDY_NAME_PREFIX[model]

    results_dir = REPO_ROOT / model / f"{base_results_dir}_{sched_tag}"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "hparam_results_letd.jsonl"
    db_path      = results_dir / "study.db"
    study_name   = f"{study_prefix}_{sched_tag}"

    print("=" * 60)
    print(f"Unified hyperparameter search — model={model}, LETd (experimental)")
    print(f"  Schedule:      {sched_tag} (warmup={args.warmup_epochs}, transition={args.transition_epochs})")
    print(f"  Max steps:     {MAX_STEPS}")
    print(f"  Trials:        {args.n_trials}")
    print(f"  Test fraction: {TEST_FRACTION:.0%}")
    print(f"  Data:          {args.data}")
    print(f"  Results:       {results_path}")
    print(f"  Optuna DB:     {db_path}")
    print(f"  Dashboard:     optuna-dashboard sqlite:///{db_path}")
    print("=" * 60)

    shared = prepare_shared_data(args.data)
    data = BACKENDS[model]["prepare"](shared)
    print(f"  Train points: {len(shared['o_train'])}  |  Test points: {len(shared['o_test'])}")

    sampler = TPESampler(
        seed=0, n_startup_trials=N_STARTUP_TRIALS[model], multivariate=True, group=True,
    )
    study = optuna.create_study(
        direction="minimize", study_name=study_name,
        storage=f"sqlite:///{db_path}", sampler=sampler, load_if_exists=True,
    )

    objective = make_objective(model, data, str(results_path),
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

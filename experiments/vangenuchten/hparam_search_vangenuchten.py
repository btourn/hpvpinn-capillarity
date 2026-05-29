"""
hparam_search_vangenuchten.py
==============================
Optuna-based hyperparameter search for the hp-VPINN inverse problem
on the van~Genuchten dataset (Case 2, Whatman No.~1 filter paper).

Usage
-----
    uv run python experiments/vangenuchten/hparam_search_vangenuchten.py \
        --config es --n-trials 250

Arguments
---------
    --config       : weighting strategy: 'gb' (gradient balancing, default)
                     or 'es' (exponential schedule).
    --n-trials     : number of Optuna trials (default: 50).
    --results-dir  : output directory (default: ./results/results_vangenuchten).
    --study-name   : Optuna study name (default: vpinn_hparam_search).
    --data         : path to the dataset file.

Search space
------------
    nn_n_layers  {1, 2, 3}           nn_width     {8, 16, 32}
    activation   {tanh, sigmoid,      n_elements   {10, 20, 30, 40, 50}
                  GeLU, Swish}        poly_degree  {3, 4, 5, 6}

Note
----
The van~Genuchten dataset provides per-point measurement uncertainties
(column 3 of the data file), used directly as sigma. No noise level
argument is required and no synthetic noise is added.

Results
-------
Results are appended trial-by-trial to
<results-dir>/hparam_results_vangenuchten.jsonl and persisted to
<results-dir>/study.db (viewable with optuna-dashboard).
"""

import argparse
import collections
import json
import os
import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import optax
import optuna
from optuna.samplers import TPESampler

from hpvpinn.nn_model           import FeedForwardNN
from hpvpinn.pde                import CapillarityBoltzmannPDE
from hpvpinn.pinn_model         import PINN
from hpvpinn.constitutive_model import (
    BoundedParam, FreeParam, LowerBoundedParam, VanGenuchtenDiffusivity,
)
from hpvpinn.transformations    import CapillarityScaling
from hpvpinn.datasets           import MeasurementSet
from hpvpinn.train_vpinn        import VPINNTrainer


# ---------------------------------------------------------------------------
# Fixed experimental / training configuration
# ---------------------------------------------------------------------------

# True parameters (used only for reporting recovery error, not for training)
TRUE_PARAMS = dict(
    KsAlpha = 2.079e-6, #m2/s   #74.84  #cm2/h
    m       = 0.8764,
    l       = 2.344,
    theta_r = 0.004943,
)

# Fixed physical constants
THETA_S = 0.7
THETA_I = 0.025
THETA_B = THETA_S - 1e-7

# Training budget
MAX_EPOCHS = 200_000

# LR schedule — matches vangenuchten.py exactly
LR_INIT  = 1e-3
LR_END   = 1e-4
LR_STEPS = 100_000

# hp-VPINN quadrature degree (fixed, not a search hyperparameter)
N_QUAD = 8

# Random seed for the train/test index split (fixed for reproducibility)
SEED_SPLIT = 456

# Fraction of points held out as a validation set
VAL_FRACTION  = 0.0

# Fraction of points held out as a test set
TEST_FRACTION = 0.20

# Physics loss threshold below which a trial is considered converged and above which it is pruned
PHYSICS_THRESHOLD = 1e-5

# Number of trainable diffusivity parameters (KsAlpha, m, l, theta_r)
N_PARAMS = 4  

# ---------------------------------------------------------------------------
# Available activation functions
# ---------------------------------------------------------------------------

ACTIVATIONS = {
    "tanh"    : jax.nn.tanh,
    "sigmoid" : jax.nn.sigmoid,
    "swish"   : jax.nn.swish,
    "gelu"    : jax.nn.gelu,
}


# ---------------------------------------------------------------------------
# Data preparation — called once, shared across all trials
# ---------------------------------------------------------------------------

def prepare_data(data_path: str) -> dict:
    """Load the vangenuchten dataset, apply scaling, and split into train/test.

    Unlike the Grenoble sand case, sigma is read directly from the data file
    (column 3) rather than added artificially. There is therefore no
    noise_level argument and only one dataset per run.

    Returns
    -------
    dict with keys:
        scaling           : CapillarityScaling instance
        phi_ref           : float
        measurements_train : MeasurementSet (train split)
        measurements_val   : MeasurementSet or None (val split)
        measurements_test  : MeasurementSet (test split)
    """
    data      = np.loadtxt(data_path, skiprows=1).astype(np.float32)
    phi_raw   = data[:, 0]
    theta_raw = data[:, 1]
    std_raw   = data[:, 2]

    phi_ref = 1.05 * float(phi_raw[-1])
    scaling = CapillarityScaling(
        phi_ref = phi_ref,
        theta_i = THETA_I,
        theta_b = THETA_B,
    )

    phi_nd   = np.array(scaling.phi_forward(jnp.array(phi_raw)))
    theta_nd = np.array(scaling.theta_forward(jnp.array(theta_raw)))
    sigma_nd = np.array(scaling.sigma_forward(jnp.array(std_raw)))

    measurements_all = MeasurementSet(
        phi=phi_nd,
        theta=theta_nd,
        sigma=sigma_nd,
        true_value=False,
    )

    measurements_train, measurements_val, measurements_test = measurements_all.split(
        val_fraction  = VAL_FRACTION,
        test_fraction = TEST_FRACTION,
        key           = jax.random.PRNGKey(SEED_SPLIT),
    )

    return dict(
        scaling            = scaling,
        phi_ref            = phi_ref,
        measurements_train = measurements_train,
        measurements_val   = measurements_val,
        measurements_test  = measurements_test,
    )


# ---------------------------------------------------------------------------
# Build a fresh PINN for each trial
# ---------------------------------------------------------------------------

def build_pinn(
    n_layers       : int,
    width          : int,
    activation_name: str,
    phi_ref        : float,
    scaling,
    init_seed      : int,
) -> PINN:
    """Construct a fresh PINN with the given architecture and random seed."""
    diffusivity = VanGenuchtenDiffusivity(
        KsAlpha = LowerBoundedParam(value=1.e-4, min=0.0),
        m       = BoundedParam(value=0.5, min=0.0, max=1.0),
        l       = FreeParam(value=1.),
        theta_r = BoundedParam(value=0.02, min=0.0, max=THETA_I),
        theta_s = THETA_S,
    )
    nn_model = FeedForwardNN(
        width      = width,
        depth      = n_layers,
        activation = ACTIVATIONS[activation_name],
        key        = jax.random.PRNGKey(init_seed),
    )
    pde = CapillarityBoltzmannPDE(
        diffusivity = diffusivity,
        phi_ref     = phi_ref,
        scaling     = scaling,
    )
    return PINN(nn_model=nn_model, pde=pde)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(data: dict, results_path: str, warmup_epochs: int, transition_epochs: int) -> callable:
    """Return an Optuna objective closure capturing the fixed dataset."""

    measurements_train = data["measurements_train"]
    measurements_val   = data["measurements_val"]
    measurements_test  = data["measurements_test"]

    def objective(trial: optuna.Trial) -> float:

        # ---- Sample hyperparameters ----------------------------------------
        n_layers    = trial.suggest_int("nn_n_layers",  1, 3)
        width       = trial.suggest_categorical("nn_width", [8, 12, 16, 24, 32])
        act_name    = trial.suggest_categorical("activation", list(ACTIVATIONS))
        n_elements  = trial.suggest_int("n_elements", 10, 40, step=10)
        poly_degree = trial.suggest_int("poly_degree", 3, 6)

        # Unique, reproducible network initialisation per trial
        init_seed = trial.number

        hparams = dict(
            nn_n_layers        = n_layers,
            nn_width           = width,
            activation         = act_name,
            n_elements         = n_elements,
            poly_degree        = poly_degree,
            init_seed          = init_seed,
        )

        print(f"\n[Trial {trial.number}] {hparams}")

        # ---- Build model ---------------------------------------------------
        pinn = build_pinn(
            n_layers, width, act_name,
            data["phi_ref"], data["scaling"],
            init_seed=init_seed,
        )

        # ---- LR schedule (matches vangenuchten.py) ---------------------------------
        lr_schedule = optax.exponential_decay(
            init_value       = LR_INIT,
            transition_steps = LR_STEPS,
            decay_rate       = LR_END / LR_INIT,
            end_value        = LR_END,
        )

        # ---- Trainer -------------------------------------------------------
        trainer = VPINNTrainer(
            pinn               = pinn,
            measurements_train = measurements_train,
            measurements_val   = measurements_val,   # None disables early stopping
            n_params           = N_PARAMS,
            lambda_data        = 1.0,
            lr_schedule_fn     = lr_schedule,
            n_elements         = n_elements,
            poly_degree        = poly_degree,
            n_quad             = N_QUAD,
            use_lbfgs          = False,
        )

        # ---- Train ---------------------------------------------------------
        t0 = time.perf_counter()
        try:
            trained_pinn, history = trainer.fit(
                epochs=MAX_EPOCHS,
                warmup_epochs=warmup_epochs,
                transition_epochs=transition_epochs,
            )
        except Exception as exc:
            warnings.warn(f"Trial {trial.number} failed: {exc}")
            _save_result(results_path, trial.number, hparams, {}, failed=True)
            return float("inf")

        elapsed = time.perf_counter() - t0

        # ---- Metrics -------------------------------------------------------
        final_total   = float(np.mean(history[0][-10_000:]))
        final_data    = float(np.mean(history[1][-10_000:]))
        final_physics = float(np.mean(history[2][-10_000:]))

        if np.isnan(final_physics) or np.isnan(final_data):
            warnings.warn(f"Trial {trial.number}: NaN in history, diverged during training.")
            _save_result(results_path, trial.number, hparams,
                        dict(elapsed_s=elapsed), failed=True)
            return float("inf")

        # Count trainable diffusivity parameters for chi2_nu degrees of freedom
        diff = trained_pinn.pde.diffusivity
        n_trainable_params = sum(
            1 for name in diff.__dataclass_fields__
            if hasattr(getattr(diff, name), "hat")
        )

        chi2_train = float(
            measurements_train.chi2_nu(trained_pinn.nn_model, n_params=n_trainable_params)
        )
        chi2_test = float(
            measurements_test.chi2_nu(trained_pinn.nn_model, n_params=n_trainable_params)
        )

        # Single objective: one-sided chi2_test
        # Values <= 1 are acceptable (physics acts as a regulariser, some underfitting is fine)
        # Values >> 1 indicate the model fails to fit the data
        gap          = max(0.0, chi2_test - chi2_train)
        relative_gap = min(gap / (chi2_train + 1e-8), 10.0)
        obj = max(0.0, chi2_test - 1.0) + (
            chi2_test * relative_gap if chi2_test > 1.0 else 0.0
        )

        # Recovered parameter values and relative errors
        recovered = {
            name: float(getattr(diff, name).value())
            for name in TRUE_PARAMS
            if hasattr(getattr(diff, name), "hat")
        }
        param_errors = {
            f"err_{name}": abs(recovered[name] - TRUE_PARAMS[name]) / abs(TRUE_PARAMS[name])
            for name in recovered
        }

        metrics = dict(
            chi2_nu_train = chi2_train,
            chi2_nu_test  = chi2_test,
            obj           = obj,
            final_total   = final_total,
            final_data    = final_data,
            final_physics = final_physics,
            elapsed_s     = elapsed,
            **recovered,
            **param_errors,
        )

        # Hard constraint: prune trials where physics never converged.
        # Still compute and save all diagnostic metrics, but mark the trial pruned.
        if final_physics > PHYSICS_THRESHOLD:
            _save_result(results_path, trial.number, hparams, metrics,
                         failed=False, pruned=True)
            raise optuna.TrialPruned()

        print(
            f"  chi2_nu_train={chi2_train:.3f}  chi2_nu_test={chi2_test:.3f}"
            f"  obj={obj:.3f}"
            f"  data={final_data:.3e}  phys={final_physics:.3e}  t={elapsed:.0f}s"
        )
        for name in recovered:
            print(
                f"    {name}: {recovered[name]:.4g}  "
                f"(true={TRUE_PARAMS[name]:.4g}, "
                f"err={param_errors[f'err_{name}']:.1%})"
            )

        _save_result(results_path, trial.number, hparams, metrics, failed=False)
        return obj

    return objective


# ---------------------------------------------------------------------------
# Incremental results saving
# ---------------------------------------------------------------------------

def _save_result(path, trial_number, hparams, metrics, failed, pruned=False):
    """Append one trial result as a JSON line."""
    record = dict(trial=trial_number, failed=failed, pruned=pruned, **hparams, **metrics)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="hp-VPINN hyperparameter search — van Genuchten case"
    )
    parser.add_argument("--data",        default="./data/vangenuchten/data_vangenuchten.dat")
    parser.add_argument("--n-trials",    type=int, default=50)
    parser.add_argument("--results-dir", default="./results/results_vangenuchten")
    parser.add_argument("--study-name",  default="vpinn_hparam_search_vangenuchten")
    parser.add_argument("--method",      default="gb", choices=["gb", "es"])
    args = parser.parse_args()

    results_dir = args.results_dir
    if args.method == "es":
        results_dir += "_es"
        warmup_epochs = MAX_EPOCHS
        transition_epochs = 0
    elif args.method == "gb":
        results_dir += "_gb"
        warmup_epochs = 0
        transition_epochs = 20_000
    else:
        raise ValueError(f"Unknown method: {args.method}")

    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "hparam_results_vangenuchten.jsonl")
    db_path      = os.path.join(results_dir, "study.db")

    print("=" * 60)
    print("hp-VPINN hyperparameter search  —  van Genuchten case")
    print(f"  Trials:        {args.n_trials}")
    print(f"  Test fraction: {TEST_FRACTION:.0%}")
    print(f"  Results:       {results_path}")
    print(f"  Optuna DB:     {db_path}")
    print(f"  Dashboard:     optuna-dashboard sqlite:///{db_path}")
    print("=" * 60)

    data = prepare_data(args.data)

    storage = f"sqlite:///{db_path}"
    sampler = TPESampler(
        seed             = 0,
        n_startup_trials = 15,
        multivariate     = True,
        group            = True,
    )
    study = optuna.create_study(
        study_name     = args.study_name,
        storage        = storage,
        sampler        = sampler,
        direction      = "minimize",
        load_if_exists = True,
    )

    objective = make_objective(data, results_path, warmup_epochs, transition_epochs)
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1)

    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete_trials:
        print("\nNo complete trials were found in the study.")
        print("Study trial counts:")
        states = collections.Counter(t.state for t in study.trials)
        for state, count in states.items():
            print(f"  {state}: {count}")
        print("Nothing to report for best_trial.")
        return

    best = study.best_trial
    print("\n" + "=" * 60)
    print("Best trial")
    print("=" * 60)
    print(f"  Objective : {best.value:.4f}")
    print(f"  Params    : {best.params}")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
hparam_search_grenoblesand.py
==============================
Optuna-based hyperparameter search for the hp-VPINN inverse problem
on the Grenoble sand benchmark (Case 1).

Usage
-----
    uv run python experiments/grenoblesand/hparam_search_grenoblesand.py \
        --noise-level 0.01 --config gb --n-trials 250

Arguments
---------
    --noise-level  : noise std dev as fraction of moisture content range;
                     must be one of the values defined in NOISE_LEVELS.
                     Required.
    --config       : weighting strategy: 'gb' (gradient balancing, default)
                     or 'es' (exponential schedule).
    --n-trials     : number of Optuna trials (default: 50).
    --results-dir  : output directory (default: ./results/results_grenoblesand).
    --study-name   : Optuna study name (default: vpinn_hparam_search).
    --data         : path to the dataset file.

Search space
------------
    nn_n_layers  {1, 2, 3}           nn_width     {8, 16, 32}
    activation   {tanh, sigmoid,      n_elements   {10, 20, 30, 40, 50}
                  GeLU, Swish}        poly_degree  {3, 4, 5, 6}

Results
-------
Results are appended trial-by-trial to <results-dir>/hparam_results.jsonl
and persisted to <results-dir>/study.db (viewable with optuna-dashboard).
"""

import argparse
from ast import arg
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
from hpvpinn.constitutive_model import BoundedParam, FixedParam, LowerBoundedParam, VanGenuchtenDiffusivity
from hpvpinn.transformations    import CapillarityScaling
from hpvpinn.datasets           import MeasurementSet
from hpvpinn.train_vpinn        import VPINNTrainer


# ---------------------------------------------------------------------------
# Fixed experimental / training configuration
# ---------------------------------------------------------------------------

# True parameters (used only for reporting recovery error, not for training)
TRUE_PARAMS = dict(
    KsAlpha = 15.37 / 0.0432,
    m       = 0.5096,
)

# Diffusivity parameters held FIXED throughout all trials
L_FIXED = 0.5
THETA_R = 0.0
THETA_S = 0.312

# Training budget
MAX_EPOCHS = 200_000

# Fixed LR schedule parameters
LR_INIT  = 1e-3
LR_END   = 1e-4
LR_STEPS = 100_000

# hp-VPINN quadrature degree (fixed, not a search hyperparameter): 
N_QUAD = 8

# Random seeds (fixed for reproducibility across trials)
# SEED_SPLIT : seed for the train/test index permutation
SEED_SPLIT = 7

# Fraction of points held out as a validation set
VAL_FRACTION = 0.0

# Fraction of points held out as a test set
TEST_FRACTION = 0.20

# Valid noise levels for --noise-level argument
NOISE_LEVELS = (0.01, 0.02)

# Physics loss threshold below which a trial is considered converged
PHYSICS_THRESHOLD = 1e-5

# Number of trainable diffusivity parameters (KsAlpha and m)
N_PARAMS = 2  


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
# Data preparation (done once, shared across all trials)
# ---------------------------------------------------------------------------

def prepare_data(data_path: str, noise_level: float) -> dict:
    """Load frontx data, apply scaling, and pre-generate one fixed noisy
    training dataset and one fixed noisy test dataset per noise level.

    The train/test index split is the same for every noise level (same
    SEED_SPLIT), so all strata cover the same phi locations.  Each stratum
    has its own independent noise seeds for train and test, so noise
    realisations never leak between strata or between train and test.

    Returns a dict with:
      scaling          : CapillarityScaling instance
      phi_ref          : float
      measurements_all : clean (noise-free) MeasurementSet
      noisy_datasets   : {sigma -> noisy train MeasurementSet}
      test_datasets    : {sigma -> noisy test  MeasurementSet}
    """
    data      = np.loadtxt(data_path, skiprows=1).astype(np.float32)
    phi_raw   = data[:, 0]
    theta_raw = data[:, 1]

    phi_ref = 1.05 * float(phi_raw[-1])
    scaling = CapillarityScaling(
        phi_ref = phi_ref,
        theta_i = 0.0,
        theta_b = float(THETA_S) - 1e-7,
    )

    phi_nd   = np.array(scaling.phi_forward(jnp.array(phi_raw)))
    theta_nd = np.array(scaling.theta_forward(jnp.array(theta_raw)))
    measurements_all = MeasurementSet(
        phi_nd, 
        theta_nd,
        sigma=noise_level,
        true_value=True,
    )
    measurements_train, measurements_val, measurements_test = measurements_all.split(
        val_fraction  = 0.,
        test_fraction = TEST_FRACTION,
        key           = jax.random.PRNGKey(SEED_SPLIT),
    )

    return dict(
        scaling         = scaling,
        phi_ref         = phi_ref,
        noisy_dataset   = measurements_train,
        val_dataset     = measurements_val,
        test_dataset    = measurements_test,
    )


# ---------------------------------------------------------------------------
# Build a fresh PINN for each trial
# ---------------------------------------------------------------------------

def build_full_pinn(n_layers: int, width: int, activation_name: str,
                    phi_ref: float, scaling, init_seed: int) -> PINN:
    """Construct a fresh PINN with the given architecture and random seed."""
    diffusivity = VanGenuchtenDiffusivity(
        KsAlpha = LowerBoundedParam(value=1.0, min=0.0),
        m       = BoundedParam(value=0.5, min=0.0, max=1.0),
        l       = FixedParam(value=L_FIXED),
        theta_r = FixedParam(value=THETA_R),
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

def make_objective(data: dict, results_path: str, noise_level: float, warmup_epochs: int, transition_epochs: int) -> callable:
    """Return an Optuna objective closure that captures the fixed data."""

    def objective(trial: optuna.Trial) -> float:

        # ---- Sample hyperparameters ----
        # Note: n_elements uses suggest_int (not suggest_categorical) so that
        # TPE can exploit the natural ordering of the grid density values.
        n_layers    = trial.suggest_int("nn_n_layers", 1, 3)
        width       = trial.suggest_categorical("nn_width", [8, 12, 16, 24, 32])
        act_name    = trial.suggest_categorical("activation", list(ACTIVATIONS.keys()))
        n_elements  = trial.suggest_int("n_elements", 10, 40, step=10)
        poly_degree = trial.suggest_int("poly_degree", 3, 6)

        # Derive the network init seed from the trial number.
        # This gives each trial a unique, reproducible initialisation without
        # exposing seed as a degree of freedom for the TPE sampler to exploit.
        init_seed = trial.number

        hparams = dict(
            nn_n_layers  = n_layers,
            nn_width     = width,
            activation   = act_name,
            n_elements   = n_elements,
            poly_degree  = poly_degree,
            noise_level  = noise_level,
            init_seed    = init_seed,
        )

        print(f"\n[Trial {trial.number}] {hparams}")

        # ---- Select pre-built datasets ----
        measurements_train = data["noisy_dataset"]
        measurements_val   = data["val_dataset"]
        measurements_test  = data["test_dataset"]

        # ---- Build model ----
        pinn = build_full_pinn(
            n_layers, width, act_name,
            data["phi_ref"], data["scaling"],
            init_seed=init_seed,
        )

        # ---- LR schedule ----
        lr_schedule = optax.exponential_decay(
            init_value       = LR_INIT,
            transition_steps = LR_STEPS,
            decay_rate       = LR_END / LR_INIT,
            end_value        = LR_END,
        )

        # ---- Trainer ----
        # measurements_val=None disables early stopping; the has_validation=False
        # branch in _train_vpinn is compiled statically at JIT time.
        trainer = VPINNTrainer(
            pinn               = pinn,
            measurements_train = measurements_train,
            measurements_val   = measurements_val,
            n_params           = N_PARAMS,
            lambda_data        = 1.0,
            lr_schedule_fn     = lr_schedule,
            n_elements         = n_elements,
            poly_degree        = poly_degree,
            n_quad             = N_QUAD,
            use_lbfgs          = False,
        )

        # ---- Train ----
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

        # ---- Metrics ----
        final_total   = float(np.mean(history[0][-10_000:]))
        final_data    = float(np.mean(history[1][-10_000:]))
        final_physics = float(np.mean(history[2][-10_000:]))

        if np.isnan(final_physics) or np.isnan(final_data):
            warnings.warn(f"Trial {trial.number}: NaN in history, diverged during training.")
            _save_result(results_path, trial.number, hparams,
                        dict(elapsed_s=elapsed), failed=True)
            return float("inf")

        # Count trainable diffusivity parameters dynamically for chi2_nu d.o.f.
        diff = trained_pinn.pde.diffusivity
        n_trainable_params = sum(
            1 for name in diff.__dataclass_fields__
            if hasattr(getattr(diff, name), "hat")
        )

        # chi2_nu on training set: diagnostic for training-set fit quality.
        chi2_train = float(
            measurements_train.chi2_nu(trained_pinn.nn_model, n_params=n_trainable_params)
        )
        # chi2_nu on test set: primary model-selection metric.
        chi2_test = float(
            measurements_test.chi2_nu(trained_pinn.nn_model, n_params=n_trainable_params)
        )

        # Single objective: penalise underfitting and overfitting jointly.
        # The first term is one-sided: chi2_nu_test <= 1 is acceptable
        # (physics acts as a regulariser so some underfitting is fine).
        # The second term penalises overfitting via the relative train/test gap.
        gap          = max(0.0, chi2_test - chi2_train)
        relative_gap = min(gap / (chi2_train + 1e-8), 10.0)
        obj = max(0.0, chi2_test - 1.0) + (
            chi2_test * relative_gap if chi2_test > 1.0 else 0.0
        )

        # Recovered parameters and relative errors — saved as diagnostics,
        # NOT used as Optuna objectives.
        recovered = {
            name: float(getattr(diff, name).value())
            for name in TRUE_PARAMS
        }
        param_errors = {
            f"err_{name}": abs(recovered[name] - TRUE_PARAMS[name]) / abs(TRUE_PARAMS[name])
            for name in TRUE_PARAMS
        }
        obj_recovery = float(np.mean(list(param_errors.values())))

        metrics = dict(
            chi2_nu_train = chi2_train,
            chi2_nu_test  = chi2_test,
            obj           = obj,
            obj_recovery  = obj_recovery,   # diagnostic only
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
            f"  obj={obj:.3f}  obj_recovery={obj_recovery:.3f}"
            f"  data={final_data:.3e}  phys={final_physics:.3e}  t={elapsed:.0f}s"
        )
        for name in TRUE_PARAMS:
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
    """Append one trial result as a JSON line to the results file."""
    record = dict(trial=trial_number, failed=failed, pruned=pruned, **hparams, **metrics)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # String-based noise level validation avoids float == comparison fragility.
    valid_noise_strs = [str(s) for s in NOISE_LEVELS]

    parser = argparse.ArgumentParser(description="hp-VPINN hyperparameter search")
    parser.add_argument("--data",              default="./data/grenoblesand/data_grenoblesand.dat")
    parser.add_argument("--n-trials",          type=int, default=50)
    parser.add_argument("--results-dir",       default="./results/results_grenoblesand")
    parser.add_argument("--study-name",        default="vpinn_hparam_search")
    parser.add_argument("--config",            default="gb", choices=["gb", "es"])
    parser.add_argument(
        "--noise-level",
        type=str,
        required=True,
        choices=valid_noise_strs,
        metavar=f"{{{','.join(valid_noise_strs)}}}",
        help=f"Measurement noise std dev. Must be one of: {valid_noise_strs}",
    )
    args = parser.parse_args()

    noise_level = float(args.noise_level)
    study_name  = f"{args.study_name}_sigma{int(noise_level * 1000):04d}"

    base_results_dir = args.results_dir
    if args.config == "es":
        base_results_dir += "_es"
        warmup_epochs = MAX_EPOCHS
        transition_epochs = 0
    elif args.config == "gb":
        base_results_dir += "_gb"
        warmup_epochs = 0
        transition_epochs = 20_000
    else:
        raise ValueError(f"Unknown config: {args.config}")

    results_dir = f"{base_results_dir}_{int(noise_level * 100):01d}pct_noise"
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(
        results_dir, f"hparam_results_{int(noise_level * 100):01d}pct_noise.jsonl"
    )
    db_path = os.path.join(results_dir, "study.db")

    print("=" * 60)
    print("hp-VPINN hyperparameter search")
    print(f"  Noise level:   sigma={noise_level}")
    print(f"  Trials:        {args.n_trials}")
    print(f"  Test fraction: {TEST_FRACTION:.0%}")
    print(f"  Results:       {results_path}")
    print(f"  Optuna DB:     {db_path}")
    print(f"  Dashboard:     optuna-dashboard sqlite:///{db_path}")
    print("=" * 60)

    noise_level /= THETA_S  # convert to relative noise level for reporting and data generation

    data = prepare_data(args.data, noise_level)

    storage = f"sqlite:///{db_path}"
    sampler = TPESampler(
        seed             = 0,
        n_startup_trials = 25,
        multivariate     = True,
        group            = True,
    )
    study = optuna.create_study(
        direction      = "minimize",
        study_name     = args.study_name,
        storage        = storage,
        sampler        = sampler,
        load_if_exists = True,
    )

    objective = make_objective(data, results_path, noise_level, warmup_epochs, transition_epochs)
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

"""
grenoblesand_trial236.py
========================
Driver script reproducing the best-performing \hpVPINN-gb result for
Case~1 (Grenoble sand, 2% noise) reported in Tourn et al. (2025).
Hyperparameter configuration: 2 hidden layers, 16 units per layer,
tanh activation, N_el = 40 elements, polynomial degree p = 6,
configuration type "-gb", and initialisation seed 236.

Usage
-----
    uv run python experiments/grenoblesand/grenoblesand_trial236.py

Outputs
-------
- Terminal: recovered KsAlpha and m, chi2_nu on train and test sets,
  relative L2 error on the full dataset, and elapsed training time.
- Figures: training loss curves with parameter trajectories, and the
  recovered moisture content profile overlaid on the observations.

Data
----
Reads ``data/grenoblesand/data_grenoblesand.dat`` (phi, theta columns,
one header line). The dataset is included in the repository.
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import time

from hpvpinn.nn_model import FeedForwardNN
from hpvpinn.pde import CapillarityBoltzmannPDE
from hpvpinn.pinn_model import PINN
from hpvpinn.constitutive_model import FixedParam, LowerBoundedParam, BoundedParam, VanGenuchtenDiffusivity
from hpvpinn.transformations import CapillarityScaling
from hpvpinn.datasets import MeasurementSet
from hpvpinn.train_vpinn import VPINNTrainer
from hpvpinn.plot_utils import plot_training_losses, plot_profile


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    # ---- Van Genuchten reference parameters (Grenoble sand) ----
    Ks_true   = 15.37        # cm/h
    alpha_vg  = 0.0432       # 1/cm
    m_true    = 0.5096
    l_true    = 0.5
    theta_s   = 0.312
    theta_r   = 0.

    # ---- Load dataset ----
    data_path = "./data/grenoblesand/data_grenoblesand.dat"
    data = np.loadtxt(data_path, skiprows=1).astype(np.float32)
    phi_raw   = data[:, 0]
    theta_raw = data[:, 1]

    # ---- Scaling ----
    phi_ref = 1.05 * phi_raw[-1]
    scaling = CapillarityScaling(
        phi_ref = phi_ref,
        theta_i = 0.0,
        theta_b = theta_s - 1e-7,
    )

    phi_nd   = scaling.phi_forward(jnp.array(phi_raw))
    theta_nd = scaling.theta_forward(jnp.array(theta_raw))

    # ---- Measurements ----
    sigma = 0.064  # Corresponds to 2% noise in theta
    measurements_all = MeasurementSet(
        phi_nd,
        theta_nd,
        sigma=sigma,
        true_value=True,
    )

    # Split clean data into train / val / test index sets.
    # Set val_fraction=0.0 or test_fraction=0.0 to disable either split.
    measurements_train, measurements_val, measurements_test = measurements_all.split(
        val_fraction  = 0.,
        test_fraction = 0.2,
        key           = jax.random.PRNGKey(7),
    )

    # ---- Diffusivity model (Ks and m are trainable) ----
    KsAlpha_true = Ks_true / alpha_vg   # recoverable compound parameter

    diffusivity = VanGenuchtenDiffusivity(
        KsAlpha = LowerBoundedParam(value=1., min=0.0),
        m       = BoundedParam(value=0.5, min=0.0, max=1.0),
        l       = FixedParam(value=l_true),
        theta_r = FixedParam(value=theta_r),
        theta_s = theta_s,
    )

    # ---- PDE container ----
    pde = CapillarityBoltzmannPDE(
        diffusivity = diffusivity,
        phi_ref     = phi_ref,
        scaling     = scaling,
    )

    # ---- Neural network ----
    nn_model = FeedForwardNN(
        width=16, 
        depth=2,
        activation=jax.nn.tanh,
        key=jax.random.PRNGKey(236),
    )

    # ---- PINN ----
    pinn = PINN(nn_model=nn_model, pde=pde)

    # ---- Learning rate schedule ----
    lr_schedule = optax.exponential_decay(
        init_value       = 1e-3,
        transition_steps = 100_000,
        decay_rate       = 0.1,
        end_value        = 1e-4,
    )

    # ---- Trainer ----
    trainer = VPINNTrainer(
        pinn                = pinn,
        measurements_train  = measurements_train,
        measurements_val    = measurements_val,
        n_params            = 2,  # KsAlpha and m
        lambda_data         = 1.0,
        lr_schedule_fn      = lr_schedule,
        # hp-VPINN grid
        n_elements          = 40,
        poly_degree         = 6,
        n_quad              = 8,
        element_clustering  = 1.0,
        # optional L-BFGS refinement
        use_lbfgs           = False,
        lbfgs_maxiter       = 5_000,
        l2_reg              = False,
    )

    # ---- Training ----
    max_epochs = 200_000

    print("\n" + "=" * 60)
    print("Starting hp-VPINN training ...")
    print("=" * 60)
    t0 = time.perf_counter()

    trained_pinn, history = trainer.fit(
        epochs = max_epochs,
    )

    elapsed = time.perf_counter() - t0
    print("=" * 60)
    print(f"Training complete.  Elapsed: {elapsed:.1f} s  ({elapsed/60:.2f} min)")
    print("=" * 60 + "\n")

    # ---- Recovered parameters ----
    diff        = trained_pinn.pde.diffusivity
    KsAlpha_rec = float(diff.KsAlpha.value())
    m_rec       = float(diff.m.value())
    print(f"  True  KsAlpha = {KsAlpha_true:.4g} cm2/h    "
          f"Recovered KsAlpha = {KsAlpha_rec:.4g} cm2/h")
    print(f"  True  m       = {m_true:.4f}              "
          f"Recovered m       = {m_rec:.4f}")

    # ---- Goodness-of-fit metrics ----
    if measurements_train._sigma_known:
        # chi2_nu on the training set (noisy observations, same realisation).
        chi2_train = float(measurements_train.chi2_nu(trained_pinn.nn_model, n_params=2))
        print(f"\n  chi2_nu (train) = {chi2_train:.4f}  (target ≈ 1 for pure data fit;"
            f" physics regularisation typically yields values slightly below 1)")
    else:
        mse_train = float(measurements_train.mse(trained_pinn.nn_model))
        print(f"\n  MSE (train) = {mse_train:.4e}  (sigma unknown → chi2_nu undefined)")

    if measurements_test is not None:
        if measurements_test._sigma_known:
            # chi2_nu on the held-out test set (independent noise realisation).
            # This is the cleanest model-selection metric: the model never saw these
            # points during training or early-stopping, and the noise is independent.
            chi2_test = float(measurements_test.chi2_nu(trained_pinn.nn_model, n_params=2))
            print(f"  chi2_nu (test)  = {chi2_test:.4f}  "
                  f"({measurements_test.phi.shape[0]} held-out points, independent noise)")
        else:
            mse_test = float(measurements_test.mse(trained_pinn.nn_model))
            print(f"  MSE (test) = {mse_test:.4e}  (sigma unknown → chi2_nu undefined)")

    # ---- L2 error ----
    l2_err = measurements_all.relative_l2(trained_pinn.nn_model)
    print(f"\n  Relative L2 error on full dataset: {float(l2_err):.3e}")

    # ---- Plots ----
    plot_training_losses(history, logscale=True, true_params=dict(KsAlpha=KsAlpha_true, m=m_true))
    plot_profile(trained_pinn, measurements_train)


if __name__ == "__main__":
    main()

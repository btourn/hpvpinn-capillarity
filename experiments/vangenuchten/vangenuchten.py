"""
vangenuchten.py
===============
Driver script for the hp-VPINN inverse problem on the van Genuchten dataset.

Pipeline
--------
1. Load the synthetic Van Genuchten dataset.
2. Normalise phi and theta.
3. Optionally add measurement noise.
4. Build the PINN (network + physical model with trainable Ks and m).
5. Train with VPINNTrainer (Adam + optional L-BFGS).
6. Report and plot results.
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
from hpvpinn.constitutive_model import FixedParam, FreeParam, LowerBoundedParam, BoundedParam, VanGenuchtenDiffusivity
from hpvpinn.transformations import CapillarityScaling
from hpvpinn.datasets import MeasurementSet
from hpvpinn.train_vpinn import VPINNTrainer
from hpvpinn.plot_utils import *


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    # ---- Van Genuchten reference parameters ----
    KsAlpha_true = 2.079e-6 #m2/s   #74.84  #cm2/h
    m_true       = 0.8764
    l_true       = 2.344
    theta_s_true = 0.7
    theta_r_true = 0.004943
    
    theta_i      = 0.025
    theta_b      = theta_s_true - 1e-7

    # ---- Load dataset ----
    data_path = "./data/vangenuchten/data_vangenuchten.dat"
    data = np.loadtxt(data_path, skiprows=1).astype(np.float32)
    phi_raw   = data[:, 0]  #m / s^0.5
    theta_raw = data[:, 1]
    std_raw   = data[:, 2]

    # ---- Scaling ----
    phi_ref = 1.05 * phi_raw[-1]
    scaling = CapillarityScaling(
        phi_ref = phi_ref,
        theta_i = theta_i,
        theta_b = theta_b,
    )

    phi_nd   = scaling.phi_forward(jnp.array(phi_raw))
    theta_nd = scaling.theta_forward(jnp.array(theta_raw))
    sigma_nd = scaling.sigma_forward(jnp.array(std_raw))

    # ---- Measurements ----
    measurements_all = MeasurementSet(
        phi=phi_nd, 
        theta=theta_nd, 
        sigma=sigma_nd, 
        true_value=False
    )

    # Split clean data into train / val / test index sets.
    # Set val_fraction=0.0 or test_fraction=0.0 to disable either split.
    val_fraction  = 0.
    test_fraction = 0.2
    measurements_train, measurements_val, measurements_test = measurements_all.split(
        val_fraction  = val_fraction,
        test_fraction = test_fraction,
        key           = jax.random.PRNGKey(456),
    )

    diffusivity = VanGenuchtenDiffusivity(
        KsAlpha = LowerBoundedParam(value=1.e-4, min=0.0),
        m       = BoundedParam(value=0.5, min=0.0, max=1.0),
        l       = FreeParam(value=1.),
        theta_r = BoundedParam(value=0.02, min=0.0, max=theta_i),
        theta_s = theta_s_true,
    )

    # ---- PDE container ----
    pde = CapillarityBoltzmannPDE(
        diffusivity = diffusivity,
        phi_ref     = phi_ref,
        scaling     = scaling,
    )

    # ---- Neural network ----
    nn_model = FeedForwardNN(
        width=32, 
        depth=3,
        activation=jax.nn.tanh,
        key=jax.random.PRNGKey(182)
    )

    # ---- PINN ----
    pinn = PINN(nn_model=nn_model, pde=pde)

    # ---- Learning rate schedule ----
    lr_schedule = optax.exponential_decay(
        init_value       = 1e-3, # 1e-1,
        transition_steps = 100_000, # 10_000,
        decay_rate       = 0.1, # 0.1,
        end_value        = 1e-4, # 1e-3,
    )

    # ---- Trainer ----
    trainer = VPINNTrainer(
        pinn                = pinn,
        measurements_train  = measurements_train,
        measurements_val    = measurements_val,
        n_params            = 4, # KsAlpha, m, l, theta_r
        lambda_data         = 1.0,
        lr_schedule_fn      = lr_schedule,
        # hp-VPINN grid
        n_elements          = 10,
        poly_degree         = 5,
        n_quad              = 8,
        element_clustering  = 1.0,
        # optional L-BFGS refinement
        use_lbfgs           = False,
        lbfgs_maxiter       = 1_000,
        l2_reg              = False,
        weight_decay        = 0.1,
    )

    # ---- Training ----
    max_epochs = 200_000

    print("\n" + "=" * 60)
    print("Starting hp-VPINN training ...")
    print("=" * 60)
    t0 = time.perf_counter()

    trained_pinn, history = trainer.fit(
        epochs = max_epochs,
        warmup_epochs=max_epochs
    )

    elapsed = time.perf_counter() - t0
    print("=" * 60)
    print(f"Training complete.  Elapsed: {elapsed:.1f} s  ({elapsed/60:.2f} min)")
    print("=" * 60 + "\n")

    # ---- Recovered parameters ----
    diff        = trained_pinn.pde.diffusivity
    trainable   = {
        name: getattr(diff, name)
        for name in diff.__dataclass_fields__
        if hasattr(getattr(diff, name), "hat")
    }
    n_params    = len(trainable)
    KsAlpha_rec = float(diff.KsAlpha.value())
    m_rec       = float(diff.m.value())
    l_rec       = float(diff.l.value())
    theta_r_rec = float(diff.theta_r.value())
    print(f"  True  KsAlpha = {KsAlpha_true:.4g} cm2/h    "
          f"Recovered KsAlpha = {KsAlpha_rec:.4g} cm2/h")
    print(f"  True  m       = {m_true:.4f}              "
          f"Recovered m       = {m_rec:.4f}")
    print(f"  True  l       = {l_true:.4f}              "
          f"Recovered l       = {l_rec:.4f}")
    print(f"  True  theta_r = {theta_r_true:.4f}          "
          f"Recovered theta_r = {theta_r_rec:.4f}")

    # ---- Goodness-of-fit metrics ----
    # chi2_nu on the training set (noisy observations, same realisation).
    chi2_train = float(measurements_train.chi2_nu(trained_pinn.nn_model, n_params=n_params))
    print(f"\n  chi2_nu (train) = {chi2_train:.4f}  (target ≈ 1 for pure data fit;"
          f" physics regularisation typically yields values slightly below 1)")

    # chi2_nu on the held-out test set (independent noise realisation).
    # This is the cleanest model-selection metric: the model never saw these
    # points during training or early-stopping, and the noise is independent.
    if measurements_test is not None:
        chi2_test = float(measurements_test.chi2_nu(trained_pinn.nn_model, n_params=n_params))
        print(f"  chi2_nu (test)  = {chi2_test:.4f}  "
              f"({measurements_test.phi.shape[0]} held-out points, independent noise)")

    # ---- Plots ----
    plot_training_losses(history, logscale=True, true_params=dict(KsAlpha=KsAlpha_true, m=m_true, l=l_true, theta_r=theta_r_true))
    plot_profile(trained_pinn, measurements_train)


if __name__ == "__main__":
    main()

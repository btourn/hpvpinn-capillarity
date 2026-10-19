"""
plot_utils.py
=============
Shared plotting utilities for hp-VPINN driver scripts.

Functions
---------
plot_training_losses(history, logscale=True, true_params=None)
    Plot scalar loss curves and trainable diffusivity parameter trajectories
    from the history tuple returned by VPINNTrainer.fit().

plot_profile(trained_pinn, measurements, phi_range=(0.0, 1.5), n_points=1000)
    Plot the predicted moisture-content profile against a MeasurementSet.
"""

import shutil
import matplotlib
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

_use_latex = shutil.which("latex") is not None

matplotlib.rcParams.update({
    "text.usetex":            _use_latex,
    "text.latex.preamble":    r"\usepackage{amsbsy}",
    "font.family":            "serif",
    "font.serif":             ["Computer Modern Roman"],
    "axes.labelsize":         14,
    "xtick.labelsize":        11,
    "ytick.labelsize":        11,
    "legend.fontsize":        10,
    "figure.dpi":             300,
})


def plot_training_losses(history, logscale=True, true_params=None):
    """Plot loss curves and trainable parameter trajectories from history.

    Parameters
    ----------
    history : train_vpinn.TrainingHistory
        The NamedTuple returned by VPINNTrainer.fit(). Fields are accessed
        by name (history.total_loss, history.pde_loss, etc.) rather than by
        position, so this function is agnostic to field order/additions.
    logscale : bool
        Use semilogy / loglog axes.  Default True.
    true_params : dict or None
        Optional mapping {param_name: true_value} drawn as horizontal
        reference lines on the parameter trajectory plots.
    """
    loss_labels = [
        "Train+Val loss", "Data loss", "Physics loss",
        "PDE loss", "Dirichlet loss", "Derivative loss",
        "lambda_phys",
    ]
    loss_colors = [
        "k", "tab:blue", "tab:orange",
        "tab:green", "tab:red", "tab:brown",
        "tab:purple",
    ]

    epochs      = np.arange(len(history.total_loss))
    train_loss  = np.array(history.total_loss)
    data_loss   = np.array(history.data_loss)
    phys_loss   = np.array(history.physics_loss)
    val_loss    = np.array(history.val_loss)
    pde_loss    = np.array(history.pde_loss)
    dirichlet_loss  = np.array(history.dirichlet_loss)
    derivative_loss = np.array(history.derivative_loss)
    lambda_phys = np.array(history.lambda_phys)

    list_of_losses = [
        None, data_loss, phys_loss,
        pde_loss, dirichlet_loss, derivative_loss,
        lambda_phys,
    ]

    n_panels = len(loss_labels)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4))
    for ax, label, color, idx in zip(axes, loss_labels, loss_colors, range(n_panels)):
        if idx == 0:
            if logscale:
                ax.semilogy(epochs, train_loss, color="k",         linewidth=1.5, label="Train total")
                ax.semilogy(epochs, val_loss,   color="tab:green", linewidth=1.5, label="Val")
            else:
                ax.plot(epochs, train_loss, color="k",         linewidth=1.5, label="Train total")
                ax.plot(epochs, val_loss,   color="tab:green", linewidth=1.5, label="Val")
            ax.legend(fontsize=8)
        else:
            values = list_of_losses[idx]
            if logscale:
                ax.semilogy(epochs, values, color=color, linewidth=1.5)
            else:
                ax.plot(epochs, values, color=color, linewidth=1.5)

        ax.set_xlabel("Epoch")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ---- Trainable parameter trajectories --------------------------------
    diff_hist = history.diffusivity

    def _to_physical(param):
        """Reconstruct physical value(s) from a (possibly stacked) param object."""
        if hasattr(param, "min") and hasattr(param, "max"):   # BoundedParam
            return param.min + (param.max - param.min) * jax.nn.sigmoid(param.hat)
        elif hasattr(param, "min"):                            # LowerBoundedParam
            return param.min + jnp.exp(param.hat)
        elif hasattr(param, "max"):                            # UpperBoundedParam
            return param.max - jnp.exp(param.hat)
        return param.hat                                       # FreeParam

    trainable = {
        name: _to_physical(getattr(diff_hist, name))
        for name in diff_hist.__dataclass_fields__
        if hasattr(getattr(diff_hist, name), "hat")
    }

    if not trainable:
        return

    n = len(trainable)
    fig2, axes2 = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes2 = [axes2]

    for ax, (name, values) in zip(axes2, trainable.items()):
        ax.plot(epochs, np.array(values), linewidth=1.5)
        if true_params is not None and name in true_params:
            ax.axhline(
                true_params[name],
                color="r", linestyle="--", linewidth=1.2,
                label=f"true = {true_params[name]:.4g}",
            )
            ax.legend(fontsize=8)
        ax.set_xlabel("Epoch")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_profile(trained_pinn, measurements, phi_range=(0.0, 1.5), n_points=1000):
    """Plot the predicted moisture-content profile against measurements.

    Parameters
    ----------
    trained_pinn : PINN
        Trained model with a callable ``nn_model`` attribute.
    measurements : MeasurementSet
        Data points to overlay as a scatter plot.
    phi_range : (float, float)
        Range of the normalised Boltzmann variable for the prediction curve.
    n_points : int
        Number of evaluation points for the smooth prediction curve.
    """
    phi_plot   = jnp.linspace(phi_range[0], phi_range[1], n_points)
    theta_pred = jax.vmap(trained_pinn.nn_model)(phi_plot)

    plt.figure(figsize=(8, 5))
    plt.plot(
        np.array(phi_plot), np.array(theta_pred),
        "b-", linewidth=2, label="hp-VPINN prediction",
    )
    plt.scatter(
        np.array(measurements.phi), np.array(measurements.theta),
        c="r", s=20, alpha=0.7, label="Measurements",
    )
    plt.xlabel(r"$\tilde{\phi}$")
    plt.ylabel(r"$\tilde{\theta}$")
    plt.title("Moisture content profile")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

"""Pointwise strong-form residual of the capillarity Boltzmann ODE, evaluated on a dense grid of nodes for a
frozen, converged model.

Post-hoc diagnostic only: it is never used in the training loop. The pointwise strong-form residual and the
element-wise weak-form residual of the hp-VPINN are different quantities, so both are evaluated on the same
dense grid to give a common metric.

Governing ODE (nondimensional phi in [0, 1]):
    -phi/2 * dtheta/dphi = (1/phi_ref^2) * d/dphi( D(theta) * dtheta/dphi )

Zero-residual form:
    r(phi) = -phi/2 * dtheta/dphi - (1/phi_ref^2) * d/dphi( D(theta) * dtheta/dphi )

D(.) takes the dimensional theta: nn_model outputs theta_tilde, which is un-normalised via
scaling.theta_inverse before being passed to pde.diffusivity, as in vpinn_loss.py's _element_residual.
"""

import jax
import jax.numpy as jnp


def strong_form_residual(nn_model, pde, scaling, phi):
    """r(phi) = -(phi/2) dtheta/dphi - (1/phi_ref^2) d/dphi( D(theta) dtheta/dphi ).

    Parameters
    ----------
    nn_model : scalar phi_nd -> scalar theta_nd  (trained network callable)
    pde      : CapillarityBoltzmannPDE  (carries .diffusivity and .phi_ref)
    scaling  : CapillarityScaling  (theta_nd <-> theta_dim conversion)
    phi      : (N,) array of nondimensional phi nodes in [0, 1] (dense grid)

    Returns
    -------
    r : (N,) array of pointwise strong-form residuals
    """

    def theta_tilde_fn(p):
        return nn_model(p)

    dtheta_tilde_dphi_fn = jax.grad(theta_tilde_fn)

    def flux_fn(p):
        theta_tilde = theta_tilde_fn(p)
        theta = scaling.theta_inverse(theta_tilde)
        dtheta_tilde_dphi = dtheta_tilde_dphi_fn(p)
        D = pde.diffusivity(theta)
        return D * dtheta_tilde_dphi

    dflux_dphi_fn = jax.grad(flux_fn)

    def r_fn(p):
        return -0.5 * p * dtheta_tilde_dphi_fn(p) - dflux_dphi_fn(p) / pde.phi_ref**2

    return jax.vmap(r_fn)(phi)


def evaluate_strong_form_rms(trained_pinn, phi_dense) -> float:
    """RMS of the strong-form residual for a frozen, converged model on a
    dense grid. Convenience wrapper around strong_form_residual for use in
    HPO scripts and other post-hoc diagnostics.
    """
    r_vals = strong_form_residual(
        trained_pinn.nn_model, trained_pinn.pde, trained_pinn.pde.scaling, phi_dense
    )
    return float(jnp.sqrt(jnp.mean(r_vals**2)))

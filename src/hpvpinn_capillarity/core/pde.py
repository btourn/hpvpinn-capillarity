"""Physical parameters of the capillarity Boltzmann problem.

CapillarityBoltzmannPDE carries the trainable diffusivity model and the fixed scaling / phi_ref. It is
embedded in the PINN module so that the diffusivity parameters are updated by gradient descent alongside the
network weights. The physics loss is computed in vpinn_loss.py.
"""

import equinox as eqx
import jax.numpy as jnp


class CapillarityBoltzmannPDE(eqx.Module):
    """
    Container for the physical model of the similarity-transformed Richards ODE.

    Attributes
    ----------
    diffusivity : VanGenuchtenDiffusivity (or any callable theta_dim -> D)
                  The constitutive model whose parameters are learnable.
    phi_ref     : float
                  Dimensional reference value of the Boltzmann variable
                  (typically the largest phi in the dataset, in cm or m).
                  Appears as phi_ref^2 in the normalised ODE.
    scaling     : CapillarityScaling
                  Forward/inverse maps between normalised and dimensional
                  phi and theta.  Fixed (not trained).
    """

    diffusivity : eqx.Module
    phi_ref     : float = eqx.field(static=True)
    scaling     : eqx.Module

    def __init__(self, diffusivity, phi_ref: float, scaling):
        self.diffusivity = diffusivity
        self.phi_ref     = float(phi_ref)
        self.scaling     = scaling

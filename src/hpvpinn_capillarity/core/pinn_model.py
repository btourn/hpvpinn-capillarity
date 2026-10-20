"""PINN module: bundles the neural network and the physical model.

It acts as a container that lets equinox partition the trainable leaves (network weights and diffusivity
parameters) from the static structure.
"""

import equinox as eqx
from jax import Array


class PINN(eqx.Module):
    """
    Bundles the neural network approximation and the physical PDE model.

    Attributes
    ----------
    nn_model : FeedForwardNN
               The scalar-to-scalar network  phi_nd -> theta_nd.
    pde      : CapillarityBoltzmannPDE
               Carries the trainable diffusivity and the fixed scaling /
               phi_ref.  Embedded here so that eqx.partition correctly
               identifies the diffusivity parameters as trainable leaves.
    """

    nn_model : eqx.Module
    pde      : eqx.Module


import equinox as eqx
import jax.numpy as jnp
from jax import Array


class CapillarityScaling(eqx.Module):
    phi_ref: float = eqx.field(static=True)  #Array
    theta_i: float = eqx.field(static=True)  #Array
    theta_b: float = eqx.field(static=True)  #Array

    def __init__(
        self,
        phi_ref,
        theta_i,
        theta_b,
        dtype=jnp.float32,
    ):
        self.phi_ref = float(phi_ref)  #jnp.asarray(phi_ref, dtype)
        self.theta_i = float(theta_i)  #jnp.asarray(theta_i, dtype)
        self.theta_b = float(theta_b)  #jnp.asarray(theta_b, dtype)

    # ---- phi (Boltzmann variable) ----
    def phi_forward(self, phi: Array) -> Array:
        return phi / self.phi_ref

    def phi_inverse(self, phi_nd: Array) -> Array:
        return phi_nd * self.phi_ref

    # ---- theta (moisture content) ----
    def theta_forward(self, theta: Array) -> Array:
        return (theta - self.theta_i) / (self.theta_b - self.theta_i)

    def theta_inverse(self, theta_nd: Array) -> Array:
        return theta_nd * (self.theta_b - self.theta_i) + self.theta_i
    
    # ---- sigma (moisture content) ----
    def sigma_forward(self, sigma: Array) -> Array:
        return sigma / (self.theta_b - self.theta_i)


import equinox as eqx
import jax.numpy as jnp
import jax.nn as jnn
from jax import Array


class FixedParam(eqx.Module):
    val: float = eqx.field(static=True)

    def __init__(self, value):
        self.val = float(value)

    def value(self):
        return self.val


class FreeParam(eqx.Module):
    hat: Array

    def __init__(self, value=1., dtype=jnp.float32):
        self.hat = jnp.asarray(value, dtype)

    def value(self):
        return self.hat


class BoundedParam(eqx.Module):
    hat: Array
    min: float = eqx.field(static=True)
    max: float = eqx.field(static=True)

    def __init__(
        self,
        value: float,
        min: float,
        max: float,
        dtype=jnp.float32,
    ):
        if not (min < value < max):
            raise ValueError("value must be strictly between min and max")

        z = (value - min) / (max - value)
        self.hat = jnp.asarray(jnp.log(z), dtype)
        self.min = float(min)
        self.max = float(max)

    def value(self) -> Array:
        return self.min + (self.max - self.min) * jnn.sigmoid(self.hat)


class LowerBoundedParam(eqx.Module):
    hat: Array
    min: float = eqx.field(static=True)

    def __init__(
        self,
        value: float,
        min: float,
        dtype=jnp.float32,
    ):
        if value <= min:
            raise ValueError("value must be > min")

        self.hat = jnp.asarray(jnp.log(value - min), dtype)
        self.min = float(min)

    def value(self) -> Array:
        return self.min + jnp.exp(self.hat)


class UpperBoundedParam(eqx.Module):
    hat: Array
    max: float = eqx.field(static=True)

    def __init__(
        self,
        value: float,
        max: float,
        dtype=jnp.float32,
    ):
        if value >= max:
            raise ValueError("value must be < max")

        self.hat = jnp.asarray(jnp.log(max - value), dtype)
        self.max = float(max)

    def value(self) -> Array:
        return self.max - jnp.exp(self.hat)


ParamLike = FixedParam | LowerBoundedParam | UpperBoundedParam | BoundedParam


class VanGenuchtenDiffusivity(eqx.Module):
    """
    Van Genuchten (1980) moisture diffusivity.

        D(theta) = KsAlpha * (1-m) / (m * (theta_s - theta_r))
                 * Se^(l - 1/m)
                 * [(1 - Se^(1/m))^(-m) + (1 - Se^(1/m))^m - 2]

        where  Se = (theta - theta_r) / (theta_s - theta_r)

    Design notes
    ------------
    * KsAlpha = Ks/alpha is optimised as a single quantity.  Ks and alpha
      are individually unidentifiable from moisture-content data alone (they
      appear only as a ratio in D), so collapsing them avoids a spurious
      degree of freedom.  This is consistent with the frontx convention and
      with how the parameter is reported in Table 1 of the reference papers.

    * theta_r is the *residual moisture content*, a material property of D.
      It is conceptually distinct from theta_i (the experimental initial
      condition), which is fixed externally via CapillarityScaling and the
      boundary conditions.  theta_r must be kept < theta_s.

    * theta_s (saturated moisture content) is treated as fixed because it
      is straightforward to measure independently (e.g. from porosity) and
      is not reported as a fitted parameter in the reference works.

    Trainable parameters (via ParamLike):  KsAlpha, m, l, theta_r
    Fixed parameters:                      theta_s
    """
    KsAlpha: ParamLike
    m:       ParamLike
    l:       ParamLike
    theta_r: ParamLike

    theta_s: float = eqx.field(static=True)

    def __init__(
        self,
        KsAlpha: ParamLike,
        m:       ParamLike,
        l:       ParamLike,
        theta_r: ParamLike,
        theta_s: float,
    ):
        self.KsAlpha = KsAlpha
        self.m       = m
        self.l       = l
        self.theta_r = theta_r
        self.theta_s = float(theta_s)

    def __call__(self, theta: Array) -> Array:
        KsAlpha = self.KsAlpha.value()
        m       = self.m.value()
        l       = self.l.value()
        theta_r = self.theta_r.value()

        # Effective saturation
        Se = (theta - theta_r) / (self.theta_s - theta_r)
        #Se = jnp.clip(Se, 1e-8, 1.0 - 1e-8)

        inv_m  = 1.0 / m
        Se_pow = Se ** inv_m

        term1 = Se ** (l - inv_m)
        term2 = (1.0 - Se_pow) ** (-m)
        term3 = (1.0 - Se_pow) ** m

        D = (
            KsAlpha
            * (1.0 - m)
            / (m * (self.theta_s - theta_r))
            * term1
            * (term2 + term3 - 2.0)
        )

        return D


class LETdDiffusivity(eqx.Module):
    """
    LETd moisture diffusivity (Gerlero et al., 2022).

    The model expresses D as proportional to the derivative of the
    LET (Lomeland-Ebeltoft-Thomas) correlation F with respect to Se:

        F(Se; L, E, T)  =  Se^L / (Se^L + E * (1-Se)^T)

        dF/dSe  =  E * Se^(L-1) * (1-Se)^(T-1)
                   * [L*(1-Se) + T*Se]
                   / [Se^L + E*(1-Se)^T]^2

        D(theta) = Dwt * (theta_s - theta_r) * dF/dSe

    where Se = (theta - theta_r) / (theta_s - theta_r).

    Trainable parameters (via ParamLike):  Dwt, L, E, T, theta_r
    Fixed parameters:  theta_s
    """
    Dwt:     ParamLike
    L:       ParamLike
    E:       ParamLike
    T:       ParamLike
    theta_r: ParamLike

    theta_s: float = eqx.field(static=True)

    def __init__(
        self,
        Dwt:     ParamLike,
        L:       ParamLike,
        E:       ParamLike,
        T:       ParamLike,
        theta_r: ParamLike,
        theta_s: float,
    ):
        self.Dwt     = Dwt
        self.L       = L
        self.E       = E
        self.T       = T
        self.theta_r = theta_r
        self.theta_s = float(theta_s)

    def __call__(self, theta: Array) -> Array:
        Dwt     = self.Dwt.value()
        L       = self.L.value()
        E       = self.E.value()
        T       = self.T.value()
        theta_r = self.theta_r.value()

        # Effective saturation
        Se = (theta - theta_r) / (self.theta_s - theta_r)
        Se = jnp.clip(Se, 1e-8, 1.0 - 1e-8)
        D = Dwt * Se**L / (Se**L + E * (1.0 - Se)**T)
        return D

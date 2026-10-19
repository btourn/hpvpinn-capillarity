
import jax
import jax.numpy as jnp
import equinox as eqx


class MeasurementSet(eqx.Module):
    """
    Immutable dataset for measurements.

    Design principles
    -----------------
    - Constructor defines *data state*
    - split() defines *data generation* (noise + partition)
    """

    phi: jnp.ndarray
    theta: jnp.ndarray
    theta_exact: jnp.ndarray | None
    sigma: jnp.ndarray
    _sigma_known: bool = eqx.field(static=True)

    def __init__(
        self,
        phi,
        theta,
        *,
        sigma,
        theta_exact=None,
        true_value: bool = False,
        _sigma_known: bool | None = None,
    ):
        phi   = jnp.asarray(phi).reshape(-1)
        theta = jnp.asarray(theta).reshape(-1)

        # ---------------------------
        # Sigma normalization
        # ---------------------------
        if jnp.isscalar(sigma):
            sigma_arr = jnp.full_like(theta, float(sigma))
        else:
            sigma_arr = jnp.asarray(sigma).reshape(-1)
            if sigma_arr.shape != theta.shape:
                raise ValueError("sigma array must match theta shape")

        if jnp.any(sigma_arr < 0):
            raise ValueError("sigma must be non-negative")

        # ---------------------------
        # theta_exact logic
        # ---------------------------
        if true_value:
            theta_exact_arr = theta
        else:
            if theta_exact is None:
                theta_exact_arr = None
            else:
                theta_exact_arr = jnp.asarray(theta_exact).reshape(-1)
                if theta_exact_arr.shape != theta.shape:
                    raise ValueError("theta_exact must match theta shape")

        # ---------------------------
        # sigma_known logic
        # ---------------------------
        if _sigma_known is None:
            sigma_known = bool(jnp.any(sigma_arr > 0))
        else:
            sigma_known = bool(_sigma_known)

        # ---------------------------
        # Store
        # ---------------------------
        self.phi          = phi
        self.theta        = theta
        self.theta_exact  = theta_exact_arr
        self.sigma        = sigma_arr
        self._sigma_known = sigma_known

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def N(self):
        return self.phi.shape[0]

    @property
    def sigma_known(self):
        return self._sigma_known

    @property
    def has_ground_truth(self):
        return self.theta_exact is not None

    # ------------------------------------------------------------------
    # Splitting + noise injection
    # ------------------------------------------------------------------

    def split(
        self,
        val_fraction: float,
        test_fraction: float,
        key,
    ):
        """
        Split dataset and inject noise.

        Uses self.sigma as noise model.
        """

        if val_fraction + test_fraction >= 1.0:
            raise ValueError("val_fraction + test_fraction must be < 1")

        N = self.N
        perm = jax.random.permutation(key, N)

        n_test = int(test_fraction * N)
        n_val  = int(val_fraction * N)

        test_idx  = jnp.sort(perm[:n_test]) if n_test > 0 else None
        val_idx   = jnp.sort(perm[n_test:n_test + n_val]) if n_val > 0 else None
        train_idx = jnp.sort(perm[n_test + n_val:])

        noise_model_available = self.theta_exact is not None
        noise_free = jnp.all(self.sigma == 0.0)

        def _subset(idx, subkey):
            if idx is None:
                return None

            phi_sub   = self.phi[idx]
            theta_sub = self.theta[idx]
            sigma_sub = self.sigma[idx]

            if noise_model_available:
                # Synthetic case (we can generate noise)
                theta_ex = self.theta_exact[idx]

                if noise_free:
                    theta_obs = theta_ex
                    sigma_eff = jnp.ones_like(theta_ex)
                    sigma_known = False
                else:
                    noise = sigma_sub * jax.random.normal(subkey, theta_ex.shape)
                    theta_obs = theta_ex + noise
                    sigma_eff = sigma_sub
                    sigma_known = True

                return MeasurementSet(
                    phi_sub,
                    theta_obs,
                    sigma=sigma_eff,
                    theta_exact=theta_ex,
                    true_value=False,
                    _sigma_known=sigma_known,
                )

            else:
                # Real-data case → NO noise added
                return MeasurementSet(
                    phi_sub,
                    theta_sub,
                    sigma=sigma_sub,
                    theta_exact=None,
                    true_value=False,
                    _sigma_known=self._sigma_known,
                )

        k1, k2, k3 = jax.random.split(key, 3)

        train = _subset(train_idx, k1)
        val   = _subset(val_idx,   k2)
        test  = _subset(test_idx,  k3)

        if noise_free:
            print(
                "[Warning] sigma=0 → noise-free dataset. "
                "chi2_nu is undefined; use mse or l2_relative."
            )

        return train, val, test

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def mse(self, model):
        theta_pred = jax.vmap(model)(self.phi)
        return jnp.mean((theta_pred - self.theta) ** 2)

    def relative_l2(self, model):
        if self.theta_exact is None:
            raise RuntimeError("Ground truth not available")

        theta_pred = jax.vmap(model)(self.phi)
        return (
            jnp.linalg.norm(theta_pred - self.theta_exact)
            / jnp.linalg.norm(self.theta_exact)
        )

    def chi2_nu(self, model, n_params):
        """Compute the reduced chi-square statistic for a fitted *model*.
        chi2_nu = (1 / (N - n_params)) * sum_j [(theta_j - theta_pred_j) / sigma_j]^2
        
        Parameters
        ----------
        model    : callable (phi -> theta), typically a trained PINN.
        n_params : number of free parameters that were estimated (degrees of
                   freedom correction).  For Van Genuchten with Ks, m, l, theta_r
                   this is 4; for LETd with Dwt, L, E, T, theta_r it is 5.

        Returns
        -------
        Scalar JAX array. Represents the reduced chi-square statistic. Values near 1 indicate
        a good fit with correctly estimated noise. Values >> 1 suggest underfitting or underestimated noise; 
        values << 1 suggest overfitting or overestimated noise.
        """

        theta_pred = jax.vmap(model)(self.phi)
        dof = self.N - n_params
        if dof <= 0:
            raise ValueError("Degrees of freedom must be positive")

        if not self._sigma_known:
            print("Sigma unknown → chi2_nu is undefined. Returning MSE instead.")
            return self.mse(model)

        res = (theta_pred - self.theta) / self.sigma
        chi2 = jnp.sum(res**2)

        return chi2 / dof
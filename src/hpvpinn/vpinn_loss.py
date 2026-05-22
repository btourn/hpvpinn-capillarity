"""
vpinn_loss.py
=============
Precomputed hp-quadrature grid and weak-form residual evaluation for the
capillarity Boltzmann ODE.

Mathematical background
-----------------------
The governing ODE on the normalised domain phi in [0, 1] is:

    -phi/2 * dtheta/dphi = (1/phi_ref^2) * d/dphi( D(theta) * dtheta/dphi )

Multiplying by a test function v and integrating over element [a, b], then
applying integration by parts to the right-hand side:

    Term A  =  int_a^b  phi/2 * dtheta/dphi * v  dphi
    Term B  =  (1/phi_ref^2) * int_a^b  D(theta) * dtheta/dphi * dv/dphi  dphi
    Flux    =  (1/phi_ref^2) * [ D(theta) * dtheta/dphi * v ]_a^b

The element weak residual is  R = -Term_A + Term_B - Flux.

Why the Flux term vanishes entirely
------------------------------------
  * Interior interfaces: the network is a smooth global function, so
    D * dtheta/dphi is continuous.  Test functions vanish at element
    boundaries (we use bubble-type basis v_k = L_{k+1} - L_{k-1}), so
    the flux contribution from each side of every interior interface is
    exactly zero.
  * phi = 1 (far field): D(theta(1)) -> 0 by the Van Genuchten model,
    and dtheta/dphi(1) -> 0 is enforced by the BC loss.
  * phi = 0 (inlet): the ansatz theta = 2*sigmoid(-phi*softplus(NN(phi)))
    makes theta(0) = 1 exactly.  The inlet flux is a free (natural) BC —
    it must NOT enter the loss — and it multiplies v(0) = 0 anyway.

Therefore R = -Term_A + Term_B, involving only FIRST derivatives of the
network.  The second derivative d/dphi(D*dtheta/dphi) that caused training
instabilities in the strong-form PINN is entirely absent.

Test function construction
--------------------------
On each element [a, b] we use p bubble functions derived from Legendre
polynomials:

    v_k(xi) = L_{k+1}(xi) - L_{k-1}(xi),   k = 1, ..., p

where xi in [-1, 1] is the reference coordinate.  These satisfy
v_k(±1) = 0 by the property L_n(1) = 1, L_n(-1) = (-1)^n.

Their phi-derivative is:

    dv_k/dphi = (2/(b-a)) * (L'_{k+1}(xi) - L'_{k-1}(xi))

Mesh grading
------------
Elements are clustered near phi = 0 using a power-law map:

    phi_i = phi_max * (i / N_e)^(1/gamma),   gamma >= 1

gamma = 1 gives a uniform mesh; gamma > 1 increases density near phi = 0
where D(theta) varies most rapidly.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss

import jax
import jax.numpy as jnp
import equinox as eqx


# ---------------------------------------------------------------------------
# Low-level helpers (pure NumPy, called at construction time only)
# ---------------------------------------------------------------------------

def _gl_on_interval(a: float, b: float, n_quad: int):
    """Gauss-Legendre nodes and weights mapped from [-1,1] to [a, b]."""
    xi, w = leggauss(n_quad)
    phi   = 0.5 * (a + b) + 0.5 * (b - a) * xi
    w_map = 0.5 * (b - a) * w  # jacobian of the linear map from xi to phi
    return phi.astype(np.float32), w_map.astype(np.float32)


def _legendre_recurrence(k: int, xi: np.ndarray):
    """
    Evaluate L_k(xi) and dL_k/dxi via the three-term recurrence.
    Works for k >= 0.
    """
    n = len(xi)
    if k == 0:
        return np.ones(n), np.zeros(n)
    if k == 1:
        return xi.copy(), np.ones(n)

    P_km2, P_km1   = np.ones(n),  xi.copy()
    dP_km2, dP_km1 = np.zeros(n), np.ones(n)
    for j in range(2, k + 1):
        P   = ((2*j - 1) * xi * P_km1   - (j - 1) * P_km2)   / j
        dP  = ((2*j - 1) * (P_km1 + xi * dP_km1) - (j - 1) * dP_km2) / j
        P_km2, P_km1   = P_km1,  P
        dP_km2, dP_km1 = dP_km1, dP
    return P, dP


def _bubble_and_deriv(k: int, xi: np.ndarray, a: float, b: float):
    """
    Bubble test function v_k = L_{k+1} - L_{k-1} and its phi-derivative.

    Parameters
    ----------
    k   : test function index, k >= 1
    xi  : reference coordinates in [-1, 1]
    a,b : physical element endpoints  (used to convert dxi -> dphi)
    """
    L_hi,  dL_hi  = _legendre_recurrence(k + 1, xi)
    L_lo,  dL_lo  = _legendre_recurrence(k - 1, xi)
    v    = L_hi - L_lo
    dvdx = dL_hi - dL_lo
    dvdp = dvdx * (2.0 / (b - a))          # chain rule: dxi/dphi = 2/(b-a)
    return v.astype(np.float32), dvdp.astype(np.float32)


# ---------------------------------------------------------------------------
# VPINNLoss
# ---------------------------------------------------------------------------

class VPINNLoss(eqx.Module):
    """
    Precomputed hp-quadrature data and weak-residual evaluator.

    All quadrature nodes, weights, and test-function values are computed
    once at construction time (pure NumPy) and stored as frozen JAX arrays.
    At training time only JAX operations are performed.

    Attributes
    ----------
    quad_pts : (Ne, nq)     quadrature nodes on each element
    quad_wts : (Ne, nq)     quadrature weights on each element
    test_v   : (Ne, p, nq)  test function values at quadrature nodes
    test_dv  : (Ne, p, nq)  test function phi-derivatives at nodes
    phi_ref  : float        reference phi (dimensional) for phi_ref^2 scaling
    scaling  : CapillarityScaling instance for theta_nd <-> theta_dim conversion
    """

    quad_pts : jnp.ndarray
    quad_wts : jnp.ndarray
    test_v   : jnp.ndarray
    test_dv  : jnp.ndarray
    phi_ref  : float = eqx.field(static=True)
    scaling  : eqx.Module

    def __init__(
        self,
        phi_min         : float,
        phi_max         : float,
        phi_ref         : float,
        scaling,
        n_elements      : int   = 20,
        poly_degree     : int   = 3,
        n_quad          : int   = 8,
        element_clustering : float = 3.0,
    ):
        """
        Parameters
        ----------
        phi_min, phi_max   : normalised domain endpoints (typically 0.0 and 1.0)
        phi_ref            : dimensional reference value (cm or m) used in the
                             phi_ref^2 factor of the scaled ODE
        scaling            : CapillarityScaling instance
        n_elements         : number of elements (h-refinement)
        poly_degree        : number of bubble test functions per element (p-refinement);
                             uses v_1 ... v_{poly_degree}
        n_quad             : Gauss-Legendre points per element; choose
                             n_quad >= poly_degree + 2 for exact integration of
                             the polynomial part of the integrand
        element_clustering : power-law grading exponent gamma >= 1;
                             gamma = 1 is uniform, larger values cluster
                             more elements near phi_min (the wetting front)
        """
        # --- Graded breakpoints ---
        u = np.linspace(0.0, 1.0, n_elements + 1)
        t = u ** (1.0 / element_clustering)
        bp = phi_min + (phi_max - phi_min) * t   # shape (Ne+1,)

        q_pts_list, q_wts_list, v_list, dv_list = [], [], [], []

        for i in range(n_elements):
            a, b = float(bp[i]), float(bp[i + 1])
            pts, wts = _gl_on_interval(a, b, n_quad)

            # Reference coordinates for test-function evaluation
            xi = 2.0 * (pts - a) / (b - a) - 1.0

            v_elem, dv_elem = [], []
            for k in range(1, poly_degree + 1):
                vk, dvk = _bubble_and_deriv(k, xi, a, b)
                v_elem.append(vk)
                dv_elem.append(dvk)

            q_pts_list.append(pts)
            q_wts_list.append(wts)
            v_list.append(np.stack(v_elem,  axis=0))   # (p, nq)
            dv_list.append(np.stack(dv_elem, axis=0))  # (p, nq)

        self.quad_pts = jnp.array(np.stack(q_pts_list, axis=0))  # (Ne, nq)
        self.quad_wts = jnp.array(np.stack(q_wts_list, axis=0))  # (Ne, nq)
        self.test_v   = jnp.array(np.stack(v_list,     axis=0))  # (Ne, p, nq)
        self.test_dv  = jnp.array(np.stack(dv_list,    axis=0))  # (Ne, p, nq)
        self.phi_ref  = float(phi_ref)
        self.scaling  = scaling

    # ------------------------------------------------------------------
    # Single-element, single-test-function residual
    # ------------------------------------------------------------------

    def _element_residual(self, model, diffusivity, pts, wts, v_k, dv_k):
        """
        Compute the weak residual R = -Term_A + Term_B for one element and
        one test function.

        Parameters
        ----------
        model       : scalar phi -> scalar theta_nd  (network callable)
        diffusivity : scalar theta_dim -> scalar D   (diffusivity callable)
        pts         : (nq,)  quadrature nodes on the element
        wts         : (nq,)  quadrature weights
        v_k         : (nq,)  test function values at nodes
        dv_k        : (nq,)  test function phi-derivatives at nodes

        Returns
        -------
        R       : scalar weak residual
        term_A  : scalar contribution from the -phi/2 * dtheta/dphi term
        term_B  : scalar contribution from the diffusion term
        """
        # Network output and first derivative at quadrature points
        theta_nd  = jax.vmap(model)(pts)                   # (nq,)
        dtheta    = jax.vmap(jax.grad(model))(pts)         # (nq,)  first deriv only

        # Diffusivity at quadrature points (convert theta_nd -> theta_dim first)
        theta_dim = self.scaling.theta_inverse(theta_nd)   # (nq,)
        D         = jax.vmap(diffusivity)(theta_dim)       # (nq,)

        # Term A: int  phi/2 * dtheta/dphi * v_k  dphi
        term_A = jnp.dot(wts, 0.5 * pts * dtheta * v_k)

        # Term B: (1/phi_ref^2) * int  D * dtheta/dphi * dv_k/dphi  dphi
        term_B = jnp.dot(wts, D * dtheta * dv_k) / (self.phi_ref ** 2)

        R = -term_A + term_B
        return R, term_A, term_B

    # ------------------------------------------------------------------
    # Full hp-VPINN physics loss
    # ------------------------------------------------------------------

    def __call__(self, full_model):
        """
        Evaluate the hp-VPINN physics loss over the entire domain.

        Returns
        -------
        loss       : scalar MSE of all weak residuals  (Ne * p values)
        all_R      : (Ne, p)  Weak residuals per element per test fn
        all_A      : (Ne, p)  Term A contributions per element per test fn
        all_B      : (Ne, p)  Term B contributions per element per test fn
        """

        model = full_model.nn_model
        diffusivity = full_model.pde.diffusivity

        def residuals_for_element(pts, wts, v_elem, dv_elem):
            """Returns (p,), (p,), (p,) for one element."""
            def per_test_fn(v_k, dv_k):
                return self._element_residual(
                    model, diffusivity, pts, wts, v_k, dv_k
                )
            return jax.vmap(per_test_fn)(v_elem, dv_elem)

        # vmap over elements -> each output has shape (Ne, p)
        all_R, all_A, all_B = jax.vmap(residuals_for_element)(
            self.quad_pts,   # (Ne, nq)
            self.quad_wts,   # (Ne, nq)
            self.test_v,     # (Ne, p, nq)
            self.test_dv,    # (Ne, p, nq)
        )

        loss = jnp.mean(all_R ** 2)
        return loss, all_R, all_A, all_B

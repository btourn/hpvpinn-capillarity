"""
conventional_inverse.py
=======================
Conventional (forward-solver-in-the-loop) inverse modelling baseline for the
hp-VPINN benchmark (R1-22, R2-M09, R3-M2).

Procedure -- follows frontx's own reference example
(frontx/examples/inverse/validity/letd.py, upstream commit e04d264):

  * outer loop  : scipy.optimize.differential_evolution (DE) over the SHAPE
                  parameters of the diffusivity model;
  * forward     : frontx.solve -- shooting method on the Boltzmann-transformed
                  ODE (bisection on dtheta/do at o=0, Kvaerno5 integration);
  * scale       : the diffusivity scale (Ks/alpha or Dwt) is held at a fixed
                  reference value inside the model and profiled out by
                  frontx.ScaledSolution.fitting_data (Levenberg-Marquardt on
                  D0, exploiting theta(o; D0*D) = theta(o/sqrt(D0); D));
                  estimated scale = reference value * D0;
  * objective   : reduced chi-square on the fitting points,
                  sum(((theta - theta_model)/sigma)^2) / (N - n_params).

Differences from frontx.fit / de_fit, all needed for the benchmark and none
of which change the optimisation problem:
  * DE seed, popsize, tol, atol, maxiter and polish are exposed;
  * the OptimizeResult (nfev, nit, success, message) is kept;
  * an initial guess x0 (the hp-VPINN starting point) is passed to DE;
  * failed forward solves return a large finite penalty instead of +inf so
    that DE's population-spread convergence test stays well defined;
  * compilation time, single-evaluation time and DE time are measured
    separately.

Case parametrisations (see the benchmark plan):
  grenoblesand : VG, l=0.5 and theta_r=0 fixed; DE over m;              scale Ks/alpha
  letd         : LETd; DE over L, E, T, theta_r (upstream bounds);      scale Dwt
  vangenuchten : VG; DE over m, l, theta_r;                             scale Ks/alpha
  brooks_corey : BC; DE over p = l + 1 + 1/n and theta_r;               scale A = (Ks/alpha)/n
                 (only A, p, theta_r are identifiable from theta(o) data:
                  D = A/(theta_s - theta_r) * Se**p).
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.metadata
import os
import platform
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

import jax.numpy as jnp
import numpy as np
from scipy.optimize import differential_evolution

import frontx
from frontx.models import BrooksAndCorey, LETd, VanGenuchten

try:
    from frontx import ScaledSolution
except ImportError:  # older frontx checkouts
    from frontx._inverse.fit import ScaledSolution


PENALTY = 1e12  # cost assigned to candidates whose forward solve fails


# ---------------------------------------------------------------------------
# DE presets
# ---------------------------------------------------------------------------

DE_PRESETS = {
    # frontx's reference inverse example (examples/inverse/validity/letd.py):
    # scipy defaults except polish=False and atol=1 on the reduced chi-square.
    "upstream": dict(strategy="best1bin", maxiter=1000, popsize=15,
                     tol=0.01, atol=1.0, polish=False),
    # frontx.fit / de_fit defaults: scipy defaults except maxiter=15.
    "frontx_fit": dict(strategy="best1bin", maxiter=15, popsize=15,
                       tol=0.01, atol=0.0, polish=True),
    # Benchmark headline: scipy's own convergence test (relative tol only) plus
    # the final L-BFGS-B polish. atol=1 on a reduced chi-square of O(1) stops
    # DE almost immediately (pilot: nit=2 for Grenoble), before convergence.
    "converged": dict(strategy="best1bin", maxiter=1000, popsize=15,
                      tol=0.01, atol=0.0, polish=True),
}


# ---------------------------------------------------------------------------
# Case specification
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ParamSpec:
    name: str
    lo: float
    hi: float
    x0: float  # hp-VPINN starting value, in DE coordinates


@dataclasses.dataclass(frozen=True)
class CaseSpec:
    name: str
    params: tuple[ParamSpec, ...]
    build_D: Callable[[jax.Array], Callable]  # DE vector -> diffusivity (scale fixed)
    scale_name: str
    scale_ref: float                            # fixed scale inside build_D
    theta_i: float
    theta_b: float
    n_params: int                               # dof correction, same as hp-VPINN
    reference: dict[str, float]                 # reporting coordinates
    report: Callable[[np.ndarray, float], dict[str, float]]

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [(p.lo, p.hi) for p in self.params]

    @property
    def x0(self) -> np.ndarray:
        return np.array([p.x0 for p in self.params], dtype=np.float64)


def build_case(case: str) -> CaseSpec:
    """Build the case spec, reading constants from the matching
    hparam_search_<case>.py so that there is a single source of truth."""
    m = importlib.import_module(f"hparam_search_{case}")

    if case == "grenoblesand":
        theta_s, l_fix, theta_r = float(m.THETA_S), float(m.L_FIXED), float(m.THETA_R)
        scale_ref = 1.0  # == hp-VPINN KsAlpha initial value

        def build_D(x):
            return VanGenuchten(m=x[0], l=l_fix, Ks=scale_ref,
                                theta_range=(theta_r, theta_s))

        return CaseSpec(
            name=case,
            params=(ParamSpec("m", 0.01, 0.99, 0.5),),
            build_D=build_D, scale_name="KsAlpha", scale_ref=scale_ref,
            theta_i=float(m.I_BC), theta_b=float(m.B_BC),
            n_params=int(m.N_PARAMS),
            reference=dict(m.TRUE_PARAMS),
            report=lambda x, s: dict(KsAlpha=s, m=float(x[0])),
        )

    if case == "letd":
        theta_s, theta_i = float(m.THETA_S), float(m.THETA_I)
        scale_ref = 1e-3  # as in frontx's reference example

        def build_D(x):
            return LETd(Dwt=scale_ref, L=x[0], E=x[1], T=x[2],
                        theta_range=(x[3], theta_s))

        return CaseSpec(
            name=case,
            params=(  # bounds exactly as in frontx's reference example
                ParamSpec("L", 0.0, 10.0, float(m.L_INIT)),
                ParamSpec("E", 0.0, 1e5, float(m.E_INIT)),
                ParamSpec("T", 0.0, 10.0, float(m.T_INIT)),
                ParamSpec("theta_r", 0.0, theta_i, float(m.THETA_R_INIT)),
            ),
            build_D=build_D, scale_name="Dwt", scale_ref=scale_ref,
            theta_i=theta_i, theta_b=float(m.THETA_B),
            n_params=int(m.N_PARAMS),
            reference=dict(m.TRUE_PARAMS),
            report=lambda x, s: dict(Dwt=s, L=float(x[0]), E=float(x[1]),
                                     T=float(x[2]), theta_r=float(x[3])),
        )

    if case == "vangenuchten":
        theta_s, theta_i = float(m.THETA_S), float(m.THETA_I)
        scale_ref = float(m.KSALPHA_INIT)

        def build_D(x):
            return VanGenuchten(m=x[0], l=x[1], Ks=scale_ref,
                                theta_range=(x[2], theta_s))

        return CaseSpec(
            name=case,
            params=(
                ParamSpec("m", 0.01, 0.99, float(m.M_INIT)),
                ParamSpec("l", -2.0, 10.0, float(m.L_INIT)),
                ParamSpec("theta_r", 0.0, theta_i, float(m.THETA_R_INIT)),
            ),
            build_D=build_D, scale_name="KsAlpha", scale_ref=scale_ref,
            theta_i=theta_i, theta_b=float(m.THETA_B),
            n_params=int(m.N_PARAMS),
            reference=dict(m.TRUE_PARAMS),
            report=lambda x, s: dict(KsAlpha=s, m=float(x[0]), l=float(x[1]),
                                     theta_r=float(x[2])),
        )

    if case == "brooks_corey":
        theta_s, theta_i = float(m.THETA_S), float(m.THETA_I)
        scale_ref = float(m.KSALPHA_INIT)
        # n = 1 is a representative of the (n, l) equivalence class:
        # with n = 1, l = p - 2 gives Se**(l + 1 + 1/n) = Se**p and A = Ks/alpha.

        def build_D(x):
            return BrooksAndCorey(n=1.0, l=x[0] - 2.0, Ks=scale_ref,
                                  theta_range=(x[1], theta_s))

        ref = m.TRUE_PARAMS
        n0, l0 = float(m.N_INIT), float(m.L_INIT)
        return CaseSpec(
            name=case,
            params=(
                # p > 20 makes D(theta_i)/D(theta_s) < 1e-28: the shooting
                # solve fails for most such candidates (pilot: 54% failures).
                ParamSpec("p", 1.0, 20.0, l0 + 1.0 + 1.0 / n0),
                ParamSpec("theta_r", 0.0, theta_i, float(m.THETA_R_INIT)),
            ),
            build_D=build_D, scale_name="A", scale_ref=scale_ref,
            theta_i=theta_i, theta_b=float(m.THETA_B),
            n_params=int(m.N_PARAMS),
            reference=dict(
                A=ref["KsAlpha"] / ref["n"],
                p=ref["l"] + 1.0 + 1.0 / ref["n"],
                theta_r=ref["theta_r"],
            ),
            report=lambda x, s: dict(A=s, p=float(x[0]), theta_r=float(x[1])),
        )

    raise ValueError(f"unknown case {case!r}")


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _solve_kw(itol):
    return {} if itol is None else {"itol": itol}


def make_problem(spec: CaseSpec, o, theta, sigma, itol: float | None = None):
    """Return (solve_scaled, cost) for the given fitting points.

    cost(x) is the reduced chi-square of the scaled forward solution, or
    PENALTY if the forward solve fails. It is jitted once per data set.
    """
    o = jnp.asarray(o, dtype=jnp.float64)
    theta = jnp.asarray(theta, dtype=jnp.float64)
    sigma = jnp.broadcast_to(jnp.asarray(sigma, dtype=jnp.float64), theta.shape)
    dof = o.shape[0] - spec.n_params

    def solve_scaled(x):
        D = spec.build_D(x)
        sol = frontx.solve(D, i=spec.theta_i, b=spec.theta_b, throw=False, **_solve_kw(itol))
        return ScaledSolution.fitting_data(sol, o, theta, sigma, throw=False)

    @jax.jit
    def cost(x):
        s = solve_scaled(x)
        chi2 = jnp.sum(((theta - s(o)) / sigma) ** 2) / dof
        ok = (s.original.result == frontx.RESULTS.successful) & jnp.isfinite(chi2)
        return jnp.where(ok, chi2, PENALTY)

    return solve_scaled, cost


def reduced_chi2(pred, theta, sigma, n_params) -> float:
    pred, theta = np.asarray(pred, float), np.asarray(theta, float)
    sigma = np.broadcast_to(np.asarray(sigma, float), theta.shape)
    return float(np.sum(((theta - pred) / sigma) ** 2) / max(len(theta) - n_params, 1))


class CountingCost:
    """Wraps the jitted cost for scipy; counts evaluations and failures."""

    def __init__(self, fn):
        self.fn = fn
        self.nfev = 0
        self.nfail = 0
        self.best = float("inf")

    def __call__(self, x) -> float:
        v = float(self.fn(jnp.asarray(x, dtype=jnp.float64)))
        self.nfev += 1
        if v >= PENALTY:
            self.nfail += 1
        self.best = min(self.best, v)
        return v


# ---------------------------------------------------------------------------
# One DE run
# ---------------------------------------------------------------------------

def run_de(spec: CaseSpec, fit_data: dict, test_data: dict | None, *,
           seed: int, preset: str = "converged", n_timing: int = 20,
           progress: bool = True, max_minutes: float | None = None,
           itol: float | None = None, **overrides) -> dict:
    """Run one DE inversion and return a flat, JSON-serialisable record.

    fit_data / test_data: dict(o=..., theta=..., sigma=...) in physical units.
    """
    settings = dict(DE_PRESETS[preset], **overrides)
    solve_scaled, cost = make_problem(spec, fit_data["o"], fit_data["theta"],
                                      fit_data["sigma"], itol=itol)
    x0 = spec.x0

    # -- compilation (first call) and steady-state single-evaluation time --
    t0 = time.perf_counter()
    cost_x0 = float(cost(jnp.asarray(x0)))
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_timing):
        float(cost(jnp.asarray(x0)))
    t_eval = (time.perf_counter() - t0) / n_timing

    # -- DE ------------------------------------------------------------------
    counting = CountingCost(cost)
    state = dict(gen=0, stopped_by_time=False)
    t0 = time.perf_counter()

    def callback(intermediate_result):  # new-style scipy callback (name matters)
        state["gen"] += 1
        elapsed = time.perf_counter() - t0
        if progress:
            conv = getattr(intermediate_result, "convergence", float("nan"))
            print(f"    gen {state['gen']:4d}  nfev={counting.nfev:6d}  "
                  f"best chi2_nu={counting.best:.6g}  convergence={conv:.3g}  "
                  f"failed={counting.nfail}  t={elapsed:7.1f}s", flush=True)
        if max_minutes is not None and elapsed > 60.0 * max_minutes:
            state["stopped_by_time"] = True
            return True
        return False

    opt = differential_evolution(
        counting, bounds=spec.bounds, x0=x0, seed=seed,
        init="latinhypercube", updating="immediate", workers=1,
        callback=callback, **settings,
    )
    t_de = time.perf_counter() - t0

    # -- post-processing -----------------------------------------------------
    x = np.asarray(opt.x, dtype=np.float64)
    s = solve_scaled(jnp.asarray(x))
    D0 = float(s.D0)
    scale_est = spec.scale_ref * D0
    params = spec.report(x, scale_est)

    chi2_fit = reduced_chi2(s(jnp.asarray(fit_data["o"])), fit_data["theta"],
                            fit_data["sigma"], spec.n_params)
    chi2_test = (reduced_chi2(s(jnp.asarray(test_data["o"])), test_data["theta"],
                              test_data["sigma"], spec.n_params)
                 if test_data is not None else float("nan"))

    # Independent check (as in frontx's example): re-solve with the fitted,
    # already-scaled diffusivity and no rescaling.
    sol_check = frontx.solve(s.D, i=spec.theta_i, b=spec.theta_b, throw=False,
                             **_solve_kw(itol))
    chi2_fit_check = reduced_chi2(sol_check(jnp.asarray(fit_data["o"])),
                                  fit_data["theta"], fit_data["sigma"], spec.n_params)

    near_bound = [
        p.name for p, xi in zip(spec.params, x)
        if min(xi - p.lo, p.hi - xi) < 0.01 * (p.hi - p.lo)
    ]
    rel_err = {
        f"err_{k}": abs(params[k] - v) / abs(v)
        for k, v in spec.reference.items() if k in params and v != 0
    }

    # Curves for figures (physical units)
    o_max = float(np.max(np.concatenate(
        [fit_data["o"]] + ([test_data["o"]] if test_data is not None else []))))
    o_grid = np.linspace(0.0, 1.05 * o_max, 200)
    theta_grid = np.linspace(spec.theta_i + 1e-6, spec.theta_b, 200)

    return dict(
        case=spec.name, preset=preset, seed=seed,
        de_settings=settings, bounds=spec.bounds, x0=x0.tolist(),
        de_param_names=[p.name for p in spec.params],
        x_opt=x.tolist(), D0=D0, scale_name=spec.scale_name,
        scale_ref=spec.scale_ref,
        **params, **rel_err,
        reference=spec.reference,
        chi2_nu_fit=chi2_fit, chi2_nu_test=chi2_test,
        chi2_nu_fit_check=chi2_fit_check, de_fun=float(opt.fun),
        cost_x0=cost_x0,
        n_fit=int(len(fit_data["o"])),
        n_test=int(len(test_data["o"])) if test_data is not None else 0,
        nfev=int(opt.nfev), nfev_counted=counting.nfev, nfail=counting.nfail,
        nit=int(opt.nit), success=bool(opt.success), message=str(opt.message),
        hit_maxiter=bool(opt.nit >= settings["maxiter"]),
        stopped_by_time=state["stopped_by_time"], max_minutes=max_minutes,
        itol=itol if itol is not None else 1e-3,  # frontx.solve default
        near_bound=near_bound,
        t_compile_s=t_compile, t_eval_s=t_eval, t_de_s=t_de,
        t_total_s=t_compile + t_de,
        o_grid=o_grid.tolist(),
        theta_profile=np.asarray(s(jnp.asarray(o_grid))).tolist(),
        theta_grid=theta_grid.tolist(),
        D_grid=np.asarray(s.D(jnp.asarray(theta_grid))).tolist(),
    )


# ---------------------------------------------------------------------------
# Environment record (for the SI)
# ---------------------------------------------------------------------------

def _cpu_name() -> str:
    try:
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if platform.system() == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or "unknown"


def environment_info() -> dict:
    import diffrax
    import scipy

    info = dict(
        python=platform.python_version(), system=platform.system(),
        machine=platform.machine(), cpu=_cpu_name(), n_cpus=os.cpu_count(),
        jax=jax.__version__, jax_backend=jax.default_backend(),
        diffrax=diffrax.__version__, scipy=scipy.__version__,
        xla_flags=os.environ.get("XLA_FLAGS", ""),
        frontx_path=str(Path(frontx.__file__).parent),
    )
    try:
        info["frontx"] = importlib.metadata.version("frontx")
    except importlib.metadata.PackageNotFoundError:
        info["frontx"] = "unknown"
    try:
        info["frontx_git"] = subprocess.check_output(
            ["git", "-C", info["frontx_path"], "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        info["frontx_git"] = "unknown"
    return info


# ---------------------------------------------------------------------------
# Profile scan (1-D cases): chi2_nu(m) with the scale profiled out
# ---------------------------------------------------------------------------

def profile_scan(spec: CaseSpec, data: dict, grid, itol: float | None = None) -> list[dict]:
    """Evaluate the profiled objective on a grid of the (single) DE parameter.

    Returns one dict per grid point: value, chi2_nu, scale estimate. With
    dof = N - n_params, Delta chi2 = (chi2_nu - min) * dof <= 1 (3.84) gives
    the 68% (95%) profile-likelihood interval for that parameter.
    """
    if len(spec.params) != 1:
        raise ValueError("profile_scan supports 1-parameter cases only")
    solve_scaled, cost = make_problem(spec, data["o"], data["theta"], data["sigma"], itol=itol)
    rows = []
    for v in np.asarray(grid, dtype=np.float64):
        x = jnp.asarray([v])
        c = float(cost(x))
        s = solve_scaled(x) if c < PENALTY else None
        rows.append(dict(value=float(v), chi2_nu=c,
                         scale=spec.scale_ref * float(s.D0) if s is not None else float("nan")))
    return rows


# ---------------------------------------------------------------------------
# Forward-consistency check of externally estimated parameters (e.g. hp-VPINN)
# ---------------------------------------------------------------------------

def model_from_params(case: str, prm: dict):
    """Diffusivity built from parameters in the hp-VPINN reporting convention
    (same keys as TRUE_PARAMS in hparam_search_<case>.py). Ks/alpha enters
    frontx models as Ks with alpha = 1."""
    m = importlib.import_module(f"hparam_search_{case}")
    if case == "grenoblesand":
        return VanGenuchten(m=prm["m"], l=float(m.L_FIXED), Ks=prm["KsAlpha"],
                            theta_range=(float(m.THETA_R), float(m.THETA_S)))
    if case == "vangenuchten":
        return VanGenuchten(m=prm["m"], l=prm["l"], Ks=prm["KsAlpha"],
                            theta_range=(prm["theta_r"], float(m.THETA_S)))
    if case == "letd":
        return LETd(Dwt=prm["Dwt"], L=prm["L"], E=prm["E"], T=prm["T"],
                    theta_range=(prm["theta_r"], float(m.THETA_S)))
    if case == "brooks_corey":
        return BrooksAndCorey(n=prm["n"], l=prm["l"], Ks=prm["KsAlpha"],
                              theta_range=(prm["theta_r"], float(m.THETA_S)))
    raise ValueError(case)


def forward_check(spec: CaseSpec, prm: dict, fit_data: dict, test_data: dict | None,
                  itol: float | None = None) -> dict:
    """chi2_nu of the EXACT forward solution for the given parameters (no
    rescaling), plus chi2_nu after refitting only the scale D0 (shape check)."""
    D = model_from_params(spec.name, prm)
    sol = frontx.solve(D, i=spec.theta_i, b=spec.theta_b, throw=False, **_solve_kw(itol))
    ok = bool(sol.result == frontx.RESULTS.successful)
    out = dict(solve_ok=ok)
    if not ok:
        return out
    out["chi2_nu_fit_exact"] = reduced_chi2(sol(jnp.asarray(fit_data["o"])), fit_data["theta"],
                                            fit_data["sigma"], spec.n_params)
    if test_data is not None:
        out["chi2_nu_test_exact"] = reduced_chi2(sol(jnp.asarray(test_data["o"])),
                                                 test_data["theta"], test_data["sigma"],
                                                 spec.n_params)
    sigma = np.broadcast_to(np.asarray(fit_data["sigma"], float), np.shape(fit_data["theta"]))
    s = ScaledSolution.fitting_data(sol, jnp.asarray(fit_data["o"]), jnp.asarray(fit_data["theta"]),
                                    jnp.asarray(sigma), throw=False)
    out["D0_refit"] = float(s.D0)
    out["chi2_nu_fit_scale_refit"] = reduced_chi2(s(jnp.asarray(fit_data["o"])), fit_data["theta"],
                                                  fit_data["sigma"], spec.n_params)
    return out

"""Shared machinery of the final hp-VPINN runs of each case study (grenoblesand.py, vangenuchten.py,
letd.py).

A final run fixes the configuration (architecture, hp discretisation, schedule) and trains it with N
fresh network-initialisation seeds, one after another on the same machine. It reports the ensemble:
figures from a representative run chosen by a fixed rule (bands over the converged seeds), parameter and
metric statistics over the converged seeds, and computing time (JIT compilation, training, per epoch)
as median and range over all seeds, plus the total cost of the HPO campaign.

Configuration: by default the winner selected for the case, winners.json["selected"][<case>], from
    python -m hpvpinn_capillarity.hpo.postprocess --model hp_vpinn --winners
  --schedule es|gb                       use that schedule's winner instead
  --arch "W=..,L=..,act=..,Nel=..,p=.."  user-defined configuration (with --schedule)

Training: each seed runs the HPO's own run_vpinn_trial() with the seed as trial number, so data, model,
settings and metrics are those of the HPO. Seeds run sequentially, never in parallel, so that timings
are not affected by contention.

Outputs:
<results>/hp_vpinn/final_runs/<tag>/
    reference.npz                     data, noise-free curve, reference D(theta)
    seeds/seed_<n>/record.json        config, metrics, parameters, timing, environment
    seeds/seed_<n>/history.npz        training history
    seeds/seed_<n>/curves.npz         theta~(phi~) and D(theta) of the trained model
    seeds/seed_<n>/model.eqx          trained PINN
<results>/hp_vpinn/Figures/   (--figures-dir)
    final_losses_<tag>.eps, final_profile_<tag>.eps, final_diffusivity_<tag>.eps
<results>/hp_vpinn/tables/    (--tables-dir)
    final_diffusivity_<tag>.csv, latex_final_params_<tag>.tex, latex_final_timing_<tag>.tex,
    final_summary_<tag>.json

tag = <case>_<es|gb>[_<n>pct_noise][_arch-...][_ep<epochs>]

Training needs JAX; aggregation (--aggregate-only) needs only NumPy and Matplotlib.
"""

import argparse
import datetime
import inspect
import json
import os
import platform
import shutil
import socket
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import numpy as np

from hpvpinn_capillarity import paths

from hpvpinn_capillarity.final_runs.profile_figures import (ADMISSIBLE_PHYSICS, CALIBRATED_OBJ, ZOOM_BY_CASE,
                             band_from_pool, draw_profile, is_qualified)

MODEL = "hp_vpinn"

# (warmup_epochs, transition_epochs), as launched by submit_all_<case>.sh
SCHEDULES = {"gb": (0, 20_000), "es": (200_000, 0)}

DEFAULT_SEED_OFFSET = 1000      # fresh seeds, never equal to an HPO trial number
DEFAULT_N_SEEDS     = 10
DEFAULT_EVAL_EVERY  = 100
N_PHI_CURVE         = 1001      # grid of the wetting-profile curve (phi~ in [0, 1])
N_THETA_CURVE       = 1000      # grid of the D(theta) curve
SE_RANGE            = (1e-3, 0.999)   # D(theta) grid, as a fraction of [theta_i, theta_b]


# ---------------------------------------------------------------------------
# Case specification (filled in by each case script)
# ---------------------------------------------------------------------------

@dataclass
class CaseSpec:
    case: str                               # 'grenoblesand', 'letd', ...
    search_module: str                      # 'hpvpinn_capillarity.hpo.search_grenoblesand'
    label: str                              # 'Case 1 (Grenoble sand)'
    param_labels: dict                      # {param: LaTeX label}
    D_units: str                            # LaTeX units of D, e.g. r'cm$^2$/h'
    prepare_shared: Callable                # (m, data_path, noise_level) -> shared dict
    default_data: str                       # file name at the repo root
    noise_levels: Optional[tuple] = None    # (0.01, 0.02) for Case 1, else None
    reference_params: Optional[dict] = None # {param: value} drawn as reference
    reference_label: str = "Reference"
    reference_short: str = "ref."           # legend prefix in parameter panels
    has_ground_truth: bool = False          # True: reference = truth (errors, bias)
    reference_diffusivity: Optional[Callable] = None   # (m) -> D(theta) callable (jnp)
    noise_free_curve: Optional[Callable] = None        # (m, data_path) -> (o, theta)


# ---------------------------------------------------------------------------
# Module constants read from source (keeps --aggregate-only free of JAX)
# ---------------------------------------------------------------------------

def read_constant(path: Path, name: str, default=None):
    """Value of a top-level literal assignment NAME = <literal> in a .py file."""
    import ast
    try:
        tree = ast.parse(Path(path).read_text())
    except (OSError, SyntaxError):
        return default
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            v = node.value
            try:
                if (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                        and v.func.id == "dict" and not v.args):
                    return {kw.arg: ast.literal_eval(kw.value) for kw in v.keywords}
                return ast.literal_eval(v)
            except ValueError:
                return default
    return default


# ---------------------------------------------------------------------------
# Paths and tags
# ---------------------------------------------------------------------------

def noise_tag(noise_level) -> str:
    return "" if noise_level is None else f"_{int(round(noise_level * 100))}pct_noise"


def group_key(case: str, noise_level) -> str:
    """Same key as hpo.postprocess.group_key (winners.json['selected'])."""
    return f"{case}{noise_tag(noise_level)}"


def run_tag(case, schedule, noise_level, arch_suffix="", epochs=None, max_steps=None) -> str:
    tag = f"{case}_{schedule}{noise_tag(noise_level)}{arch_suffix}"
    if epochs is not None and max_steps is not None and epochs != max_steps:
        tag += f"_ep{epochs}"
    return tag


# ---------------------------------------------------------------------------
# Configuration: winners.json or user-defined
# ---------------------------------------------------------------------------

HP_COLS = ["nn_n_layers", "nn_width", "activation", "n_elements", "poly_degree"]
_ALIASES = {
    "l": "nn_n_layers", "layers": "nn_n_layers", "depth": "nn_n_layers", "nn_n_layers": "nn_n_layers",
    "w": "nn_width", "width": "nn_width", "nn_width": "nn_width",
    "act": "activation", "activation": "activation",
    "nel": "n_elements", "n_el": "n_elements", "elements": "n_elements", "n_elements": "n_elements",
    "p": "poly_degree", "degree": "poly_degree", "poly_degree": "poly_degree",
}
_CASTERS = {"nn_n_layers": int, "nn_width": int, "activation": str,
            "n_elements": int, "poly_degree": int}


def parse_arch_spec(spec: str) -> dict:
    """'W=16,L=2,act=tanh,Nel=30,p=5' -> {nn_width: 16, nn_n_layers: 2, ...}."""
    out = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Malformed item {item!r} in --arch (expected key=value).")
        key, val = (s.strip() for s in item.split("=", 1))
        if key.lower() in ("seed", "init_seed"):
            raise ValueError("--arch does not take a seed; seeds are set with --seeds/--n-seeds.")
        canon = _ALIASES.get(key.lower())
        if canon is None:
            raise ValueError(f"Unknown key {key!r} in --arch. Known: {sorted(_ALIASES)}")
        out[canon] = _CASTERS[canon](val)
    missing = [c for c in HP_COLS if c not in out]
    if missing:
        raise ValueError(f"--arch is missing: {missing}")
    return {c: out[c] for c in HP_COLS}


def arch_suffix(hp: dict) -> str:
    return (f"_arch-{hp['nn_width']}x{hp['nn_n_layers']}-{hp['activation']}"
            f"-Nel{hp['n_elements']}-p{hp['poly_degree']}")


def resolve_config(spec: CaseSpec, args) -> dict:
    """Return {hparams, schedule, source} for this run."""
    noise = getattr(args, "noise_level_f", None)
    if args.arch:
        if not args.schedule:
            raise SystemExit("--arch requires --schedule es|gb.")
        hp = parse_arch_spec(args.arch)
        return dict(hparams=hp, schedule=args.schedule,
                    source=dict(kind="user", arch=args.arch), suffix=arch_suffix(hp))

    wpath = Path(args.winners) if args.winners else paths.tables_dir(MODEL) / "winners.json"
    if not wpath.exists():
        raise SystemExit(f"{wpath} not found. Run 'python -m hpvpinn_capillarity.hpo.postprocess --model {MODEL} "
                         f"--winners' first, or pass --winners PATH, or use --arch.")
    payload = json.loads(wpath.read_text())

    if args.schedule:
        tag = f"{spec.case}_{args.schedule}{noise_tag(noise)}"
        how = f"--schedule {args.schedule}"
    else:
        key = group_key(spec.case, noise)
        tag = payload.get("selected", {}).get(key)
        if tag is None:
            raise SystemExit(f"No selected configuration for {key!r} in {wpath} "
                             f"(re-run the post-processing with --winners, or pass --schedule).")
        how = "selected (winners.json)"

    entry = payload["winners"].get(tag)
    if entry is None or entry.get("status") != "ok":
        raise SystemExit(f"No usable winner {tag!r} in {wpath}.")
    hp = {c: _CASTERS[c](entry["hparams"][c]) for c in HP_COLS}
    return dict(
        hparams=hp, schedule=entry["config"], suffix="",
        source=dict(kind="winners", how=how, winners_file=str(wpath), tag=tag,
                    hpo_trial=entry.get("trial"),
                    physics_converged=entry.get("physics_converged"),
                    admissible=entry.get("admissible"),
                    selection_reason=entry.get("selection_reason"),
                    hpo_metrics=entry.get("metrics"), hpo_recovered=entry.get("recovered")),
    )


def parse_seeds(args) -> list:
    if args.seeds:
        seeds = []
        for tok in args.seeds:
            if "-" in tok:
                a, b = tok.split("-")
                seeds.extend(range(int(a), int(b) + 1))
            else:
                seeds.append(int(tok))
        return seeds
    return list(range(args.seed_offset, args.seed_offset + args.n_seeds))


# ---------------------------------------------------------------------------
# Environment and timing
# ---------------------------------------------------------------------------

def environment_info() -> dict:
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    info = dict(
        hostname=socket.gethostname(), platform=platform.platform(),
        python=platform.python_version(), cpu_model=cpu, n_cpus=os.cpu_count(),
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
        env={k: os.environ.get(k) for k in ("XLA_FLAGS", "OMP_NUM_THREADS",
                                             "MKL_NUM_THREADS", "JAX_PLATFORMS")},
    )
    try:
        info["affinity_cpus"] = len(os.sched_getaffinity(0))
    except AttributeError:
        info["affinity_cpus"] = None
    try:
        import jax
        import jaxlib
        info.update(jax=jax.__version__, jaxlib=jaxlib.__version__,
                    jax_backend=jax.default_backend(),
                    x64=bool(jax.config.read("jax_enable_x64")))
    except Exception:
        pass
    return info


_COMPILE = dict(active=False, seconds=0.0, installed=False, available=False)


def _install_compile_listener() -> None:
    """Accumulate JAX tracing/lowering/compilation time while _COMPILE['active']."""
    if _COMPILE["installed"]:
        return
    _COMPILE["installed"] = True
    try:
        import jax.monitoring

        def _cb(event, duration, *a, **kw):
            if _COMPILE["active"] and str(event).startswith("/jax/core/compile/"):
                _COMPILE["seconds"] += float(duration)

        jax.monitoring.register_event_duration_secs_listener(_cb)
        _COMPILE["available"] = True
    except Exception as exc:
        warnings.warn(f"JAX compile-time listener unavailable ({exc!r}); "
                      "compile time will not be separated.")


# ---------------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------------

def _history_arrays(history) -> dict:
    out = {}
    for name in history._fields:
        if name == "diffusivity":
            continue
        a = np.asarray(getattr(history, name))
        out[name] = a.astype(np.float32) if a.dtype.kind == "f" else a
    diff = history.diffusivity
    for name in diff.__dataclass_fields__:
        p = getattr(diff, name)
        if hasattr(p, "hat"):
            out[f"param_{name}"] = np.asarray(p.value(), dtype=np.float64)
    return out


def _trainable_params(diffusivity) -> dict:
    return {n: float(getattr(diffusivity, n).value())
            for n in diffusivity.__dataclass_fields__
            if hasattr(getattr(diffusivity, n), "hat")}


def train_one_seed(m, spec: CaseSpec, cfg: dict, data: dict, seed: int, noise_level,
                   eval_every: int, grids: dict) -> tuple:
    """Train one seed through m.run_vpinn_trial; return (record, arrays dict)."""
    import jax
    import jax.numpy as jnp

    Base = m.VPINNTrainer
    init_params = inspect.signature(Base.__init__).parameters
    fit_params = inspect.signature(Base.fit).parameters
    track_eval = eval_every > 0 and "q_eval" in init_params and hasattr(m, "Q_EVAL")
    cap = {}

    class _CapturingTrainer(Base):
        def __init__(self, *a, **kw):
            if track_eval:
                kw.setdefault("q_eval", m.Q_EVAL)
            super().__init__(*a, **kw)

        def fit(self, *a, **kw):
            if track_eval and "eval_every" in fit_params:
                kw.setdefault("eval_every", eval_every)
            _COMPILE["seconds"], _COMPILE["active"] = 0.0, True
            t0 = time.perf_counter()
            try:
                out = super().fit(*a, **kw)
                jax.block_until_ready(out[1].total_loss)
            finally:
                _COMPILE["active"] = False
            cap["t_fit_s"] = time.perf_counter() - t0
            cap["t_compile_s"] = _COMPILE["seconds"] if _COMPILE["available"] else None
            cap["epochs"] = int(kw.get("epochs", a[0] if a else 0))
            cap["pinn"], cap["history"] = out
            return out

    warmup, transition = SCHEDULES[cfg["schedule"]]
    trial = SimpleNamespace(number=int(seed))       # run_vpinn_trial: init seed = trial.number
    takes_noise = "noise_level" in inspect.signature(m.run_vpinn_trial).parameters
    call_args = ((trial, dict(cfg["hparams"]), data, noise_level, warmup, transition)
                 if takes_noise else
                 (trial, dict(cfg["hparams"]), data, warmup, transition))
    m.VPINNTrainer = _CapturingTrainer
    try:
        t0 = time.perf_counter()
        _, metrics, diverged, recovered = m.run_vpinn_trial(*call_args)
        t_trial = time.perf_counter() - t0
    finally:
        m.VPINNTrainer = Base

    pinn, history = cap["pinn"], cap["history"]
    arrays = _history_arrays(history)
    t_compile = cap["t_compile_s"]
    t_exec = cap["t_fit_s"] - t_compile if t_compile is not None else None
    if t_exec is not None and t_exec <= 0:
        warnings.warn(f"Measured compile time ({t_compile:.2f} s) >= fit time "
                      f"({cap['t_fit_s']:.2f} s); execution time not separated for this seed.")
        t_exec = None
    frozen = np.asarray(arrays.get("ema_physics_frozen", np.zeros(1, bool))).astype(bool)

    record = dict(
        seed=int(seed), failed=bool(diverged), diverged=bool(diverged),
        config=cfg["hparams"], schedule=cfg["schedule"],
        warmup_epochs=warmup, transition_epochs=transition, noise_level=noise_level,
        held_out_tracking=bool(track_eval), eval_every=eval_every if track_eval else None,
        quadrature=dict(n_quad=getattr(m, "N_QUAD", None), q_eval=getattr(m, "Q_EVAL", None)),
        timing=dict(
            epochs=cap["epochs"], t_fit_s=cap["t_fit_s"], t_compile_s=t_compile,
            t_exec_s=t_exec,
            per_epoch_ms=(1e3 * t_exec / cap["epochs"]) if (t_exec and cap["epochs"]) else None,
            t_trial_s=t_trial, hpo_elapsed_s=metrics.get("elapsed_s"),
        ),
        ema_freeze_epoch=int(np.argmax(frozen)) if frozen.any() else None,
    )
    if diverged:
        return record, dict(history=arrays)

    diff = pinn.pde.diffusivity
    params = _trainable_params(diff)
    chi2_tr, chi2_te = metrics["chi2_nu_train"], metrics["chi2_nu_test"]
    record.update(
        metrics={k: v for k, v in metrics.items() if k != "elapsed_s"},
        obj=float(chi2_tr + chi2_te),          # current HPO objective
        physics_converged=bool(metrics["final_physics"] < m.PHYSICS_THRESHOLD),
        params=params,
    )
    if spec.has_ground_truth and spec.reference_params:
        record["rel_errors"] = {k: abs(params[k] - v) / abs(v)
                                for k, v in spec.reference_params.items() if k in params}

    theta_pred = np.asarray(jax.vmap(pinn.nn_model)(jnp.asarray(grids["phi_nd"])))
    D_curve = np.asarray(diff(jnp.asarray(grids["theta"])))
    curves = dict(phi_nd=grids["phi_nd"], theta_nd=theta_pred,
                  theta=grids["theta"], D=D_curve)
    # pointwise strong-form residual on the dense grid of the HPO (same ODE, same grid as frontx)
    from hpvpinn_capillarity.core.strong_form_residual import strong_form_residual
    phi_dense = jnp.asarray(m.PHI_DENSE)
    curves["residual_phi_nd"] = np.asarray(phi_dense)
    curves["residual"] = np.asarray(
        strong_form_residual(pinn.nn_model, pinn.pde, pinn.pde.scaling, phi_dense))
    return record, dict(history=arrays, curves=curves, pinn=pinn)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default))
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def run_seeds(spec: CaseSpec, args, cfg: dict, run_dir: Path) -> None:
    """Training phase: needs JAX. Skips seeds whose record.json exists."""
    import importlib
    m = importlib.import_module(spec.search_module)       # enables x64
    m.BACKENDS[MODEL]["loader"]()
    import equinox as eqx
    import jax.numpy as jnp

    if args.epochs != m.MAX_STEPS:
        warnings.warn(f"epochs={args.epochs} differs from the HPO budget ({m.MAX_STEPS}).")
        m.MAX_STEPS = args.epochs           # read by run_vpinn_trial at call time
    _install_compile_listener()

    data_path = args.data or str(paths.data_file(spec.default_data))
    shared = spec.prepare_shared(m, data_path, args.noise_level_f)
    data = m.BACKENDS[MODEL]["prepare"](shared)
    scaling = data["scaling"]

    # ---- case-level reference file (data, grids, reference D) ----
    th_i, th_b = float(scaling.theta_i), float(scaling.theta_b)
    s = np.linspace(*SE_RANGE, N_THETA_CURVE)
    grids = dict(phi_nd=np.linspace(0.0, 1.0, N_PHI_CURVE), theta=th_i + s * (th_b - th_i))
    mt, ms = data["measurements_train"], data["measurements_test"]
    theta_obs = np.concatenate([np.asarray(scaling.theta_inverse(mt.theta)),
                                np.asarray(scaling.theta_inverse(ms.theta))])
    ref = dict(
        theta=grids["theta"], phi_nd=grids["phi_nd"], theta_i=th_i, theta_b=th_b,
        phi_train_nd=np.asarray(mt.phi), theta_train_nd=np.asarray(mt.theta),
        phi_test_nd=np.asarray(ms.phi), theta_test_nd=np.asarray(ms.theta),
        theta_data_range=np.array([max(theta_obs.min(), grids["theta"][0]),
                                   min(theta_obs.max(), grids["theta"][-1])]),
    )
    if spec.reference_diffusivity is not None:
        ref["D_ref"] = np.asarray(spec.reference_diffusivity(m)(jnp.asarray(grids["theta"])))
    if spec.noise_free_curve is not None:
        o_c, th_c = spec.noise_free_curve(m, data_path)
        ref["phi_clean_nd"] = np.asarray(scaling.phi_forward(jnp.asarray(o_c)))
        ref["theta_clean_nd"] = np.asarray(scaling.theta_forward(jnp.asarray(th_c)))
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "reference.npz", **ref)

    env = environment_info()
    print(f"[env] {env['cpu_model']} | {env['n_cpus']} logical CPUs "
          f"(affinity {env['affinity_cpus']}) | jax {env.get('jax')} ({env.get('jax_backend')})")
    if env.get("jax_backend") not in (None, "cpu"):
        warnings.warn(f"JAX backend is {env.get('jax_backend')!r}; timings are meant for CPU.")

    seeds = parse_seeds(args)
    for k, seed in enumerate(seeds, 1):
        sdir = run_dir / "seeds" / f"seed_{seed:05d}"
        if (sdir / "record.json").exists() and not args.overwrite:
            print(f"[{k}/{len(seeds)}] seed {seed}: exists, skipping")
            continue
        print(f"\n[{k}/{len(seeds)}] seed {seed}: training {cfg['hparams']} "
              f"({cfg['schedule']}, {args.epochs} epochs) ...", flush=True)
        try:
            record, out = train_one_seed(m, spec, cfg, data, seed, args.noise_level_f,
                                         args.eval_every, grids)
        except Exception as exc:
            _write_json(sdir / "record.json", dict(seed=seed, failed=True, error=repr(exc)))
            print(f"  FAILED: {exc!r}")
            continue
        record.update(env=env, source=cfg["source"], data_path=data_path)
        sdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sdir / "history.npz", **out["history"])
        if "curves" in out:
            np.savez(sdir / "curves.npz", **out["curves"])
            eqx.tree_serialise_leaves(sdir / "model.eqx", out["pinn"])
        _write_json(sdir / "record.json", record)

        t = record["timing"]
        if record["failed"]:
            print(f"  diverged (NaN)  t_fit={t['t_fit_s']:.1f}s")
            continue
        mt_ = record["metrics"]
        par = "  ".join(f"{n}={v:.5g}" for n, v in record["params"].items())
        comp = f"{t['t_compile_s']:.1f}" if t["t_compile_s"] is not None else "n/a"
        print(f"  {par}")
        print(f"  chi2 tr/te={mt_['chi2_nu_train']:.3f}/{mt_['chi2_nu_test']:.3f}  "
              f"phys={mt_['final_physics']:.2e} ({'ok' if record['physics_converged'] else 'NOT converged'})  "
              f"t_fit={t['t_fit_s']:.1f}s (compile {comp}s)")


# ---------------------------------------------------------------------------
# Aggregation (NumPy + Matplotlib only)
# ---------------------------------------------------------------------------

def _load_seed_runs(run_dir: Path) -> list:
    runs = []
    for sdir in sorted((run_dir / "seeds").glob("seed_*")):
        rp = sdir / "record.json"
        if not rp.exists():
            continue
        rec = json.loads(rp.read_text())
        rec["_dir"] = sdir
        runs.append(rec)
    return runs


def choose_representative(runs: list, D: np.ndarray, theta: np.ndarray,
                          theta_range, rule: str) -> int:
    """Index (into runs) of the representative run.

    d-medoid   : run whose log10 D(theta) is closest (RMS) to the pointwise
                 median over runs, within the data-supported theta range.
    median-obj : run whose obj is closest to the median obj.
    """
    if rule == "median-obj":
        obj = np.array([r["obj"] for r in runs])
        return int(np.argmin(np.abs(obj - np.median(obj))))
    lo, hi = theta_range
    mask = (theta >= lo) & (theta <= hi)
    if mask.sum() < 2:
        mask = np.ones_like(theta, bool)
    L = np.log10(np.clip(D[:, mask], 1e-300, None))
    dist = np.sqrt(np.mean((L - np.median(L, axis=0)) ** 2, axis=1))
    return int(np.argmin(dist))


def _stats(values) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return dict(n=0, mean=None, std=None, median=None, min=None, max=None)
    return dict(n=int(v.size), mean=float(v.mean()),
                std=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                median=float(np.median(v)), min=float(v.min()), max=float(v.max()))


def hpo_cost(spec: CaseSpec, noise_level) -> dict:
    """Total recorded training time of the HPO campaign for this case (both
    schedules; Stage A search and Stage B replication), from the JSONLs."""
    out = {}
    base = read_constant(paths.module_source(spec.search_module), "BASE_RESULTS_DIR",
                         f"results_{spec.case}")
    for sched in SCHEDULES:
        nt = noise_tag(noise_level)
        fname = f"hparam_results{nt}.jsonl" if nt else f"hparam_results_{spec.case}.jsonl"
        path = paths.model_dir(MODEL) / f"{base}_{sched}{nt}" / fname
        if not path.exists():
            out[sched] = dict(available=False, path=str(path))
            continue
        per_stage = {}
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                st = r.get("stage", "search")
                d = per_stage.setdefault(st, dict(n_trials=0, total_s=0.0, n_with_time=0))
                d["n_trials"] += 1
                if r.get("elapsed_s") is not None:
                    d["total_s"] += float(r["elapsed_s"])
                    d["n_with_time"] += 1
        out[sched] = dict(available=True, path=str(path), stages=per_stage)
    return out


# ---- figures ---------------------------------------------------------------

def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "text.usetex":         shutil.which("latex") is not None,
        "text.latex.preamble": r"\usepackage{amsbsy}",
        "font.family":         "serif",
        "font.serif":          ["Computer Modern Roman"],
        "axes.labelsize":      13,
        "xtick.labelsize":     10,
        "ytick.labelsize":     10,
        "legend.fontsize":     9,
    })
    import matplotlib.pyplot as plt
    return matplotlib, plt


BAND_COLOR = "#c6dbef"    # opaque light blue (PostScript has no transparency)
RANGE_COLOR = "#eeeeee"   # data-supported theta range


def _pct(matplotlib) -> str:
    return r"\%" if matplotlib.rcParams["text.usetex"] else "%"


def _save(fig, plt, path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format=path.suffix[1:], bbox_inches="tight", dpi=300)
    print(f"[saved] {path}")
    if show:
        plt.show()
    plt.close(fig)


def _minmax_decimate(x, y, n_bins=3000):
    """Per-bin min/max (keeps spikes visible, ~30x fewer vertices)."""
    x, y = np.asarray(x), np.asarray(y, float)
    if len(y) <= 2 * n_bins:
        return x, y
    edges = np.linspace(0, len(y), n_bins + 1).astype(int)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = y[a:b]
        ok = np.isfinite(seg)
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        for i in sorted({a + idx[np.argmin(seg[ok])], a + idx[np.argmax(seg[ok])]}):
            xs.append(x[i]); ys.append(y[i])
    return np.asarray(xs), np.asarray(ys)


def _line(ax, x, y, log=True, **kw):
    xd, yd = _minmax_decimate(x, y)
    return (ax.semilogy if log else ax.plot)(xd, yd, **kw)


def _epoch_axis(ax, n):
    step = 50_000 if n > 100_000 else max(n // 4, 1)
    ticks = np.arange(step, n + 1, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 1000:d}k" if t >= 1000 else str(t) for t in ticks])
    ax.set_xlim(0, n)
    ax.set_xlabel("Epochs")
    ax.grid(True, alpha=0.3)


def _letter(ax, s):
    ax.text(0.03, 0.95, f"{s})", transform=ax.transAxes, fontsize=12, va="top", ha="left")


def _latex_number(v: float, sig: int = 4) -> str:
    """Number for math mode: plain for 1e-3 <= |v| < 1e4, else m\times10^{e}."""
    if v == 0 or 1e-3 <= abs(v) < 1e4:
        return f"{v:.{sig}g}"
    mant, exp = f"{v:.{sig - 1}e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    return rf"{mant}\times 10^{{{int(exp)}}}"


def plot_losses(spec, rep_hist: dict, pool_hists: list, path: Path, lambda_bc,
                ema_threshold, show=False):
    matplotlib, plt = _setup_matplotlib()
    import matplotlib.gridspec as gridspec

    h = rep_hist
    n = len(h["total_loss"])
    ep = np.arange(n)
    params = [k[len("param_"):] for k in h if k.startswith("param_")]
    kw = dict(linewidth=0.9)

    n_par = max(len(params), 1)
    fig = plt.figure(figsize=(max(14.0, 3.3 * n_par), 8.2))
    gt = gridspec.GridSpec(1, 3, figure=fig, left=0.07, right=0.98, top=0.96, bottom=0.57, wspace=0.28)
    gb = gridspec.GridSpec(1, n_par, figure=fig, left=0.07, right=0.98,
                           top=0.46, bottom=0.07, wspace=0.32 if n_par <= 4 else 0.45)
    frozen = np.asarray(h.get("ema_physics_frozen", np.zeros(1, bool))).astype(bool)
    f_ep = int(np.argmax(frozen)) if frozen.any() else None

    ax = fig.add_subplot(gt[0, 0])
    _line(ax, ep, h["data_loss"], color="tab:red", label=r"$\mathcal{L}_{\mathrm{data}}$", **kw)
    _line(ax, ep, h["physics_loss"], color="k", label=r"$\mathcal{L}_{\mathrm{phys}}$", **kw)
    if "ema_physics" in h:
        ema = np.where(h["ema_physics"] > 0, h["ema_physics"], np.nan)
        _line(ax, ep, ema, color="tab:green", label=r"$\hat{\mathcal{L}}_{\mathrm{phys}}$ (EMA)", **kw)
    if ema_threshold is not None:
        ax.axhline(ema_threshold, color="gray", linestyle="--", linewidth=1.0)
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    _letter(ax, "a")

    ax = fig.add_subplot(gt[0, 1])
    _line(ax, ep, h["pde_loss"], color="tab:blue",
          label=r"$\mathcal{L}_{\mathrm{pde}}^{\mathrm{train}}$", **kw)
    if "pde_loss_eval" in h and np.isfinite(h["pde_loss_eval"]).any():
        pe = np.asarray(h["pde_loss_eval"], float)
        ok = np.isfinite(pe)
        _line(ax, ep[ok], pe[ok], color="tab:orange",
              label=r"$\mathcal{L}_{\mathrm{pde}}^{\mathrm{eval}}$", **kw)
    bc = "" if lambda_bc is None else rf" ($\lambda_{{\mathrm{{bc}}}}={lambda_bc:g}$)"
    _line(ax, ep, h["dirichlet_loss"], color="tab:purple", label=r"$\mathcal{L}_{\mathrm{D}}$" + bc, **kw)
    _line(ax, ep, h["derivative_loss"], color="tab:brown", label=r"$\mathcal{L}_{\mathrm{N}}$", **kw)
    ax.legend(loc="upper right")
    _letter(ax, "b")

    ax = fig.add_subplot(gt[0, 2])
    _line(ax, ep, h["lambda_phys"], color="k", **kw)
    ax.set_ylabel(r"$\lambda_{\mathrm{phys}}$")
    _letter(ax, "c")

    for a in fig.axes:
        _epoch_axis(a, n)
        if f_ep is not None:
            a.axvline(f_ep, color="k", linestyle=":", linewidth=1.0)

    ref = spec.reference_params or {}
    for i, name in enumerate(params):
        ax = fig.add_subplot(gb[0, i])
        trajs = [ph[f"param_{name}"] for ph in pool_hists if f"param_{name}" in ph]
        allv = np.concatenate([np.ravel(t) for t in trajs + [h[f"param_{name}"]]])
        allv = allv[np.isfinite(allv)]
        # log axis for positive parameters spanning more than a factor of 20
        # (e.g. Ks/alpha starting from its initial guess), linear otherwise
        use_log = allv.size > 0 and allv.min() > 0 and allv.max() / allv.min() > 20
        for tr in trajs:
            _line(ax, np.arange(len(tr)), tr, log=use_log, color="0.7", linewidth=0.6)
        _line(ax, ep, h[f"param_{name}"], log=use_log, color="k", linewidth=1.1)
        if name in ref:
            ax.axhline(ref[name], color="tab:blue", linestyle="--", linewidth=1.2,
                       label=rf"{spec.reference_short} $= {_latex_number(ref[name])}$")
            ax.legend(loc="best")
        ax.set_ylabel(spec.param_labels.get(name, name))
        _epoch_axis(ax, n)
        if f_ep is not None:
            ax.axvline(f_ep, color="k", linestyle=":", linewidth=1.0)
        _letter(ax, "defghijk"[i])
    _save(fig, plt, path, show)


def plot_profile(spec, ref: dict, rep_curve: dict, band, path: Path, schedule, noise_level,
                 show=False, zoom=None):
    matplotlib, plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    draw_profile(ax, ref, rep_curve, band, schedule, noise_level, zoom=zoom)
    _save(fig, plt, path, show)


def plot_diffusivity(spec, ref: dict, D_rep, D_pool, path: Path, schedule, show=False):
    matplotlib, plt = _setup_matplotlib()
    theta = ref["theta"]
    has_ref = "D_ref" in ref
    fig, axes = plt.subplots(1, 2 if has_ref else 1, figsize=(10.5 if has_ref else 5.2, 4.4),
                             squeeze=False)
    axes = axes[0]
    lo, hi = ref["theta_data_range"]
    band = D_pool is not None and len(D_pool) >= 2
    # shade the data-supported range only if it is narrower than the plotted grid
    span = (hi - lo) < 0.98 * (theta[-1] - theta[0])

    ax = axes[0]
    if span:
        ax.axvspan(lo, hi, color=RANGE_COLOR, zorder=0, label="Data range")
    if band:
        ax.fill_between(theta, D_pool.min(0), D_pool.max(0), color=BAND_COLOR, linewidth=0,
                        label=rf"Seeds (min--max, $N={len(D_pool)}$)", zorder=1)
    if has_ref:
        ax.semilogy(theta, ref["D_ref"], "k--", linewidth=1.6, label=spec.reference_label, zorder=3)
    ax.semilogy(theta, D_rep, "b-", linewidth=1.6, label=rf"$hp$-VPINN-{schedule}", zorder=4)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(rf"$D(\theta)$ ({spec.D_units})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    if has_ref:
        _letter(ax, "a")
        ax = axes[1]
        if span:
            ax.axvspan(lo, hi, color=RANGE_COLOR, zorder=0)
        rel = lambda D: 100.0 * (np.asarray(D) / ref["D_ref"] - 1.0)
        if band:
            R = rel(D_pool)
            ax.fill_between(theta, R.min(0), R.max(0), color=BAND_COLOR, linewidth=0, zorder=1)
        ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0, zorder=2)
        ax.plot(theta, rel(D_rep), "b-", linewidth=1.6, zorder=3)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(rf"$D/D_{{\mathrm{{ref}}}} - 1$ [{_pct(matplotlib)}]")
        ax.grid(True, alpha=0.3)
        _letter(ax, "b")
    fig.tight_layout()
    _save(fig, plt, path, show)


# ---- LaTeX tables -----------------------------------------------------------

def _num(v, fmt=".4g"):
    return "---" if v is None else rf"\num{{{v:{fmt}}}}"


def latex_params_table(spec, summary: dict) -> str:
    """Parameter statistics over converged seeds, then metric statistics."""
    pstats = summary["params"]
    has_ref = bool(spec.reference_params)
    has_truth = has_ref and spec.has_ground_truth   # bias only against ground truth
    head = ["Parameter"] + ([("True" if has_truth else "Ref.")] if has_ref else []) \
        + ["Mean", "Std", "Min", "Max"] + ([r"Bias [\%]"] if has_truth else [])
    nc = len(head)
    n_conv, n_tot = summary["n_converged"], summary["n_completed"]
    L = [r"\begin{tabular}{l" + "c" * (nc - 1) + "}", r"\toprule",
         rf"\multicolumn{{{nc}}}{{l}}{{{summary['config_latex']} \quad "
         rf"$N={n_conv}$ qualified of {n_tot} seeds}} \\", r"\midrule",
         " & ".join(head) + r" \\", r"\midrule"]
    for name, st in pstats.items():
        row = [spec.param_labels.get(name, name)]
        if has_ref:
            row.append(_num(spec.reference_params.get(name)))
        row += [_num(st["mean"]), _num(st["std"], ".3g"), _num(st["min"]), _num(st["max"])]
        if has_truth:
            tv = spec.reference_params.get(name)
            b = None if (tv in (None, 0) or st["mean"] is None) else 100 * (st["mean"] - tv) / abs(tv)
            row.append("---" if b is None else rf"\num[retain-explicit-plus]{{{b:+.3g}}}")
        L.append("  " + " & ".join(row) + r" \\")
    L += [r"\midrule", rf"\multicolumn{{{nc}}}{{l}}{{Metrics (mean $\pm$ std)}} \\"]
    labels = {
        "chi2_nu_train": r"$\chi^2_{\nu,\mathrm{train}}$", "chi2_nu_test": r"$\chi^2_{\nu,\mathrm{test}}$",
        "obj": r"$\mathrm{obj}$", "final_physics": r"$\mathcal{L}_{\mathrm{phys}}^{\mathrm{train}}$",
        "final_pde": r"$\mathcal{L}_{\mathrm{pde}}^{\mathrm{train}}$",
        "pde_loss_eval": r"$\mathcal{L}_{\mathrm{pde}}^{\mathrm{eval}}$",
        "strong_form_rms": r"$r_{\mathrm{RMS}}^{\mathrm{strong}}$",
    }
    for k, lab in labels.items():
        st = summary["metrics"].get(k)
        if not st or st["n"] == 0:
            continue
        fmt = ".3f" if k in ("chi2_nu_train", "chi2_nu_test", "obj") else ".2e"
        cell = rf"{_num(st['mean'], fmt)} $\pm$ {_num(st['std'], '.2g')}"
        L.append(rf"  {lab} & \multicolumn{{{nc - 1}}}{{l}}{{{cell}}} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def latex_timing_table(summary: dict) -> str:
    t, env, hpo = summary["timing"], summary["environment"], summary["hpo_cost"]
    def mr(st, fmt):
        if not st or st["n"] == 0:
            return "---"
        return rf"{_num(st['median'], fmt)} [{_num(st['min'], fmt)}, {_num(st['max'], fmt)}]"
    cpu = (env.get("cpu_model") or "local PC").replace("(R)", "").replace("(TM)", "")
    L = [r"\begin{tabular}{llc}", r"\toprule",
         r"Quantity & Hardware & Value \\", r"\midrule",
         rf"JIT compilation [s] & {cpu} & {mr(t['t_compile_s'], '.1f')} \\",
         rf"Training, \num{{{t['epochs']}}} epochs [s] & {cpu} & {mr(t['t_fit_s'], '.1f')} \\",
         rf"Per epoch [ms] & {cpu} & {mr(t['per_epoch_ms'], '.3g')} \\"]
    for sched, d in hpo.items():
        if not d.get("available"):
            continue
        for stage, s in d["stages"].items():
            name = "search" if stage == "search" else "replication"
            L.append(rf"HPO {sched} {name} ({s['n_trials']} trials) [h] & cluster & "
                     rf"{_num(s['total_s'] / 3600, '.3g')} \\")
    tot = summary.get("hpo_total_h")
    if tot is not None:
        L.append(rf"HPO total (all studies of the case) [h] & cluster & {_num(tot, '.3g')} \\")
    L += [r"\bottomrule", r"\end{tabular}",
          rf"% Median [min, max] over {t['t_fit_s']['n']} seeds, run sequentially. "
          rf"Local: {env.get('cpu_model')}, {env.get('n_cpus')} logical CPUs, jax {env.get('jax')}."]
    return "\n".join(L) + "\n"


def aggregate(spec: CaseSpec, args, cfg: dict, run_dir: Path,
              tag: str) -> Optional[dict]:
    runs = _load_seed_runs(run_dir)
    if not (run_dir / "reference.npz").exists() or not runs:
        print(f"No completed seeds in {run_dir}; nothing to aggregate.")
        return None
    ref = dict(np.load(run_dir / "reference.npz"))
    figs = Path(args.figures_dir) if args.figures_dir else paths.figures_dir(MODEL)
    figs.mkdir(parents=True, exist_ok=True)
    tabs = Path(args.tables_dir) if args.tables_dir else paths.tables_dir(MODEL)
    tabs.mkdir(parents=True, exist_ok=True)

    done = [r for r in runs if not r.get("failed")]
    adm = read_constant(paths.module_source("hpvpinn_capillarity.hpo.postprocess"), "ADMISSIBLE_PHYSICS", ADMISSIBLE_PHYSICS)
    cal = read_constant(paths.module_source("hpvpinn_capillarity.hpo.postprocess"), "CALIBRATED_OBJ", CALIBRATED_OBJ)
    gate = [is_qualified(r, adm, cal) for r in done]
    conv = [r for r, (ok, _) in zip(done, gate) if ok]
    if any(k != "pde_loss_eval" for _, k in gate):
        warnings.warn("Some seeds have no pde_loss_eval; the admissibility gate fell back "
                      "to final_physics for them.")
    n_fail = len(runs) - len(done)
    print(f"\n[aggregate] {tag}: {len(runs)} seeds, {n_fail} failed/diverged, "
          f"{len(conv)} qualified (pde_loss_eval < {adm:g}, obj < {cal:g})")
    epochs = {r["timing"]["epochs"] for r in done}
    if len(epochs) > 1:
        warnings.warn(f"Seeds with different training budgets in {run_dir}: {sorted(epochs)}")
    pool = conv
    if not pool:
        warnings.warn("No converged seed: statistics and figures use all completed seeds.")
        pool = done
    if not pool:
        print("No completed seed to aggregate.")
        return None

    curves = [dict(np.load(r["_dir"] / "curves.npz")) for r in pool]
    D_pool = np.stack([c["D"] for c in curves])
    th_pool = np.stack([c["theta_nd"] for c in curves])
    i_rep = choose_representative(pool, D_pool, ref["theta"], ref["theta_data_range"],
                                  args.representative)
    rep = pool[i_rep]
    print(f"[aggregate] representative run ({args.representative}): seed {rep['seed']}")

    # ---- statistics ----
    names = list(pool[0]["params"])
    summary = dict(
        case=spec.case, tag=tag, schedule=cfg["schedule"], config=cfg["hparams"],
        config_latex=(rf"{cfg['hparams']['nn_width']}$\times${cfg['hparams']['nn_n_layers']}, "
                      rf"{cfg['hparams']['activation']}, $N_{{\mathrm{{el}}}}={cfg['hparams']['n_elements']}$, "
                      rf"$p={cfg['hparams']['poly_degree']}$, {cfg['schedule']}"),
        source=cfg["source"], noise_level=args.noise_level_f,
        seeds=[r["seed"] for r in runs], n_completed=len(done), n_failed=n_fail,
        n_converged=len(conv), stats_over="converged" if conv else "completed (none converged)",
        representative=dict(rule=args.representative, seed=rep["seed"]),
        params={n: _stats([r["params"][n] for r in pool]) for n in names},
        metrics={k: _stats([r["metrics"].get(k) for r in pool])
                 for k in ("chi2_nu_train", "chi2_nu_test", "final_physics", "final_pde",
                           "final_dirichlet", "final_derivative", "pde_loss_eval",
                           "strong_form_rms")},
    )
    summary["metrics"]["obj"] = _stats([r["obj"] for r in pool])
    if spec.has_ground_truth:
        summary["rel_errors"] = {n: _stats([r.get("rel_errors", {}).get(n) for r in pool])
                                 for n in (spec.reference_params or {})}
    summary["timing"] = dict(epochs=sorted(epochs)[0] if epochs else None, **{
        k: _stats([r["timing"].get(k) for r in done])
        for k in ("t_compile_s", "t_fit_s", "t_exec_s", "per_epoch_ms")})
    summary["environment"] = done[0].get("env", {})
    summary["quadrature"] = done[0].get("quadrature")
    if len({json.dumps(r.get("quadrature"), sort_keys=True) for r in done}) > 1:
        warnings.warn("Seeds in this run directory were trained with different quadrature orders.")
    envs = {(r.get("env", {}).get("hostname"), r.get("env", {}).get("cpu_model")) for r in done}
    if len(envs) > 1:
        warnings.warn(f"Seeds were run on different machines: {envs}; timings are not comparable.")
    summary["hpo_cost"] = hpo_cost(spec, args.noise_level_f)
    tot = [s["total_s"] for d in summary["hpo_cost"].values() if d.get("available")
           for s in d["stages"].values()]
    summary["hpo_total_h"] = sum(tot) / 3600 if tot else None
    summary["ema_freeze_epochs"] = [r.get("ema_freeze_epoch") for r in pool]

    # ---- figures ----
    rep_hist = dict(np.load(rep["_dir"] / "history.npz"))
    pool_hists = [dict(np.load(r["_dir"] / "history.npz")) for r in pool]
    # 5e-8 is the value hard-coded in _train_vpinn when the constant is absent
    ema_thr = read_constant(paths.module_source("hpvpinn_capillarity.core.train_vpinn"),
                            "EMA_PHYSICS_THRESHOLD", 5e-8)
    lambda_bc = read_constant(paths.module_source(spec.search_module), "LAMBDA_BC")
    plot_losses(spec, rep_hist, pool_hists, figs / f"final_losses_{tag}.eps",
                lambda_bc, ema_thr, args.show)
    band = band_from_pool(th_pool, args.band)
    plot_profile(spec, ref, curves[i_rep], band, figs / f"final_profile_{tag}.eps",
                 cfg["schedule"], args.noise_level_f, args.show,
                 zoom=ZOOM_BY_CASE.get(spec.case))
    plot_diffusivity(spec, ref, D_pool[i_rep], D_pool, figs / f"final_diffusivity_{tag}.eps",
                     cfg["schedule"], args.show)
    summary["profile_band_max_abs"] = float(np.max(th_pool.max(0) - th_pool.min(0)))

    # ---- D(theta) curves for the cross-case figure ----
    cols = [("theta", ref["theta"]), ("D_representative", D_pool[i_rep]),
            ("D_median", np.median(D_pool, 0)), ("D_min", D_pool.min(0)), ("D_max", D_pool.max(0))]
    if "D_ref" in ref:
        cols.append(("D_ref", ref["D_ref"]))
    dpath = tabs / f"final_diffusivity_{tag}.csv"
    np.savetxt(dpath, np.column_stack([c for _, c in cols]), delimiter=",", fmt="%.10e",
               header=",".join(n for n, _ in cols)
               + f"\n# data-supported theta range: {ref['theta_data_range'][0]:.6g} "
                 f"{ref['theta_data_range'][1]:.6g}; N={len(pool)}; representative seed {rep['seed']}")
    print(f"[saved] {dpath}")

    # ---- tables and summary ----
    for name, text in ((f"latex_final_params_{tag}.tex", latex_params_table(spec, summary)),
                       (f"latex_final_timing_{tag}.tex", latex_timing_table(summary))):
        (tabs / name).write_text(text)
        print(f"[saved] {tabs / name}")
    _write_json(tabs / f"final_summary_{tag}.json", summary)
    print(f"[saved] {tabs / f'final_summary_{tag}.json'}")

    # ---- console digest ----
    print(f"\n  Parameters over {len(pool)} seeds (mean +- std [min, max]):")
    for n, st in summary["params"].items():
        rv = (spec.reference_params or {}).get(n)
        extra = f"   {spec.reference_short} {rv:.5g}" if rv is not None else ""
        print(f"    {n:8s} {st['mean']:.5g} +- {st['std']:.2g}  [{st['min']:.5g}, {st['max']:.5g}]{extra}")
    t = summary["timing"]
    if t["t_fit_s"]["n"]:
        comp = t["t_compile_s"]
        print(f"  Training time ({t['epochs']} epochs): median {t['t_fit_s']['median']:.1f} s "
              f"[{t['t_fit_s']['min']:.1f}, {t['t_fit_s']['max']:.1f}]"
              + (f", of which compile median {comp['median']:.1f} s" if comp["n"] else ""))
    if summary["hpo_total_h"] is not None:
        print(f"  HPO campaign (cluster, recorded training time): {summary['hpo_total_h']:.2f} h")
    return summary


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def build_parser(spec: CaseSpec) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"Final hp-VPINN runs for {spec.label}: multi-seed training of the "
                    f"selected configuration, figures, tables and timing.")
    if spec.noise_levels:
        p.add_argument("--noise-level", required=True, choices=[str(s) for s in spec.noise_levels])
    g = p.add_argument_group("configuration (default: winners.json 'selected')")
    g.add_argument("--schedule", choices=list(SCHEDULES), default=None,
                   help="Use this schedule's winner instead of the selected one; "
                        "required with --arch.")
    g.add_argument("--arch", default=None,
                   help='User-defined configuration, e.g. "W=16,L=2,act=tanh,Nel=30,p=5".')
    g.add_argument("--winners", default=None,
                   help=f"winners.json (default: <results>/{MODEL}/tables/winners.json).")
    s = p.add_argument_group("seeds")
    s.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    s.add_argument("--seed-offset", type=int, default=DEFAULT_SEED_OFFSET)
    s.add_argument("--seeds", nargs="+", default=None, help="Explicit seeds, e.g. 1000-1009 or 1000 1003.")
    p.add_argument("--epochs", type=int, default=None, help="Default: the HPO budget (MAX_STEPS).")
    p.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY,
                   help="Stride of the held-out PDE-loss tracking (0 disables it).")
    p.add_argument("--data", default=None, help=f"Default: <repo>/{spec.default_data}.")
    p.add_argument("--representative", choices=["d-medoid", "median-obj"], default="d-medoid")
    p.add_argument("--band", choices=["minmax", "p5-95"], default="minmax",
                   help="Seed envelope of the profile figure: pointwise min-max (default) "
                        "or 5th-95th percentiles.")
    p.add_argument("--figures-dir", default=None, help=f"Figures. Default: <results>/{MODEL}/Figures.")
    p.add_argument("--tables-dir", default=None, help=f"Tables, summaries and curves. Default: <results>/{MODEL}/tables.")
    p.add_argument("--aggregate-only", action="store_true",
                   help="No training: rebuild figures/tables from the saved seeds.")
    p.add_argument("--show-config", action="store_true", help="Print the configuration and exit.")
    p.add_argument("--overwrite", action="store_true", help="Retrain seeds that already exist.")
    p.add_argument("--show", action="store_true")
    return p


def main(spec: CaseSpec, argv=None) -> None:
    args = build_parser(spec).parse_args(argv)
    args.noise_level_f = float(args.noise_level) if spec.noise_levels else None
    cfg = resolve_config(spec, args)

    max_steps = read_constant(paths.module_source(spec.search_module), "MAX_STEPS")
    if args.epochs is None:
        if max_steps is None:
            raise SystemExit(f"Could not read MAX_STEPS from {spec.search_module}.py; pass --epochs.")
        args.epochs = int(max_steps)
    tag = run_tag(spec.case, cfg["schedule"], args.noise_level_f, cfg["suffix"],
                  args.epochs, max_steps)
    run_dir = paths.model_dir(MODEL) / "final_runs" / tag

    print("=" * 72)
    print(f"{spec.label} -- final runs, tag {tag}")
    print(f"  configuration : {cfg['hparams']}  schedule={cfg['schedule']}")
    src = cfg["source"]
    if src["kind"] == "winners":
        print(f"  source        : {src['how']} -> {src['tag']} (HPO trial {src['hpo_trial']}; "
              f"physics converged: {src['physics_converged']})")
        if src.get("selection_reason"):
            print(f"                  selection: {src['selection_reason']}")
    else:
        print(f"  source        : user-defined --arch {src['arch']}")
    print(f"  seeds         : {parse_seeds(args)}   epochs={args.epochs}")
    print(f"  run dir       : {run_dir}")
    print("=" * 72)
    if src["kind"] == "winners" and src.get("physics_converged") is False:
        warnings.warn("The HPO winner of this configuration did not reach the physics threshold.")
    if args.show_config:
        return

    if not args.aggregate_only:
        run_seeds(spec, args, cfg, run_dir)
    aggregate(spec, args, cfg, run_dir, tag)

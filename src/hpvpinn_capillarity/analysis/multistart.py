"""
multistart.py
=============
Multi-start campaign over the INITIAL VALUES of the trainable hydraulic
parameters (SI sec:si_multistart), for the frozen hp-VPINN winners.

One run = one (block, start) pair = one SLURM array task. Everything else
(data, split, schedule, budget, metrics, admissibility gate) is exactly that
of final_run_common.py / the HPO: each task calls final_run_common.run_seeds()
for a single seed after overriding the module constants KSALPHA_INIT, M_INIT,
... of hparam_search_<case>.py.

Place this file in <repo>/hp_vpinn/ next to final_run_common.py. Run all commands
from the repo root (the SLURM script does so).

Workflow
--------
  1. python hp_vpinn/multistart.py manifest     # writes <repo>/multistart/manifest.csv (no JAX)
  2. sbatch --array=0-<N-1>%100 slurm_run_multistart.sh
  3. python hp_vpinn/multistart.py aggregate    # tables, figures, console digest (no JAX)

Smoke test (2000 epochs, separate run directory with suffix _ep2000; ignored by aggregate):
  python hp_vpinn/multistart.py run --task-id 0 --epochs 2000

Outputs
-------
  <repo>/hp_vpinn/final_runs/<tag>_multistart/run_<id>/seeds/seed_<seed>/...   (as final runs)
  <repo>/multistart/manifest.csv
  <repo>/multistart/summary_<block>.csv, latex_multistart_<block>.tex,
  <repo>/multistart/multistart_<block>.eps, multistart_summary.json

Requirement for Case 1
----------------------
hparam_search_grenoblesand.py hard-codes the initial values in _vpinn_build_pinn.
Replace them by module constants (behaviour unchanged):
    KSALPHA_INIT = 1.0
    M_INIT       = 0.5
    ...  KsAlpha = LowerBoundedParam(value=KSALPHA_INIT, min=0.0),
         m       = BoundedParam(value=M_INIT, min=0.0, max=1.0),
The 'run' command aborts with a message if a constant is missing, and checks
after every run that the parameter value at epoch 0 equals the requested start.
"""

import argparse
import csv
import glob
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from hpvpinn_capillarity import paths

_CASE_PKG = "hpvpinn_capillarity.final_runs."     # the "module" column of the manifest holds the short name

# ---------------------------------------------------------------------------
# CAMPAIGN DEFINITION  (edit here)
# ---------------------------------------------------------------------------

SEED_BASE = 5000          # network seed of run <id> is SEED_BASE + id (never in the HPO / final-run ranges)
RNG_SEED = 20261005       # Latin-hypercube generator, fixed for reproducibility

# n_random: random starts (Latin hypercube). Deterministic starts are added on top:
#   'default'   the fixed initial guess used in the HPO (control: new network seed only)
#   'reference' SPEC.reference_params (true values in Case 1, literature set in Case 2)
#   any key of EXTRA_STARTS[module] (e.g. a second literature set)
BLOCKS = [
    dict(name="case1_1pct_gb", module="grenoblesand", noise=0.01, schedule="gb",
         n_random=23, det=["default", "reference"]),
    dict(name="case1_2pct_gb", module="grenoblesand", noise=0.02, schedule="gb",
         n_random=23, det=["default", "reference"]),
    dict(name="case2_es", module="vangenuchten", noise=None, schedule="es",
         n_random=22, det=["default", "reference", "ref_B"]),
    dict(name="case2_gb", module="vangenuchten", noise=None, schedule="gb",
         n_random=22, det=["default", "reference", "ref_B"]),
    # dict(name="case3_<sched>", module="letd", noise=None, schedule="...", n_random=25,
    #      det=["default", "reference"]),            # optional; add RANGES["letd"] below
]

# Sampling ranges of the random starts: (scale, low, high). TODO: set to the
# ranges of tab:init_case* of the SI (the values below are placeholders).
# Starts must lie inside the bounds of the Param objects (m in (0,1),
# theta_r in (0,THETA_I), KsAlpha > 0).
RANGES = {
    "grenoblesand": {"KsAlpha": ("log", 1e-2, 1e4),     # cm^2/h   (true ~356; HPO start 1.0)
                     "m":       ("lin", 0.05, 0.95)},   # true 0.5096; HPO start 0.5
    "vangenuchten": {"KsAlpha": ("log", 1e-8, 1e-3),    # m^2/s    (ref 2.08e-6; HPO start 1e-4)
                     "m":       ("lin", 0.05, 0.95),
                     "l":       ("lin", -2.0, 5.0),     # FreeParam; ref 2.344; HPO start 1.0
                     "theta_r": ("lin", 1e-3, 2.4e-2)},  # bound (0, 0.025); ref 0.00494; HPO start 0.02
}

# Fallback HPO start if the constant cannot be read from the search module.
DEFAULT_START_FALLBACK = {"grenoblesand": {"KsAlpha": 1.0, "m": 0.5}}

# Extra deterministic starts (e.g. the second literature parameter set of Case 2).
EXTRA_STARTS = {
    "vangenuchten": {"ref_B": None},   # TODO: {"KsAlpha": ..., "m": ..., "l": ..., "theta_r": ...}; None -> skipped
}

INIT_RTOL = 0.05          # tolerance of the epoch-0 check of the requested start


def init_const_name(param: str) -> str:
    """KsAlpha -> KSALPHA_INIT, theta_r -> THETA_R_INIT (convention of hparam_search_*.py)."""
    return param.upper() + "_INIT"


def _out_dir():
    d = paths.multistart_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def lhs(n, d, rng):
    u = np.empty((n, d))
    for j in range(d):
        u[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return u


def cmd_manifest(a):
    from hpvpinn_capillarity.final_runs.common import read_constant
    rows, rng = [], np.random.default_rng(RNG_SEED)
    for b in BLOCKS:
        mod = importlib.import_module(_CASE_PKG + b["module"])
        spec = mod.SPEC
        rng_b = RANGES[b["module"]]
        params = list(rng_b)
        src = paths.module_source(spec.search_module)
        default = {p: read_constant(src, init_const_name(p),
                                    DEFAULT_START_FALLBACK.get(b["module"], {}).get(p))
                   for p in params}
        if any(v is None for v in default.values()):
            raise SystemExit(f"Cannot determine the default start of {b['module']}: {default}")
        starts = []
        for k in b["det"]:
            if k == "default":
                starts.append((k, default))
            elif k == "reference":
                starts.append((k, {p: float(spec.reference_params[p]) for p in params}))
            else:
                v = EXTRA_STARTS.get(b["module"], {}).get(k)
                if v is None:
                    print(f"[warn] {b['name']}: deterministic start {k!r} not defined, skipped")
                else:
                    starts.append((k, {p: float(v[p]) for p in params}))
        u = lhs(b["n_random"], len(params), rng)
        for i in range(b["n_random"]):
            vals = {}
            for j, p in enumerate(params):
                sc, lo, hi = rng_b[p]
                vals[p] = float(np.exp(np.log(lo) + u[i, j] * (np.log(hi) - np.log(lo)))
                                if sc == "log" else lo + u[i, j] * (hi - lo))
            starts.append(("random", vals))
        for kind, vals in starts:
            rows.append(dict(block=b["name"], module=b["module"],
                             noise="" if b["noise"] is None else b["noise"],
                             schedule=b["schedule"], kind=kind, init=json.dumps(vals)))
    for i, r in enumerate(rows):
        r["id"], r["seed"] = i, SEED_BASE + i
    path = _out_dir() / "manifest.csv"
    cols = ["id", "block", "module", "noise", "schedule", "kind", "seed", "init"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"[saved] {path}: {len(rows)} runs  ({dict(Counter(r['block'] for r in rows))})")
    print(f"submit with: sbatch --array=0-{len(rows) - 1}%100 slurm_run_multistart.sh")


def read_manifest():
    with open(_out_dir() / "manifest.csv") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# run one task
# ---------------------------------------------------------------------------

def cmd_run(a):
    from hpvpinn_capillarity.final_runs.common import (build_parser, read_constant, resolve_config,
                                                       run_seeds, run_tag, _write_json)
    tid = a.task_id if a.task_id is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    row = next(r for r in read_manifest() if int(r["id"]) == tid)
    init, seed = json.loads(row["init"]), int(row["seed"])

    mod = importlib.import_module(_CASE_PKG + row["module"])
    spec = mod.SPEC
    argv = ["--schedule", row["schedule"], "--seeds", str(seed)]
    if row["noise"] != "":
        argv += ["--noise-level", str(float(row["noise"]))]
    if a.epochs:
        argv += ["--epochs", str(a.epochs)]
    if a.winners:
        argv += ["--winners", a.winners]
    args = build_parser(spec).parse_args(argv)
    args.noise_level_f = float(args.noise_level) if spec.noise_levels else None
    cfg = resolve_config(spec, args)
    max_steps = read_constant(paths.module_source(spec.search_module), "MAX_STEPS")
    if args.epochs is None:
        args.epochs = int(max_steps)

    # override the initial values of the search module (read at call time by _vpinn_build_pinn)
    m = importlib.import_module(spec.search_module)
    for p, v in init.items():
        c = init_const_name(p)
        if not hasattr(m, c):
            raise SystemExit(f"{spec.search_module} has no constant {c}; see the module docstring "
                             f"of multistart.py (Case 1 needs KSALPHA_INIT / M_INIT).")
        setattr(m, c, float(v))

    tag = run_tag(spec.case, cfg["schedule"], args.noise_level_f, cfg["suffix"], args.epochs, max_steps)
    run_dir = paths.model_dir() / "final_runs" / f"{tag}_multistart" / f"run_{tid:04d}"
    print(f"[multistart] id={tid} block={row['block']} kind={row['kind']} seed={seed}\n"
          f"             init={init}\n             run_dir={run_dir}", flush=True)
    run_seeds(spec, args, cfg, run_dir)

    rp = run_dir / "seeds" / f"seed_{seed:05d}" / "record.json"
    rec = json.loads(rp.read_text())
    rec.update(block=row["block"], kind=row["kind"], run_id=tid, base_tag=tag, stage="multistart",
               init_requested=init)
    bad = []
    hp = run_dir / "seeds" / f"seed_{seed:05d}" / "history.npz"
    if hp.exists():
        h = np.load(hp)
        eff = {p: float(h[f"param_{p}"][0]) for p in init if f"param_{p}" in h}
        rec["init_effective"] = eff
        bad = [p for p, v in eff.items() if abs(v - init[p]) > INIT_RTOL * max(abs(init[p]), 1e-12)]
        rec["init_ok"] = not bad
    _write_json(rp, rec)
    if bad:
        print(f"[ERROR] start not applied for {bad}: requested {init}, epoch-0 values {rec['init_effective']}")
        sys.exit(3)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _ranks(x):
    return np.argsort(np.argsort(x)).astype(float)


def _corr(X):
    """Pearson and Spearman matrices of the columns of X (n x d)."""
    if X.shape[0] < 3:
        return None, None
    return np.corrcoef(X.T), np.corrcoef(np.apply_along_axis(_ranks, 0, X).T)


def _stat(v):
    v = np.asarray(v, float)
    return dict(n=int(v.size), mean=float(v.mean()), std=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                min=float(v.min()), max=float(v.max()))


def _load(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        out.append(json.loads(Path(p).read_text()))
    return out


def cmd_aggregate(a):
    from hpvpinn_capillarity.final_runs.common import read_constant
    from hpvpinn_capillarity.final_runs.profile_figures import ADMISSIBLE_PHYSICS, CALIBRATED_OBJ, is_qualified
    postprocess_src = paths.module_source("hpvpinn_capillarity.hpo.postprocess")
    adm = read_constant(postprocess_src, "ADMISSIBLE_PHYSICS", ADMISSIBLE_PHYSICS)
    cal = read_constant(postprocess_src, "CALIBRATED_OBJ", CALIBRATED_OBJ)
    out = _out_dir()
    man = read_manifest()
    summary = {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for block in dict.fromkeys(r["block"] for r in man):
        recs = [r for r in _load(str(paths.model_dir() / "final_runs/*_multistart/run_*/seeds/seed_*/record.json"))
                if r.get("block") == block and "_ep" not in r.get("base_tag", "")]
        if not recs:
            print(f"[{block}] no completed runs")
            continue
        params = list(RANGES[next(r["module"] for r in man if r["block"] == block)])
        rows = []
        for r in recs:
            ok = False
            if not r.get("failed") and "metrics" in r:
                ok = bool(is_qualified(r, adm, cal)[0])
            rows.append(dict(id=r["run_id"], kind=r["kind"], failed=bool(r.get("failed")), qualified=ok,
                             init_ok=r.get("init_ok"),
                             **{f"init_{p}": r["init_requested"][p] for p in params},
                             **({f"final_{p}": r["params"][p] for p in params} if "params" in r else {}),
                             **({k: r["metrics"].get(k) for k in
                                 ("chi2_nu_train", "chi2_nu_test", "pde_loss_eval", "final_physics",
                                  "strong_form_rms")} if "metrics" in r else {}),
                             obj=r.get("obj")))
        rows.sort(key=lambda x: x["id"])
        keys = list(dict.fromkeys(k for x in rows for k in x))
        with open(out / f"summary_{block}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

        n_all = len(rows)
        rnd = [x for x in rows if x["kind"] == "random"]
        q = [x for x in rnd if x["qualified"]]
        print(f"\n=== {block}: {n_all} runs ({len(rnd)} random), {len(q)} random runs qualified "
              f"(L_pde^eval<{adm:g}, obj<{cal:g}); {sum(1 for x in rows if x['init_ok'] is False)} with init mismatch")
        print("  deterministic starts:")
        for x in rows:
            if x["kind"] != "random":
                fin = "  ".join(f"{p}={x.get(f'final_{p}', float('nan')):.4g}" for p in params)
                print(f"    {x['kind']:9s} qual={x['qualified']!s:5s} obj={x.get('obj')}  {fin}")
        s = dict(n_runs=n_all, n_random=len(rnd), n_qualified_random=len(q))
        if len(q) >= 2:
            X = np.array([[x[f"final_{p}"] for p in params] for x in q])
            s["params"] = {p: _stat(X[:, j]) for j, p in enumerate(params)}
            pe, sp = _corr(X)
            s["pearson"], s["spearman"] = pe.tolist(), sp.tolist()
            s["obj"] = _stat([x["obj"] for x in q])
            s["obj_all_completed_sorted"] = sorted(float(x["obj"]) for x in rows if x.get("obj") is not None)
            print("  qualified random starts (mean +- std [min, max]):")
            for j, p in enumerate(params):
                st = s["params"][p]
                print(f"    {p:8s} {st['mean']:.5g} +- {st['std']:.2g} [{st['min']:.5g}, {st['max']:.5g}]")
            print(f"  Pearson r({params[0]},{params[1]}) = {pe[0, 1]:+.3f}   Spearman = {sp[0, 1]:+.3f}")
            print("  obj of all completed runs (sorted): "
                  + " ".join(f"{v:.3g}" for v in s["obj_all_completed_sorted"]))
        # seed ensemble of the same configuration (final run) for comparison
        base = recs[0]["base_tag"]
        ens = [r for r in _load(str(root / f"hp_vpinn/final_runs/{base}/seeds/seed_*/record.json"))
               if not r.get("failed") and "metrics" in r and is_qualified(r, adm, cal)[0]]
        if len(ens) >= 2 and "params" in s:
            E = np.array([[r["params"][p] for p in params] for r in ens])
            s["seed_ensemble"] = {p: _stat(E[:, j]) for j, p in enumerate(params)}
            s["std_ratio_multistart_over_seed"] = {
                p: (s["params"][p]["std"] / s["seed_ensemble"][p]["std"]
                    if s["seed_ensemble"][p]["std"] > 0 else None) for p in params}
            print("  std(multi-start)/std(seed ensemble): "
                  + "  ".join(f"{p}={v:.3g}" if v else f"{p}=n/a" for p, v in s["std_ratio_multistart_over_seed"].items()))
        summary[block] = s

        # LaTeX table
        if "params" in s:
            L = [r"\begin{tabular}{lccccc}", r"\toprule",
                 rf"\multicolumn{{6}}{{l}}{{{block.replace('_', ' ')}: $N={len(q)}$ qualified of {len(rnd)} random starts}} \\",
                 r"\midrule", r"Parameter & Mean & Std & Min & Max & Std ratio\textsuperscript{a} \\", r"\midrule"]
            for p in params:
                st = s["params"][p]
                ratio = s.get("std_ratio_multistart_over_seed", {}).get(p)
                pname = p.replace("_", "\\_")
                L.append(rf"  {pname} & \num{{{st['mean']:.4g}}} & \num{{{st['std']:.3g}}} & "
                         rf"\num{{{st['min']:.4g}}} & \num{{{st['max']:.4g}}} & "
                         + ("---" if not ratio else rf"\num{{{ratio:.3g}}}") + r" \\")
            L += [r"\bottomrule", r"\end{tabular}",
                  r"% a: std of the multi-start ensemble / std of the seed ensemble of the same configuration"]
            (out / f"latex_multistart_{block}.tex").write_text("\n".join(L) + "\n")

        # figure: final vs initial per parameter + (p0, p1) plane
        fig, axes = plt.subplots(1, len(params) + 1, figsize=(3.6 * (len(params) + 1), 3.6), squeeze=False)
        axes = axes[0]
        mod = next(r["module"] for r in man if r["block"] == block)
        ref = importlib.import_module(mod).SPEC.reference_params or {}
        for j, p in enumerate(params):
            ax = axes[j]
            for x in rows:
                if f"final_{p}" not in x:
                    continue
                ax.scatter(x[f"init_{p}"], x[f"final_{p}"],
                           marker="o" if x["kind"] == "random" else "s",
                           c="tab:green" if x["qualified"] else "tab:red", s=22)
            if p in ref:
                ax.axhline(ref[p], color="tab:blue", ls="--", lw=1)
            for axis, getter in (("x", f"init_{p}"), ("y", f"final_{p}")):
                v = np.array([x[getter] for x in rows if getter in x], float)
                if v.size and v.min() > 0 and v.max() / v.min() > 20:
                    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
            ax.set_xlabel(f"initial {p}"); ax.set_ylabel(f"final {p}"); ax.grid(alpha=0.3)
        ax = axes[-1]
        if len(ens) >= 2:
            ax.scatter(E[:, 0], E[:, 1], c="0.7", s=14, label="seed ensemble")
        if len(q) >= 2:
            ax.scatter(X[:, 0], X[:, 1], c="tab:green", s=22, label="multi-start")
        if params[0] in ref:
            ax.plot(ref[params[0]], ref[params[1]], "k*", ms=11, label="reference")
        ax.set_xlabel(params[0]); ax.set_ylabel(params[1]); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"multistart_{block}.{a.ext}", dpi=300)
        plt.close(fig)
        print(f"  [saved] {out / ('multistart_' + block + '.' + a.ext)}")

    (out / "multistart_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n[saved] {out / 'multistart_summary.json'}")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest")
    r = sub.add_parser("run")
    r.add_argument("--task-id", type=int, default=None, help="Default: $SLURM_ARRAY_TASK_ID")
    r.add_argument("--epochs", type=int, default=None, help="Smoke test; default: the HPO budget")
    r.add_argument("--winners", default=None)
    g = sub.add_parser("aggregate")
    g.add_argument("--ext", default="eps")
    a = p.parse_args()
    {"manifest": cmd_manifest, "run": cmd_run, "aggregate": cmd_aggregate}[a.cmd](a)


if __name__ == "__main__":
    main()

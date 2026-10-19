"""
benchmark_conventional.py
=========================
Driver for the conventional inverse-modelling benchmark (R1-22, R2-M09, R3-M2).
Place it at the repo root, next to hparam_search_<case>.py and
conventional_inverse.py; the data and the train/test split are taken from
each case's prepare_shared_data(), so both methods see identical arrays.

Modes
-----
--dry-run     Print data summary, bounds, x0, reference values, planned runs and
              a cost-evaluation sanity check (chi2_nu at x0 and at the reference
              parameters; implied scale vs reference -> unit check); estimated
              run time. No optimisation, no files written. Combine with
              --pilot or --seeds to preview those runs.
--pilot       Phase 0, seed 0 only. Three runs per case:
                (a) frontx_fit preset (maxiter=15, polish) on the train split,
                (b) upstream preset (to convergence)   on the train split,
                (c) upstream preset                    on the FULL data set
                    (reproduction check against the reference parameters).
              Prints a summary with per-evaluation time, to size the seed budget.
--profile-scan  grenoblesand only: chi2_nu(m) on a grid with Ks/alpha profiled,
              on train / full / noise-free data; prints 68%/95% profile intervals.
(default)     Benchmark: --seeds ... with --preset (default: converged) on
              --data-mode. Per-generation progress is printed; --max-minutes caps
              a run (flagged stopped_by_time); --maxiter/--tol/--atol/--polish
              override the preset.
--forward-check FILE.jsonl [--where stage=replication] | --params k=v ...
              chi2_nu of given (e.g. hp-VPINN) parameter sets with the exact
              forward solver on the same train/test split.

Usage
-----
    python -m hpvpinn_capillarity.conventional.benchmark --case letd --pilot --dry-run
    python -m hpvpinn_capillarity.conventional.benchmark --case grenoblesand --noise-level 0.02 --profile-scan
    python -m hpvpinn_capillarity.conventional.benchmark --case letd --seeds 0 --max-minutes 60
    python -m hpvpinn_capillarity.conventional.benchmark --case grenoblesand --noise-level 0.02 --pilot
    python -m hpvpinn_capillarity.conventional.benchmark --case letd --pilot
    python -m hpvpinn_capillarity.conventional.benchmark --case vangenuchten --seeds 0 1 2 3 4 5 6 7 8 9
    python -m hpvpinn_capillarity.conventional.benchmark --case brooks_corey --seeds 0-19 --preset upstream

Results are appended (one JSON record per line) to
    conventional/results_<case>[_<k>pct_noise].jsonl
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # CPU-only timing, same as hp-VPINN runs

import argparse
import importlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from hpvpinn_capillarity import paths
from hpvpinn_capillarity.conventional import inverse as ci  # (sets jax_enable_x64)

CASES = ("grenoblesand", "letd", "vangenuchten", "brooks_corey")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data(case: str, noise_level: float | None, data_path: str | None):
    """Return (train, test, full) as dicts with o, theta, sigma (physical units)."""
    mod = importlib.import_module(f"hpvpinn_capillarity.hpo.search_{case}")
    path = data_path or str(paths.data_file(f"data_{case}.dat"))

    if case == "grenoblesand":
        if noise_level is None:
            raise SystemExit("--noise-level is required for grenoblesand")
        sh = mod.prepare_shared_data(path, noise_level)
        s_tr = np.full_like(sh["theta_train"], sh["noise_level"])
        s_te = np.full_like(sh["theta_test"], sh["noise_level"])
    else:
        sh = mod.prepare_shared_data(path)
        s_tr, s_te = sh["std_train"], sh["std_test"]

    train = dict(o=np.asarray(sh["o_train"]), theta=np.asarray(sh["theta_train"]), sigma=np.asarray(s_tr))
    test = dict(o=np.asarray(sh["o_test"]), theta=np.asarray(sh["theta_test"]), sigma=np.asarray(s_te))

    order = np.argsort(np.concatenate([train["o"], test["o"]]))
    full = {k: np.concatenate([train[k], test[k]])[order] for k in ("o", "theta", "sigma")}
    return train, test, full


def results_path(case: str, noise_level: float | None) -> Path:
    out_dir = paths.conventional_dir()
    tag = f"_{int(round(noise_level * 100))}pct_noise" if case == "grenoblesand" else ""
    return out_dir / f"results_{case}{tag}.jsonl"


def parse_seeds(tokens: list[str]) -> list[int]:
    seeds: list[int] = []
    for tok in tokens:
        if "-" in tok:
            a, b = tok.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(tok))
    return seeds


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarise(rec: dict, spec: ci.CaseSpec, label: str) -> None:
    names = list(spec.reference)
    est = "  ".join(f"{k}={rec[k]:.4g}" for k in names)
    err = "  ".join(f"{k}:{rec.get('err_' + k, float('nan')):.1%}" for k in names)
    print(f"\n[{label}] seed={rec['seed']}  preset={rec['preset']}  data={rec['data_mode']}")
    print(f"  estimates : {est}")
    print(f"  rel. err. : {err}   (vs reference)")
    print(f"  chi2_nu   : fit={rec['chi2_nu_fit']:.4g}  test={rec['chi2_nu_test']:.4g}"
          f"  fit(check re-solve)={rec['chi2_nu_fit_check']:.4g}")
    print(f"  DE        : nit={rec['nit']}  nfev={rec['nfev']}  failed={rec['nfail']}"
          f"  success={rec['success']}  hit_maxiter={rec['hit_maxiter']}")
    if rec.get("stopped_by_time"):
        warnings.warn(f"[{label}] stopped by --max-minutes={rec['max_minutes']}: NOT converged.")
    print(f"  time [s]  : compile={rec['t_compile_s']:.2f}  per-eval={rec['t_eval_s'] * 1e3:.1f} ms"
          f"  DE={rec['t_de_s']:.2f}  total={rec['t_total_s']:.2f}")
    if rec["near_bound"]:
        warnings.warn(f"[{label}] estimate(s) within 1% of a bound: {rec['near_bound']} "
                      "-- widen the bound and rerun (see benchmark plan).")


def run_and_save(spec, fit, test, *, seed, preset, data_mode, stage, out, env, label,
                 run_kw=None):
    print(f"\n>>> {label}  seed={seed}", flush=True)
    rec = ci.run_de(spec, fit, test, seed=seed, preset=preset, **(run_kw or {}))
    rec.update(data_mode=data_mode, stage=stage, env=env)
    with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    summarise(rec, spec, label)
    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def dry_run(spec: ci.CaseSpec, train: dict, test: dict, full: dict, args, out: Path) -> None:
    """Print what would be solved; run a few cost evaluations; write nothing."""
    import jax.numpy as jnp

    def describe(name, d):
        sig = np.asarray(d["sigma"])
        print(f"  {name:<5}: N={len(d['o']):4d}  o=[{d['o'].min():.4g}, {d['o'].max():.4g}]"
              f"  theta=[{d['theta'].min():.4g}, {d['theta'].max():.4g}]"
              f"  sigma=[{sig.min():.3g}, {sig.max():.3g}]")

    print("\n--- DATA (same arrays as the hp-VPINN runs) ---")
    describe("train", train)
    describe("test", test)
    describe("full", full)
    print(f"  BCs  : theta_i={spec.theta_i}  theta_b={spec.theta_b}  n_params(dof)={spec.n_params}")

    print("\n--- MODEL / SEARCH SPACE ---")
    print(f"  scale '{spec.scale_name}' fixed at {spec.scale_ref:g} inside D, fitted via D0 "
          f"(estimate = {spec.scale_ref:g} * D0)")
    for p in spec.params:
        ref = spec.reference.get(p.name, float("nan"))
        print(f"  {p.name:<8} bounds=[{p.lo:g}, {p.hi:g}]  x0={p.x0:g}  reference={ref:.6g}")

    runs = ([("frontx_fit", "train", 0), ("upstream", "train", 0), ("upstream", "full", 0)]
            if args.pilot else
            [(args.preset, args.data_mode, s) for s in parse_seeds(args.seeds)])
    print("\n--- PLANNED RUNS ---")
    for preset, mode, seed in runs:
        st = dict(ci.DE_PRESETS[preset], **{k: v for k, v in dict(
            maxiter=args.maxiter, tol=args.tol, atol=args.atol, polish=args.polish).items()
            if v is not None})
        pop = st["popsize"] * len(spec.params)
        print(f"  preset={preset:<10} data={mode:<5} seed={seed}  population={pop}  "
              f"maxiter={st['maxiter']}  tol={st['tol']}  atol={st['atol']}  polish={st['polish']}")
    print(f"  results would be appended to: {out}")

    print("\n--- SANITY CHECK: cost evaluations (train split) ---")
    solve_scaled, cost = ci.make_problem(spec, train["o"], train["theta"], train["sigma"])
    t0 = time.perf_counter()
    c_x0 = float(cost(jnp.asarray(spec.x0)))
    t_compile = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(5):
        float(cost(jnp.asarray(spec.x0)))
    t_eval = (time.perf_counter() - t0) / 5
    print(f"  chi2_nu at x0 (hp-VPINN start)  = {c_x0:.4g}"
          + ("   <-- forward solve FAILED at x0" if c_x0 >= ci.PENALTY else ""))

    x_ref = np.array([spec.reference[p.name] for p in spec.params])
    c_ref = float(cost(jnp.asarray(x_ref)))
    s_ref = solve_scaled(jnp.asarray(x_ref))
    scale_ref_est = spec.scale_ref * float(s_ref.D0)
    scale_true = spec.reference.get(spec.scale_name, float("nan"))
    print(f"  chi2_nu at reference parameters = {c_ref:.4g}   (expect O(1))")
    print(f"  implied {spec.scale_name} at reference = {scale_ref_est:.6g}  "
          f"vs reference {scale_true:.6g}  (ratio {scale_ref_est / scale_true:.3f}; expect ~1 -> units OK)")
    print(f"  compile ~{t_compile:.1f} s, per-evaluation ~{t_eval * 1e3:.0f} ms")

    for preset in dict.fromkeys(r[0] for r in runs):
        st = dict(ci.DE_PRESETS[preset], **({"maxiter": args.maxiter} if args.maxiter else {}))
        n = st["popsize"] * len(spec.params) * (st["maxiter"] + 1)
        print(f"  upper bound for preset={preset}: {n} evals ~{n * t_eval / 60:.0f} min "
              "(runs usually stop earlier)")
    print("\nDry run complete: no optimisation run, no files written.")


def run_profile_scan(spec, train, full, args, out) -> None:
    """Grenoble only: chi2_nu(m) with Ks/alpha profiled, on train, full and the
    noise-free reference curve. Writes conventional/profile_<case>_<noise>.json."""
    lo, hi, n = args.grid
    grid = np.linspace(lo, hi, int(n))
    path = args.data or str(paths.data_file(f"data_{args.case}.dat"))
    raw = np.loadtxt(path, delimiter=",")
    # same sigma as the noisy data: LM tolerances are absolute, so sigma=1 would stop it early
    clean = dict(o=raw[:, 0], theta=raw[:, 1], sigma=np.full_like(raw[:, 1], args.noise_level))

    result = dict(case=args.case, noise_level=args.noise_level, param=spec.params[0].name,
                  itol=args.itol if args.itol is not None else 1e-3,
                  reference=spec.reference, n_params=spec.n_params)
    for name, d in (("train", train), ("full", full), ("clean", clean)):
        t0 = time.perf_counter()
        rows = ci.profile_scan(spec, d, grid, itol=args.itol)
        dof = len(d["o"]) - spec.n_params
        chi = np.array([r["chi2_nu"] for r in rows])
        ok = chi < ci.PENALTY
        i_min = int(np.argmin(np.where(ok, chi, np.inf)))
        dchi2 = (chi - chi[i_min]) * dof
        result[name] = dict(rows=rows, dof=dof, i_min=i_min)
        print(f"\n[{name}] N={len(d['o'])}  ({time.perf_counter() - t0:.0f} s)")
        print(f"  {'m':>8} {'KsAlpha':>10} {'chi2_nu':>12} {'Dchi2':>10}")
        for r, dc in zip(rows, dchi2):
            flag = " <- min" if r is rows[i_min] else ""
            print(f"  {r['value']:8.4f} {r['scale']:10.4g} {r['chi2_nu']:12.6g} {dc:10.3f}{flag}")
        if name != "clean":
            for lvl, thr in (("68%", 1.0), ("95%", 3.84)):
                inside = grid[ok & (dchi2 <= thr)]
                edge = " (reaches grid edge: widen --grid)" if inside.size and (
                    inside.min() <= grid[0] or inside.max() >= grid[-1]) else ""
                print(f"  {lvl} profile interval for m: [{inside.min():.4f}, {inside.max():.4f}]{edge}")
    out.parent.mkdir(exist_ok=True)
    tag = f"_{int(round(args.noise_level * 100))}pct_noise"
    if args.itol is not None:
        tag += f"_itol{args.itol:g}"
    with open(out.parent / f"profile_{args.case}{tag}.json", "w") as f:
        json.dump(result, f)
    print(f"\nSaved {out.parent / f'profile_{args.case}{tag}.json'}")


def run_forward_check(spec, train, test, args, out) -> None:
    """Evaluate externally estimated parameter sets (e.g. hp-VPINN runs) with
    the exact frontx forward solver on the same train/test data."""
    names = list(ci.importlib.import_module(f"hpvpinn_capillarity.hpo.search_{args.case}").TRUE_PARAMS)
    recs = []
    for path in args.forward_check:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("failed") or any(k not in r for k in names):
                    continue
                if any(str(r.get(k)) != v for k, v in (w.split("=", 1) for w in args.where)):
                    continue
                recs.append(r)
    if args.params:
        recs.append(dict((k, float(v)) for k, v in (w.split("=", 1) for w in args.params)))
    if not recs:
        raise SystemExit("no parameter sets found (check --forward-check / --where / --params)")

    print(f"\nForward check of {len(recs)} parameter set(s), exact frontx.solve "
          f"(itol={args.itol if args.itol is not None else 1e-3})")
    print("  " + "  ".join(f"{k:>10}" for k in names)
          + f"  {'chi2_fit':>9} {'chi2_test':>9} {'fit|D0':>9} {'D0':>7}")
    rows = []
    for r in recs:
        prm = {k: float(r[k]) for k in names}
        res = ci.forward_check(spec, prm, train, test, itol=args.itol)
        rows.append(dict(params=prm, source_trial=r.get("trial"), source_stage=r.get("stage"), **res))
        if not res["solve_ok"]:
            print("  " + "  ".join(f"{prm[k]:10.4g}" for k in names) + "  forward solve FAILED")
            continue
        print("  " + "  ".join(f"{prm[k]:10.4g}" for k in names)
              + f"  {res['chi2_nu_fit_exact']:9.4g} {res.get('chi2_nu_test_exact', float('nan')):9.4g}"
              + f" {res['chi2_nu_fit_scale_refit']:9.4g} {res['D0_refit']:7.3f}")
    ok = [x for x in rows if x["solve_ok"]]
    if ok:
        c = np.array([x["chi2_nu_fit_exact"] for x in ok])
        ct = np.array([x.get("chi2_nu_test_exact", np.nan) for x in ok])
        print(f"\n  chi2_nu (train, exact): median {np.median(c):.4g}  [min {c.min():.4g}, max {c.max():.4g}]"
              f"   test: median {np.nanmedian(ct):.4g}")
        dof = len(train["o"]) - spec.n_params
        print(f"  dof = {dof}; compare with the conventional optimum chi2_nu on the same split "
              f"(Delta chi2 = (chi2_nu - chi2_nu_opt) * dof)")
    out.parent.mkdir(exist_ok=True)
    tag = f"_{int(round(args.noise_level * 100))}pct_noise" if args.case == "grenoblesand" else ""
    dest = out.parent / f"forward_check_{args.case}{tag}.json"
    with open(dest, "w") as f:
        json.dump(dict(case=args.case, itol=args.itol, n_params=spec.n_params, rows=rows), f)
    print(f"\nSaved {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Conventional inverse benchmark (DE + frontx.solve)")
    ap.add_argument("--case", required=True, choices=CASES)
    ap.add_argument("--noise-level", type=float, default=None, choices=(0.01, 0.02),
                    help="grenoblesand only")
    ap.add_argument("--data", default=None, help="override data file path")
    ap.add_argument("--pilot", action="store_true", help="Phase 0: 3 runs, seed 0")
    ap.add_argument("--seeds", nargs="+", default=["0"], help="e.g. 0 1 2 or 0-9")
    ap.add_argument("--preset", default="converged", choices=list(ci.DE_PRESETS))
    ap.add_argument("--maxiter", type=int, default=None, help="override preset maxiter")
    ap.add_argument("--tol", type=float, default=None, help="override preset tol")
    ap.add_argument("--atol", type=float, default=None, help="override preset atol")
    ap.add_argument("--polish", action=argparse.BooleanOptionalAction, default=None,
                    help="override preset polish")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop a run after this wall time (flagged stopped_by_time)")
    ap.add_argument("--itol", type=float, default=None,
                    help="frontx.solve shooting tolerance (default: frontx's 1e-3)")
    ap.add_argument("--quiet", action="store_true", help="no per-generation progress")
    ap.add_argument("--profile-scan", action="store_true",
                    help="grenoblesand only: chi2_nu(m) scan with Ks/alpha profiled")
    ap.add_argument("--forward-check", nargs="+", default=None, metavar="JSONL",
                    help="evaluate parameter sets from JSONL file(s) (e.g. hp-VPINN results) "
                         "with the exact forward solver")
    ap.add_argument("--where", nargs="+", default=[], metavar="KEY=VALUE",
                    help="filter records for --forward-check, e.g. stage=replication")
    ap.add_argument("--params", nargs="+", default=None, metavar="NAME=VALUE",
                    help="a single parameter set to check, e.g. KsAlpha=242.4 m=0.6948")
    ap.add_argument("--grid", nargs=3, type=float, default=(0.35, 0.80, 46),
                    metavar=("LO", "HI", "N"), help="m grid for --profile-scan")
    ap.add_argument("--data-mode", default="train", choices=("train", "full"))
    ap.add_argument("--dry-run", action="store_true",
                    help="print data, bounds, planned runs and a few cost evaluations; "
                         "no optimisation, no files written")
    args = ap.parse_args()

    env = ci.environment_info()
    if env["jax_backend"] != "cpu":
        warnings.warn(f"JAX backend is {env['jax_backend']!r}, not 'cpu'.")
    print(f"frontx {env['frontx']} ({env['frontx_git']}) | jax {env['jax']} | "
          f"{env['cpu']} x{env['n_cpus']}")

    spec = ci.build_case(args.case)
    train, test, full = load_data(args.case, args.noise_level, args.data)
    out = results_path(args.case, args.noise_level)
    print(f"case={args.case}  n_train={len(train['o'])}  n_test={len(test['o'])}  "
          f"DE dims={len(spec.params)} {[p.name for p in spec.params]}  -> {out}")

    if args.dry_run:
        dry_run(spec, train, test, full, args, out)
        return

    if args.forward_check or args.params:
        run_forward_check(spec, train, test, args, out)
        return

    if args.profile_scan:
        if args.case != "grenoblesand":
            raise SystemExit("--profile-scan is implemented for grenoblesand (1 parameter) only")
        run_profile_scan(spec, train, full, args, out)
        return

    overrides = {k: v for k, v in dict(maxiter=args.maxiter, tol=args.tol, atol=args.atol,
                                       polish=args.polish).items() if v is not None}
    run_kw = dict(progress=not args.quiet, max_minutes=args.max_minutes, itol=args.itol,
                  **overrides)

    out.parent.mkdir(exist_ok=True)

    if args.pilot:
        recs = [
            run_and_save(spec, train, test, seed=0, preset="frontx_fit", data_mode="train",
                         stage="pilot", out=out, env=env, run_kw=run_kw, label="pilot a: frontx_fit, train"),
            run_and_save(spec, train, test, seed=0, preset="upstream", data_mode="train",
                         stage="pilot", out=out, env=env, run_kw=run_kw, label="pilot b: upstream, train"),
            run_and_save(spec, full, None, seed=0, preset="upstream", data_mode="full",
                         stage="full_data_check", out=out, env=env, run_kw=run_kw,
                         label="pilot c: upstream, FULL data (reproduction check)"),
        ]
        t_b = recs[1]["t_de_s"]
        print("\n" + "=" * 70)
        print(f"Pilot summary ({args.case}): per-eval ~{recs[1]['t_eval_s'] * 1e3:.1f} ms, "
              f"converged DE ~{t_b:.1f} s ({recs[1]['nfev']} evals)")
        print(f"  -> 10 seeds ~{10 * t_b / 60:.1f} min, 20 seeds ~{20 * t_b / 60:.1f} min")
        print("=" * 70)
        return

    fit, tst = (train, test) if args.data_mode == "train" else (full, None)
    for seed in parse_seeds(args.seeds):
        run_and_save(spec, fit, tst, seed=seed, preset=args.preset, data_mode=args.data_mode,
                     stage="benchmark", out=out, env=env, run_kw=run_kw,
                     label=f"{args.case} {args.preset} {args.data_mode}")


if __name__ == "__main__":
    main()

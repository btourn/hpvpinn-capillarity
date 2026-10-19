"""
sensitivity_noise_density.py
============================
Post-hoc sensitivity of the frozen Case 1 winner (Grenoble sand, hp-VPINN,
default: gb schedule, 1% noise) to

  (A) the noise realisation          -- R2-m08, R2-M03 (second half)
  (B) the training-data density       -- R1-19, R2-m08

Everything except the quantity under study is held fixed:

  * architecture / N_el / p        -> the Stage-A top-1 combination, resolved
                                      from the existing HPO results with the
                                      SAME observable-only criterion as
                                      hparam_confirm_top1.py, i.e. rank by
                                      (obj, final_physics); no ground truth
  * network-initialisation seed    -> the top-1 trial's own init_seed
                                      (override with --init-seed)
  * physical-parameter init        -> bound midpoint, as in the search
  * train/test split               -> SEED_SPLIT of hparam_search_grenoblesand
                                      (the test set is identical in every run)
  * training budget / schedule     -> MAX_STEPS, warm-up/transition of the
                                      selected schedule

Run plan (one entry per training run)
-------------------------------------
  realisation r = 0..R-1   ->  noise seed = SEED_NOISE + r
                               (r = 0 reproduces the Stage-A noise draw,
                                so (r=0, f=1.0) is a bitwise regression check
                                against the top-1 trial on the same machine)
  fraction f in FRACTIONS  ->  training subset of size round(f * N_pool)
                               taken from the fixed training pool

For a given r the noise vector is drawn ONCE over the full curve, so every
fraction of that realisation sees the same noisy values at the points it
keeps (paired design; subsets are nested for --subsample random, and for
stride whenever 1/f are integers dividing one another, e.g. 1, 1/2, 1/4,
1/8). The f = 1.0 column is study (A); all fractions together are (B).

Subsampling modes
-----------------
  stride  (default) every (1/f)-th point of the sorted training pool:
          models a coarser sensor/pixel resolution; variability across r is
          then due to noise only.
  random  nested random subsets, permutation seeded by SUBSAMPLE_SEED + r:
          models missing/irregular observations; variability across r mixes
          noise and sampling locations.

Output
------
One JSON per run under
  ./hp_vpinn/sensitivity_grenoblesand_<sched>_<noise>pct_noise[_random]/runs/
Runs whose JSON already exists are skipped (resumable). Aggregation is done by
postprocess_sensitivity.py.

Usage
-----
  python sensitivity_noise_density.py --list
  python sensitivity_noise_density.py --show-winner      # check the frozen config, no training
  python sensitivity_noise_density.py --task-id 0 --n-tasks 20
  python sensitivity_noise_density.py --run-index 0          # single run
  python sensitivity_noise_density.py --schedule gb --noise-level 0.01 \\
         --n-realisations 25 --fractions 1.0 0.5 0.25 0.125 --subsample stride
"""

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
MODEL = "hp_vpinn"

DEFAULT_FRACTIONS = (1.0, 0.5, 0.25, 0.125)
DEFAULT_N_REALISATIONS = 25     # same N as the Stage-B / multi-start ensembles
SUBSAMPLE_SEED = 10_000         # only used by --subsample random
MIN_TRAIN_POINTS = 8            # floor so chi2_nu keeps >= 6 degrees of freedom

SCHEDULES = {"gb": (0, 20_000), "es": (200_000, 0)}  # (warmup, transition)


# ---------------------------------------------------------------------------
# Lazy import of the search module (pulls in jax/optuna); kept out of module
# scope so --list works without the heavy stack.
# ---------------------------------------------------------------------------

def _search_module():
    import hparam_search_grenoblesand as m
    return m


# ---------------------------------------------------------------------------
# Top-1 ranking -- same criterion as hparam_confirm_top1.py (_load_ok_trials,
# _top1_and_count), reproduced here so this script does not import that file.
# Importing it would pull in all four hparam_search_<case>.py modules, and,
# once the hp_vpinn backend loader has put ./hp_vpinn at the front of
# sys.path, could resolve to a different copy of the file.
# ---------------------------------------------------------------------------

HP_COLS = ["nn_n_layers", "nn_width", "activation", "n_elements", "poly_degree"]
_CASTERS = {"nn_n_layers": int, "nn_width": int, "activation": str,
            "n_elements": int, "poly_degree": int}


def _load_ok_trials(results_path: str):
    """Successful, non-pruned trials of a results JSONL."""
    import pandas as pd
    with open(results_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(records)
    if "pruned" not in df.columns:
        df["pruned"] = False
    return df[(~df["failed"]) & (~df["pruned"])].reset_index(drop=True)


def _top1_and_count(df_ok, hp_cols):
    """Rank by (obj, final_physics) ascending; return (top1_row, count)."""
    if len(df_ok) == 0:
        return None, 0
    ranked = df_ok.sort_values(["obj", "final_physics"],
                               ascending=[True, True]).reset_index(drop=True)
    top1 = ranked.iloc[0]
    count = int((ranked[hp_cols] == top1[hp_cols]).all(axis=1).sum())
    return top1, count


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _tags(schedule: str, noise_level: float):
    noise_tag = f"{int(noise_level * 100):01d}pct_noise"
    return schedule, noise_tag


def hpo_results_path(schedule: str, noise_level: float) -> Path:
    """Stage-A results file -- mirrors hparam_search_grenoblesand.main()."""
    m = _search_module()
    sched_tag, noise_tag = _tags(schedule, noise_level)
    return (REPO_ROOT / MODEL / f"{m.BASE_RESULTS_DIR}_{sched_tag}_{noise_tag}"
            / f"hparam_results_{noise_tag}.jsonl")


def output_dir(schedule: str, noise_level: float, subsample: str) -> Path:
    sched_tag, noise_tag = _tags(schedule, noise_level)
    suffix = "" if subsample == "stride" else f"_{subsample}"
    return REPO_ROOT / MODEL / f"sensitivity_grenoblesand_{sched_tag}_{noise_tag}{suffix}"


# ---------------------------------------------------------------------------
# Frozen configuration: Stage-A top-1, observable criterion only
# ---------------------------------------------------------------------------

def resolve_winner(schedule: str, noise_level: float) -> dict:
    """Top-1 combo + its init seed, ranked exactly as hparam_confirm_top1.py
    does: successful, non-pruned trials sorted by (obj, final_physics). No
    parameter-error column is read. Restricted to stage == 'search' so the
    frozen seed is the one that won the search, not a later replicate."""
    hp_cols = HP_COLS
    path = hpo_results_path(schedule, noise_level)
    if not path.exists():
        raise FileNotFoundError(f"Stage-A results not found: {path}")

    df_ok = _load_ok_trials(str(path))
    if "stage" in df_ok.columns:
        df_ok = df_ok[df_ok["stage"].fillna("search") == "search"].reset_index(drop=True)
    top1, count = _top1_and_count(df_ok, hp_cols)
    if top1 is None:
        raise RuntimeError(f"No successful, non-pruned trials in {path}")

    combo = {k: _CASTERS[k](top1[k]) for k in hp_cols}
    init_seed = int(top1["init_seed"]) if "init_seed" in top1 else int(top1["trial"])
    return dict(
        combo=combo,
        init_seed=init_seed,
        trial=int(top1["trial"]),
        occurrences_in_search=int(count),
        obj=float(top1["obj"]),
        final_physics=float(top1["final_physics"]),
        # recorded only for the r=0 regression check in postprocessing
        KsAlpha_top1=float(top1.get("KsAlpha", np.nan)),
        m_top1=float(top1.get("m", np.nan)),
        results_path=str(path),
    )


# ---------------------------------------------------------------------------
# Run plan
# ---------------------------------------------------------------------------

def build_plan(n_realisations: int, fractions) -> list[dict]:
    """Fraction-major, realisation-minor. Run index is stable for a given
    (n_realisations, fractions), so a SLURM array can be resubmitted safely."""
    fractions = sorted({float(f) for f in fractions}, reverse=True)
    plan = []
    for f in fractions:
        for r in range(n_realisations):
            plan.append(dict(run_index=len(plan), realisation=r, fraction=f))
    return plan


# ---------------------------------------------------------------------------
# Data preparation -- mirrors prepare_shared_data() in
# hparam_search_grenoblesand.py line by line, with the noise seed and the
# training subset as the only free inputs.
# ---------------------------------------------------------------------------

def prepare_run_data(data_path: str, noise_level: float, realisation: int,
                     fraction: float, subsample: str, m=None) -> dict:
    m = m or _search_module()

    raw = np.loadtxt(data_path, delimiter=",")
    o_all, theta_clean = raw[:, 0], raw[:, 1]
    oi = 1.05 * float(o_all[-1])          # from the FULL curve, as in the search

    noise_seed = int(m.SEED_NOISE) + int(realisation)
    rng_noise = np.random.default_rng(noise_seed)
    if noise_level > 0.0:
        theta_noisy = theta_clean + rng_noise.normal(0.0, noise_level, size=theta_clean.shape)
    else:
        theta_noisy = theta_clean.copy()

    rng_split = np.random.default_rng(m.SEED_SPLIT)
    idx = rng_split.permutation(len(o_all))
    n_test = max(1, int(m.TEST_FRACTION * len(o_all)))
    test_idx = np.sort(idx[:n_test])
    pool_idx = np.sort(idx[n_test:])      # == train_idx of the search

    n_pool = len(pool_idx)
    n_keep = int(round(fraction * n_pool))
    n_keep = min(n_pool, max(MIN_TRAIN_POINTS, n_keep))
    step = 1.0 / fraction

    if n_keep == n_pool:
        train_idx = pool_idx
    elif subsample == "stride" and abs(step - round(step)) < 1e-9 \
            and len(pool_idx[::int(round(step))]) >= MIN_TRAIN_POINTS:
        # every k-th point of the sorted pool: nested whenever the steps
        # divide one another (f = 1/2, 1/4, 1/8, ...)
        train_idx = pool_idx[::int(round(step))]
    elif subsample == "stride":
        # non-integer step (or floor hit): evenly spaced, not guaranteed nested
        pos = np.unique(np.round(np.linspace(0, n_pool - 1, n_keep)).astype(int))
        train_idx = pool_idx[pos]
    elif subsample == "random":
        perm = np.random.default_rng(SUBSAMPLE_SEED + int(realisation)).permutation(n_pool)
        train_idx = np.sort(pool_idx[perm[:n_keep]])
    else:
        raise ValueError(f"unknown subsample mode {subsample!r}")

    return dict(
        shared=dict(
            oi=oi,
            o_train=o_all[train_idx], theta_train=theta_noisy[train_idx],
            o_test=o_all[test_idx],   theta_test=theta_noisy[test_idx],
            noise_level=float(noise_level),
        ),
        noise_seed=noise_seed,
        n_pool=int(n_pool),
        n_train=int(len(train_idx)),
        n_test=int(len(test_idx)),
        train_idx=train_idx.tolist(),
        test_idx=test_idx.tolist(),
        o_train_max=float(o_all[train_idx].max()),
    )


# ---------------------------------------------------------------------------
# One training run
# ---------------------------------------------------------------------------

def _objective_value(chi2_train: float, chi2_test: float) -> float:
    """Same compound objective as make_objective() in the search script."""
    gap = max(0.0, chi2_test - chi2_train)
    relative_gap = min(gap / (chi2_train + 1e-8), 10.0)
    return max(0.0, chi2_test - 1.0) + (chi2_test * relative_gap if chi2_test > 1.0 else 0.0)


def execute_run(entry: dict, winner: dict, args, out_dir: Path) -> None:
    m = _search_module()
    out_path = out_dir / "runs" / f"run_{entry['run_index']:04d}.json"
    if out_path.exists():
        print(f"[run {entry['run_index']}] exists, skipping")
        return

    prep = prepare_run_data(args.data, args.noise_level, entry["realisation"],
                            entry["fraction"], args.subsample, m)
    data = m.BACKENDS[MODEL]["prepare"](prep["shared"])
    warmup, transition = SCHEDULES[args.schedule]

    print(f"\n[run {entry['run_index']}] r={entry['realisation']} "
          f"(noise seed {prep['noise_seed']})  f={entry['fraction']} "
          f"-> N_train={prep['n_train']}/{prep['n_pool']}  "
          f"init_seed={winner['init_seed']}  combo={winner['combo']}")

    # run_vpinn_trial only reads trial.number (-> network init seed)
    fake_trial = SimpleNamespace(number=winner["init_seed"])
    record = dict(
        run_index=entry["run_index"], realisation=entry["realisation"],
        fraction=entry["fraction"], subsample=args.subsample,
        schedule=args.schedule, noise_level=args.noise_level,
        noise_seed=prep["noise_seed"], split_seed=int(m.SEED_SPLIT),
        init_seed=winner["init_seed"], winner_trial=winner["trial"],
        n_pool=prep["n_pool"], n_train=prep["n_train"], n_test=prep["n_test"],
        o_train_max=prep["o_train_max"], **winner["combo"],
    )

    t0 = time.perf_counter()
    try:
        _, metrics, diverged, recovered = m.run_vpinn_trial(
            fake_trial, dict(winner["combo"]), data, args.noise_level, warmup, transition,
        )
    except Exception as exc:  # keep the run in the ensemble as a failure
        record.update(failed=True, error=repr(exc), elapsed_s=time.perf_counter() - t0)
        _write(out_path, record)
        print(f"  FAILED: {exc!r}")
        return

    if diverged:
        record.update(failed=True, diverged=True, **metrics)
        _write(out_path, record)
        print("  diverged (NaN)")
        return

    param_errors = {
        f"err_{k}": abs(recovered[k] - m.TRUE_PARAMS[k]) / abs(m.TRUE_PARAMS[k])
        for k in m.TRUE_PARAMS
    }
    obj = _objective_value(metrics["chi2_nu_train"], metrics["chi2_nu_test"])
    record.update(
        failed=False, **metrics, **recovered, **param_errors, obj=obj,
        admissible=bool(metrics["final_physics"] < m.PHYSICS_THRESHOLD),
        calibrated=bool(obj < 2.0),
    )
    _write(out_path, record)
    print(f"  KsAlpha={recovered['KsAlpha']:.4g} ({param_errors['err_KsAlpha']:.2%})  "
          f"m={recovered['m']:.4f} ({param_errors['err_m']:.2%})  "
          f"chi2 tr/te={metrics['chi2_nu_train']:.3f}/{metrics['chi2_nu_test']:.3f}  "
          f"phys={metrics['final_physics']:.2e}  t={metrics['elapsed_s']:.0f}s")


def _write(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(record, f, indent=1)
    os.replace(tmp, path)   # atomic: a killed job never leaves a half file


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--schedule", choices=list(SCHEDULES), default="gb")
    p.add_argument("--noise-level", type=float, default=0.01)
    p.add_argument("--n-realisations", type=int, default=DEFAULT_N_REALISATIONS)
    p.add_argument("--fractions", type=float, nargs="+", default=list(DEFAULT_FRACTIONS))
    p.add_argument("--subsample", choices=["stride", "random"], default="stride")
    p.add_argument("--data", default=str(REPO_ROOT / "data_grenoblesand.dat"))
    p.add_argument("--init-seed", type=int, default=None,
                   help="Override the frozen network-init seed (default: top-1's).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true", help="Print the plan and exit.")
    g.add_argument("--show-winner", action="store_true",
                   help="Resolve and print the frozen configuration, then exit (no training).")
    g.add_argument("--run-index", type=int, default=None)
    g.add_argument("--task-id", type=int, default=None,
                   help="Strided share of the plan: indices i with i %% n_tasks == task_id.")
    p.add_argument("--n-tasks", type=int, default=1)
    args = p.parse_args()

    plan = build_plan(args.n_realisations, args.fractions)

    if args.list:
        print(f"{len(plan)} runs  (R={args.n_realisations}, fractions={sorted(set(args.fractions), reverse=True)})")
        for e in plan[: min(len(plan), 2 * args.n_realisations)]:
            print(f"  {e['run_index']:4d}  r={e['realisation']:2d}  f={e['fraction']}")
        if len(plan) > 2 * args.n_realisations:
            print("  ...")
        return

    m = _search_module()
    m.BACKENDS[MODEL]["loader"]()

    winner = resolve_winner(args.schedule, args.noise_level)
    if args.init_seed is not None:
        winner["init_seed"] = int(args.init_seed)

    if args.show_winner:
        print(json.dumps(winner, indent=1))
        return

    out_dir = output_dir(args.schedule, args.noise_level, args.subsample)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.json"
    if not cfg_path.exists():
        _write(cfg_path, dict(winner=winner, n_realisations=args.n_realisations,
                              fractions=sorted(set(args.fractions), reverse=True),
                              subsample=args.subsample, schedule=args.schedule,
                              noise_level=args.noise_level, max_steps=m.MAX_STEPS,
                              data=args.data))

    print("=" * 60)
    print(f"Frozen config (Stage-A top-1, trial {winner['trial']}, "
          f"{winner['occurrences_in_search']} occurrences in search): {winner['combo']}")
    print(f"  init seed {winner['init_seed']}   schedule {args.schedule}   "
          f"sigma {args.noise_level}   subsample {args.subsample}")
    print(f"  output {out_dir}")
    print("=" * 60)

    if args.run_index is not None:
        todo = [plan[args.run_index]]
    elif args.task_id is not None:
        todo = [e for e in plan if e["run_index"] % args.n_tasks == args.task_id]
    else:
        todo = plan

    for entry in todo:
        execute_run(entry, winner, args, out_dir)


if __name__ == "__main__":
    main()

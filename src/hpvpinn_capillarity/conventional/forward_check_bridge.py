#!/usr/bin/env python3
"""
final_runs_to_forward_check.py
==============================
Bridge between the hp-VPINN final runs and the exact-solver forward check
(benchmark_conventional.py --forward-check).

  1. convert   (default)  hp_vpinn/final_runs/<tag>/seeds/seed_*/record.json
                          -> conventional/forward_check_inputs/hpvpinn_<tag>.jsonl
                          One JSON line per seed, with the recovered parameters under
                          the keys that --forward-check expects (the TRUE_PARAMS keys of
                          hparam_search_<case>.py), trial = seed, and, for reference, the
                          hp-VPINN's own chi2 values and a `qualified` flag.
  2. --summarise          joins conventional/forward_check_<case>[_<k>pct_noise].json
                          (written by --forward-check) with the hp-VPINN's own chi2 values
                          and prints one table per tag (also written to
                          conventional/forward_check_summary_<tag>.csv).

Place it at the repo root (next to benchmark_conventional.py) and run it from there.

    python3 final_runs_to_forward_check.py                 # convert the four tags below
    python3 final_runs_to_forward_check.py --summarise     # after the forward checks
"""
import argparse
import csv
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# final-run tag -> (case, noise level or None, file-name tag of the forward-check output)
TAGS = {
    "grenoblesand_gb_1pct_noise": ("grenoblesand", 0.01, "_1pct_noise"),
    "grenoblesand_gb_2pct_noise": ("grenoblesand", 0.02, "_2pct_noise"),
    "vangenuchten_es":            ("vangenuchten", None, ""),
    "letd_gb":                    ("letd",         None, ""),
}
CALIBRATED, ADMISSIBLE = 4.0, 1e-5


def load_records(run_dir: Path):
    recs = []
    for rp in sorted((run_dir / "seeds").glob("seed_*/record.json")):
        r = json.loads(rp.read_text())
        if r.get("failed") or r.get("diverged") or "params" not in r:
            print(f"  skipping {rp.parent.name} (failed/diverged)")
            continue
        recs.append(r)
    return recs


def convert(args):
    out_dir = ROOT / "conventional" / "forward_check_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fr = Path(args.final_runs_dir)
    for tag, (case, noise, _) in TAGS.items():
        run_dir = fr / tag
        if not run_dir.is_dir():
            print(f"[{tag}] {run_dir} not found, skipped")
            continue
        recs = load_records(run_dir)
        lines = []
        for r in recs:
            m = r["metrics"]
            lines.append(dict(
                trial=int(r["seed"]), stage="final_run", failed=False, tag=tag,
                **{k: float(v) for k, v in r["params"].items()},
                hp_chi2_nu_train=float(m["chi2_nu_train"]), hp_chi2_nu_test=float(m["chi2_nu_test"]),
                hp_obj=float(r["obj"]), hp_pde_loss_eval=float(m["pde_loss_eval"]),
                qualified=bool(r["obj"] < CALIBRATED and m["pde_loss_eval"] < ADMISSIBLE),
            ))
        dest = out_dir / f"hpvpinn_{tag}.jsonl"
        dest.write_text("".join(json.dumps(x) + "\n" for x in lines))
        n_q = sum(x["qualified"] for x in lines)
        print(f"[{tag}] {len(lines)} seeds ({n_q} qualified) -> {dest}")
    print("\nNow run, from the repo root with the hp_vpinn environment active:\n")
    for tag, (case, noise, _) in TAGS.items():
        nl = f" --noise-level {noise}" if noise is not None else ""
        print(f"JAX_PLATFORMS=cpu python3 benchmark_conventional.py --case {case}{nl} "
              f"--forward-check conventional/forward_check_inputs/hpvpinn_{tag}.jsonl")


def summarise(args):
    for tag, (case, noise, ftag) in TAGS.items():
        fc_path = ROOT / "conventional" / f"forward_check_{case}{ftag}{args.suffix}.json"
        inp = ROOT / "conventional" / "forward_check_inputs" / f"hpvpinn_{tag}.jsonl"
        if not fc_path.exists() or not inp.exists():
            print(f"[{tag}] missing {fc_path.name if not fc_path.exists() else inp.name}, skipped")
            continue
        hp = {int(json.loads(l)["trial"]): json.loads(l) for l in inp.read_text().splitlines() if l.strip()}
        fc = json.loads(fc_path.read_text())
        rows = []
        for r in fc["rows"]:
            s = hp.get(r.get("source_trial"))
            if s is None:
                continue
            ok = r.get("solve_ok", False)
            rows.append(dict(
                seed=s["trial"], qualified=s["qualified"], solve_ok=ok,
                hp_train=s["hp_chi2_nu_train"], hp_test=s["hp_chi2_nu_test"],
                ex_train=r.get("chi2_nu_fit_exact", float("nan")),
                ex_test=r.get("chi2_nu_test_exact", float("nan")),
                ex_train_scale_refit=r.get("chi2_nu_fit_scale_refit", float("nan")),
                D0_refit=r.get("D0_refit", float("nan")),
            ))
        print(f"\n=== {tag}  (exact forward solve of the hp-VPINN parameters; itol={fc.get('itol') or '1e-3 (default)'}) ===")
        print(f"{'seed':>5} {'qual':>5} {'ok':>3} | {'hp train':>9} {'exact':>9} {'x':>7} | "
              f"{'hp test':>9} {'exact':>9} | {'exact,scale refit':>17} {'D0':>7}")
        for x in rows:
            ratio = x["ex_train"] / x["hp_train"] if x["solve_ok"] else float("nan")
            print(f"{x['seed']:5d} {str(x['qualified'])[0]:>5} {str(x['solve_ok'])[0]:>3} | "
                  f"{x['hp_train']:9.3f} {x['ex_train']:9.3f} {ratio:7.2f} | "
                  f"{x['hp_test']:9.3f} {x['ex_test']:9.3f} | {x['ex_train_scale_refit']:17.3f} {x['D0_refit']:7.3f}")
        good = [x for x in rows if x["solve_ok"]]
        if good:
            med = lambda k: st.median(x[k] for x in good)
            print(f"  medians over {len(good)} seeds: hp train {med('hp_train'):.3f} -> exact {med('ex_train'):.3f};  "
                  f"hp test {med('hp_test'):.3f} -> exact {med('ex_test'):.3f};  "
                  f"exact after scale refit {med('ex_train_scale_refit'):.3f}")
        dest = ROOT / "conventional" / f"forward_check_summary_{tag}{args.suffix}.csv"
        with open(dest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())) if rows else None
            if w:
                w.writeheader(); w.writerows(rows)
        print(f"  -> {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final-runs-dir", default=str(ROOT / "hp_vpinn" / "final_runs"))
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--suffix", default="",
                    help="suffix of the forward-check files to summarise, e.g. _itol1e-5 reads "
                         "conventional/forward_check_<case>_itol1e-5.json")
    args = ap.parse_args()
    (summarise if args.summarise else convert)(args)


if __name__ == "__main__":
    sys.exit(main())

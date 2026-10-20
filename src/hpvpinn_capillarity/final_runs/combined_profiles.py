"""Combined 2x2 wetting-profile figure from the saved
final runs of the four case studies. No training and no JAX: it reads
<results>/hp_vpinn/final_runs/<tag>/ (reference.npz, seeds/*/record.json,
seeds/*/curves.npz), written by grenoblesand.py, vangenuchten.py, letd.py.

    a) Case 1, 1% noise     b) Case 1, 2% noise
    c) Case 2 (with zoomed inset of the wetting front)   d) Case 3

Labels a)-d) sit at the top right of each panel. Each panel shows the
representative run of the qualified seeds (same gate as final_runs/common.py:
held-out PDE loss below the admissibility threshold and obj below the
calibration threshold), with the seed envelope where it is visible.

Usage
-----
    python -m hpvpinn_capillarity.final_runs.combined_profiles
    python -m hpvpinn_capillarity.final_runs.combined_profiles --band p5-95 --representative median-obj
    python -m hpvpinn_capillarity.final_runs.combined_profiles --tags grenoblesand_gb_1pct_noise \\
        grenoblesand_gb_2pct_noise vangenuchten_es letd_gb

Default run tags: the configurations selected in winners.json for
grenoblesand (1%, 2% noise), vangenuchten and letd. Output (default
<results>/hp_vpinn/Figures): final_profiles_combined.eps and .png.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from hpvpinn_capillarity import paths
from hpvpinn_capillarity.final_runs import common as frc
from hpvpinn_capillarity.final_runs.profile_figures import (ADMISSIBLE_PHYSICS, CALIBRATED_OBJ, ZOOM_BY_CASE,
                             band_from_pool, draw_profile, is_qualified, refit_zoom)

PANELS = [  # (winners.json group key, case name for the zoom lookup)
    ("grenoblesand_1pct_noise", "grenoblesand"),
    ("grenoblesand_2pct_noise", "grenoblesand"),
    ("vangenuchten", "vangenuchten"),
    ("letd", "letd"),
]


def load_panel(run_dir: Path, adm, cal, rule):
    """(ref, rep_curve, th_pool, schedule, noise_level, n_used) of one run dir."""
    ref = dict(np.load(run_dir / "reference.npz"))
    runs = [r for r in frc._load_seed_runs(run_dir) if not r.get("failed")]
    if not runs:
        raise SystemExit(f"No completed seeds in {run_dir}.")
    ok = [r for r in runs if is_qualified(r, adm, cal)[0]]
    if not ok:
        warnings.warn(f"{run_dir.name}: no qualified seed; using all {len(runs)} completed seeds.")
        ok = runs
    curves = [dict(np.load(r["_dir"] / "curves.npz")) for r in ok]
    D = np.stack([c["D"] for c in curves])
    th = np.stack([c["theta_nd"] for c in curves])
    i = frc.choose_representative(ok, D, ref["theta"], ref["theta_data_range"], rule)
    print(f"  {run_dir.name}: {len(ok)}/{len(runs)} qualified seeds; representative seed {ok[i]['seed']}")
    return ref, curves[i], th, ok[0].get("schedule"), ok[0].get("noise_level"), len(ok)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--winners", default=None, help="winners.json (default: <model dir>/tables/winners.json).")
    ap.add_argument("--tags", nargs=4, default=None, metavar="TAG",
                    help="Four run tags (panels a-d); default: selected in winners.json.")
    ap.add_argument("--band", choices=["minmax", "p5-95"], default="minmax")
    ap.add_argument("--representative", choices=["d-medoid", "median-obj"], default="d-medoid")
    ap.add_argument("--repo-root", default=None, help="Default: directory of this script (either <repo>/ or <repo>/hp_vpinn/).")
    ap.add_argument("--figures-dir", default=None)
    ap.add_argument("--out-name", default="final_profiles_combined")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    mdir = Path(args.repo_root).resolve() / frc.MODEL if args.repo_root else paths.model_dir(frc.MODEL)
    figs = Path(args.figures_dir) if args.figures_dir else mdir / "Figures"
    figs.mkdir(parents=True, exist_ok=True)
    tabs = mdir / "tables"

    if args.tags:
        tags = list(args.tags)
    else:
        wpath = Path(args.winners) if args.winners else tabs / "winners.json"
        sel = json.loads(wpath.read_text()).get("selected", {})
        missing = [k for k, _ in PANELS if k not in sel]
        if missing:
            raise SystemExit(f"No selected configuration for {missing} in {wpath}; pass --tags.")
        tags = [sel[k] for k, _ in PANELS]

    adm = frc.read_constant(paths.module_source("hpvpinn_capillarity.hpo.postprocess"), "ADMISSIBLE_PHYSICS", ADMISSIBLE_PHYSICS)
    cal = frc.read_constant(paths.module_source("hpvpinn_capillarity.hpo.postprocess"), "CALIBRATED_OBJ", CALIBRATED_OBJ)
    print(f"[gate] pde_loss_eval < {adm:g} and obj < {cal:g}")

    matplotlib, plt = frc._setup_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.6))
    for ax, tag, (_, case), letter in zip(axes.ravel(), tags, PANELS, "abcd"):
        ref, rep, th, sched, noise, _ = load_panel(mdir / "final_runs" / tag,
                                                   adm, cal, args.representative)
        draw_profile(ax, ref, rep, band_from_pool(th, args.band), sched, noise,
                     zoom=ZOOM_BY_CASE.get(case), legend_fontsize=7.5,
                     panel_label=f"{letter})")
    fig.tight_layout()
    for ax in axes.ravel():
        refit_zoom(ax)

    exts = ["eps"] + ([] if args.no_png else ["png"])
    for ext in exts:
        path = figs / f"{args.out_name}.{ext}"
        fig.savefig(path, format=ext, bbox_inches="tight", dpi=300)
        print(f"[saved] {path}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

"""
postprocess_sensitivity.py
==========================
Aggregate the runs written by sensitivity_noise_density.py and produce

  latex_noise_realisations_<tag>.tex   init-seed spread (fixed noise) vs
                                       noise-realisation spread (fixed init)
  latex_data_density_<tag>.tex         recovery vs N_train
  sensitivity_<tag>.{eps,pdf}          (a) K_s/alpha-m plane, both ensembles
                                       (b), (c) relative errors vs N_train
  sensitivity_<tag>_summary.csv        every number in the two tables
  sensitivity_<tag>_runs.csv           one row per run (for inspection)

All outputs go to the study directory (or --out), which is printed at the
start and at the end. `--list-studies` lists every sensitivity study found
under --root, with its modification time, number of runs and frozen trial,
so the directory of a given run can always be identified.

Pure numpy/pandas/matplotlib: no JAX needed.

Acceptance gate (observable only, never uses the true parameters)
-----------------------------------------------------------------
A run is ACCEPTED if it did not diverge, is admissible and is calibrated:

  admissible  physics metric < 1e-5. Default metric (--physics-metric eval):
              the weak-form PDE loss on the held-out quadrature
              (`pde_loss_eval`), as in the search tables of the SI. If a
              run file does not store it, BOTH ensembles fall back to the
              training-quadrature loss (`final_physics`) with a warning, so
              the two rows of the noise table always use the same gate.
  calibrated  obj = chi2_train + chi2_test < 4 (mean chi2_nu < 2), recomputed
              for every run and trial (the stored `obj` is not trusted).

Statistics are computed over accepted runs; rejected runs are counted in
"Acc./Total", drawn as crosses in the figure and kept in the per-run CSV.
The init-seed ensemble (all search trials of the frozen combination, fixed
noise draw) is filtered with the same gate.

Statistics: medians and interquartile ranges, median-based bias and relative
error, Pearson r between the recovered parameters, median chi2_nu over ALL
non-diverged runs, and the largest physics-metric value among accepted runs
(reported so that the gate can be stated quantitatively in the SI).

Usage
-----
  python -m hpvpinn_capillarity.analysis.postprocess_sensitivity --list-studies
  python -m hpvpinn_capillarity.analysis.postprocess_sensitivity                        # gb, 1%, stride
  python -m hpvpinn_capillarity.analysis.postprocess_sensitivity --noise-level 0.02     # the 2% case
  python -m hpvpinn_capillarity.analysis.postprocess_sensitivity --physics-metric train # previous rule
"""

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hpvpinn_capillarity import paths
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KSALPHA_TRUE = 15.37 / 0.0432
M_TRUE = 0.5096
PHYSICS_THRESHOLD = 1e-5
OBJ_THRESHOLD = 4.0   # sum objective: mean chi2_nu < 2 (a noise-level fit scores ~2)
PHYSICS_COLS = {"eval": "pde_loss_eval", "train": "final_physics"}
HP_COLS = ["nn_n_layers", "nn_width", "activation", "n_elements", "poly_degree"]

C_NOISE = "#1f4e9c"    # noise-realisation ensemble, accepted
C_INIT = "#8c8c8c"     # init-seed ensemble, accepted
C_REJ = "#c0392b"      # rejected by the gate


# ---------------------------------------------------------------------------
# Loading and gating
# ---------------------------------------------------------------------------

def _flag(df, col, default=False):
    if col not in df:
        df[col] = default
    df[col] = df[col].fillna(default).astype(bool)


def gate_objective(chi2_train, chi2_test):
    """obj = chi2_train + chi2_test, recomputed from the two chi2 columns so
    every run and trial is judged by the same formula, whatever version wrote
    the stored `obj` field."""
    return np.asarray(chi2_train, float) + np.asarray(chi2_test, float)


def resolve_physics_col(frames, physics):
    """Physics-metric column shared by all ensembles (falls back to the
    training-quadrature loss if any of them lacks the held-out one)."""
    col = PHYSICS_COLS[physics]
    if col != "final_physics":
        for name, df in frames.items():
            ok = col in df and pd.to_numeric(df[col], errors="coerce").notna().any()
            if not ok:
                print(f"[warn] '{col}' not stored in the {name} records; using the "
                      f"training-quadrature 'final_physics' for ALL ensembles.")
                return "final_physics"
    return col


def apply_gate(df: pd.DataFrame, gate: str, phys_col: str) -> pd.DataFrame:
    df = df.copy()
    _flag(df, "failed")
    fp = pd.to_numeric(df.get(phys_col), errors="coerce")
    obj = pd.Series(gate_objective(pd.to_numeric(df.get("chi2_nu_train"), errors="coerce"),
                                   pd.to_numeric(df.get("chi2_nu_test"), errors="coerce")),
                    index=df.index)
    df["physics_value"] = fp
    df["obj_gate"] = obj
    df["admissible"] = (~df["failed"]) & (fp < PHYSICS_THRESHOLD)
    df["calibrated"] = (~df["failed"]) & (obj < OBJ_THRESHOLD)
    df["accepted"] = df["admissible"] & (df["calibrated"] if gate == "calibrated" else True)
    return df


def load_runs_raw(run_dir: Path) -> pd.DataFrame:
    recs = [json.loads(p.read_text()) for p in sorted(run_dir.glob("run_*.json"))]
    if not recs:
        raise FileNotFoundError(f"no run_*.json in {run_dir}")
    return pd.DataFrame(recs)


def load_init_seed_raw(results_path: Path, combo: dict) -> pd.DataFrame | None:
    """Successful, non-pruned search trials of the frozen combination: fixed
    noise draw, varying network-init seed."""
    if not results_path.is_file():
        print(f"[warn] search results not found ({results_path}); init-seed row omitted.")
        return None
    recs = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(recs)
    _flag(df, "pruned")
    df = df[~df["pruned"]]
    for k, v in combo.items():
        df = df[df[k] == v]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Study discovery
# ---------------------------------------------------------------------------

def list_studies(root: Path) -> None:
    base = root / "hp_vpinn"
    print(f"Searching {base} for sensitivity studies\n")
    dirs = sorted(base.glob("sensitivity_*"), key=lambda p: p.stat().st_mtime) if base.is_dir() else []
    if not dirs:
        print("  none found (check --root: it defaults to the directory of this script)")
        return
    for d in dirs:
        runs = list((d / "runs").glob("run_*.json"))
        latest = max((p.stat().st_mtime for p in runs), default=d.stat().st_mtime)
        cfg = {}
        if (d / "config.json").exists():
            cfg = json.loads((d / "config.json").read_text())
        w = cfg.get("winner", {})
        print(f"  {d.name}\n    path     {d}\n    runs     {len(runs)}   last written "
              f"{dt.datetime.fromtimestamp(latest):%Y-%m-%d %H:%M}\n"
              f"    frozen   trial {w.get('trial')}  {w.get('combo')}  init seed {w.get('init_seed')}\n"
              f"    outputs  {sorted(p.name for p in d.glob('latex_*.tex'))}\n")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def summarise(group: pd.DataFrame) -> dict:
    """Stats over accepted runs of `group`; counts and chi2 over all of it."""
    ok = group[~group["failed"]]
    acc = group[group["accepted"]]
    k, mm = acc["KsAlpha"].to_numpy(float), acc["m"].to_numpy(float)
    ek = np.abs(k - KSALPHA_TRUE) / KSALPHA_TRUE * 100
    em = np.abs(mm - M_TRUE) / M_TRUE * 100
    n = len(acc)
    q = lambda x, p: np.percentile(x, p) if n else np.nan
    return dict(
        n_total=len(group), n_failed=int(group["failed"].sum()),
        n_admissible=int(group["admissible"].sum()), n_accepted=n,
        KsAlpha_med=q(k, 50), KsAlpha_q1=q(k, 25), KsAlpha_q3=q(k, 75),
        KsAlpha_mean=k.mean() if n else np.nan, KsAlpha_std=k.std(ddof=1) if n > 1 else np.nan,
        m_med=q(mm, 50), m_q1=q(mm, 25), m_q3=q(mm, 75),
        m_mean=mm.mean() if n else np.nan, m_std=mm.std(ddof=1) if n > 1 else np.nan,
        bias_KsAlpha_med_pct=(q(k, 50) / KSALPHA_TRUE - 1) * 100,
        bias_m_med_pct=(q(mm, 50) / M_TRUE - 1) * 100,
        err_KsAlpha_med=q(ek, 50), err_KsAlpha_max=ek.max() if n else np.nan,
        err_m_med=q(em, 50), err_m_max=em.max() if n else np.nan,
        n_underestimate_KsAlpha=int((k < KSALPHA_TRUE).sum()),
        r_KsAlpha_m=np.corrcoef(k, mm)[0, 1] if n > 2 else np.nan,
        chi2_train_med_all=ok["chi2_nu_train"].median() if len(ok) else np.nan,
        chi2_test_med_all=ok["chi2_nu_test"].median() if len(ok) else np.nan,
        physics_max_accepted=acc["physics_value"].max() if n else np.nan,
        err_max_rejected=(100 * group.loc[~group["accepted"] & ~group["failed"], "err_KsAlpha"].max()
                          if "err_KsAlpha" in group and (~group["accepted"] & ~group["failed"]).any()
                          else np.nan),
    )


# ---------------------------------------------------------------------------
# LaTeX: compact tabulars (portrait, \footnotesize), \input'ed from the SI
# ---------------------------------------------------------------------------

def _n(x, fmt):
    return "---" if x is None or not np.isfinite(x) else r"\num{" + format(x, fmt) + "}"


def _sn(x, fmt):  # signed
    return "---" if not np.isfinite(x) else r"\num[retain-explicit-plus]{" + format(x, "+" + fmt) + "}"


def _pair(a, b):
    return "---" if a == "---" or b == "---" else f"{a} / {b}"


def _miqr(s, key, fmt):
    """Median on the first line, [Q1, Q3] on the second."""
    if not np.isfinite(s[f"{key}_med"]):
        return "---"
    med = r"\num{" + format(s[key + "_med"], fmt) + "}"
    iqr = ("{[" + r"\num{" + format(s[key + "_q1"], fmt) + r"}, \num{"
           + format(s[key + "_q3"], fmt) + "}" + "]}")
    return r"\makecell{" + med + r"\\" + iqr + "}"


def latex_noise_table(s_init, s_noise) -> str:
    rows = []
    if s_init is not None:
        rows.append((r"\makecell[l]{Initialisation seed\\(noise fixed)}", s_init))
    rows.append((r"\makecell[l]{Noise realisation\\(init.\ fixed)}", s_noise))
    lines = [
        r"{\renewcommand{\arraystretch}{1.15}\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l c c c c c c}",
        r"\toprule",
        r"Ensemble & Acc./Total"
        r" & \makecell{$K_s/\alpha$ [\si{cm^2/h}]\\median [IQR]}"
        r" & \makecell{$m$\\median [IQR]}"
        r" & \makecell{$b_{K_s/\alpha}$ / $b_{m}$\\{[\%]}}"
        r" & \makecell{$\tilde\varepsilon_{K_s/\alpha}$ / $\tilde\varepsilon_{m}$\\{[\%]}}"
        r" & $r$ \\",
        r"\midrule",
    ]
    for name, s in rows:
        lines.append(
            f"{name} & {s['n_accepted']}/{s['n_total']} & "
            f"{_miqr(s, 'KsAlpha', '.1f')} & {_miqr(s, 'm', '.3f')} & "
            f"{_pair(_sn(s['bias_KsAlpha_med_pct'], '.1f'), _sn(s['bias_m_med_pct'], '.1f'))} & "
            f"{_pair(_n(s['err_KsAlpha_med'], '.1f'), _n(s['err_m_med'], '.1f'))} & "
            f"{_n(s['r_KsAlpha_m'], '.2f')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}}", ""]
    return "\n".join(lines)


def latex_density_table(rows: list) -> str:
    lines = [
        r"{\renewcommand{\arraystretch}{1.15}\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{c c c c c c c}",
        r"\toprule",
        r"\makecell{$N_{\mathrm{train}}$\\($f$)} & Acc./Total"
        r" & \makecell{$K_s/\alpha$ [\si{cm^2/h}]\\median [IQR]}"
        r" & \makecell{$m$\\median [IQR]}"
        r" & \makecell{$\tilde\varepsilon_{K_s/\alpha}$\\{[\%]}}"
        r" & \makecell{$\tilde\varepsilon_{m}$\\{[\%]}}"
        r" & \makecell{med.\ $\chi^2_{\nu}$\\train / test} \\",
        r"\midrule",
    ]
    for s in rows:
        lines.append(
            f"{s['n_train']} ({_n(s['fraction'], '.3g')}) & {s['n_accepted']}/{s['n_total']} & "
            f"{_miqr(s, 'KsAlpha', '.1f')} & {_miqr(s, 'm', '.3f')} & "
            f"{_n(s['err_KsAlpha_med'], '.1f')} & {_n(s['err_m_med'], '.1f')} & "
            f"{_pair(_n(s['chi2_train_med_all'], '.2f'), _n(s['chi2_test_med_all'], '.2f'))} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(df: pd.DataFrame, init_seed, path_base: Path, usetex: bool):
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "text.usetex": usetex,
        "font.size": 11, "axes.grid": True, "grid.color": "#d9d9d9", "grid.linewidth": 0.6,
    })
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.0), constrained_layout=True)

    a = ax[0]
    full = df[(df["fraction"] == df["fraction"].max()) & ~df["failed"]]
    acc = full[full["accepted"]]
    a.scatter(acc["KsAlpha"], acc["m"], s=36, color=C_NOISE, edgecolors="white",
              linewidths=0.8, label="Noise realisation (fixed init.)", zorder=3)
    shown = [acc]
    if init_seed is not None:
        sb = init_seed[init_seed["accepted"]]
        a.scatter(sb["KsAlpha"], sb["m"], s=70, facecolors="none", edgecolors="k",
                  linewidths=1.0, label="Init. seed (fixed noise)", zorder=4)
        shown.append(sb)
    pts = pd.concat(shown)
    kx = np.r_[pts["KsAlpha"].to_numpy(float), KSALPHA_TRUE]
    my = np.r_[pts["m"].to_numpy(float), M_TRUE]
    dx, dy = np.ptp(kx) * 0.12 + 1e-9, np.ptp(my) * 0.12 + 1e-9
    xlim, ylim = (kx.min() - dx, kx.max() + dx), (my.min() - dy, my.max() + dy)
    rej = full[~full["accepted"]]
    inside = rej[rej["KsAlpha"].between(*xlim) & rej["m"].between(*ylim)]
    if len(inside):
        a.scatter(inside["KsAlpha"], inside["m"], s=40, marker="x", color=C_REJ,
                  label="Rejected (obs. gate)", zorder=3)
    n_out = len(rej) - len(inside)
    if n_out:
        a.text(0.97, 0.03, f"{n_out} rejected run(s) off-scale", transform=a.transAxes,
               ha="right", va="bottom", fontsize=9, color=C_REJ)
    a.set_xlim(*xlim)
    a.set_ylim(*ylim)
    a.axvline(KSALPHA_TRUE, color="k", ls="--", lw=1.0)
    a.axhline(M_TRUE, color="k", ls="--", lw=1.0)
    a.set_xlabel(r"$K_s/\alpha$ (cm$^2$/h)")
    a.set_ylabel(r"$m$")
    a.legend(loc="upper right", fontsize=9, framealpha=1.0)
    a.text(0.03, 0.05, "a)", transform=a.transAxes, va="bottom")

    ok = df[~df["failed"]]
    levels = sorted(df["n_train"].unique())
    rng = np.random.default_rng(0)
    jit = {n: rng.uniform(-0.15, 0.15, 200) for n in levels}
    for axi, col, label, tag in ((ax[1], "err_KsAlpha", r"$K_s/\alpha$", "b)"),
                                 (ax[2], "err_m", r"$m$", "c)")):
        pos = np.arange(len(levels))
        data = [100 * ok.loc[(ok["n_train"] == n) & ok["accepted"], col].to_numpy() for n in levels]
        axi.boxplot([d if len(d) else [np.nan] for d in data], positions=pos, widths=0.5,
                    showfliers=False, patch_artist=True, whis=(0, 100),
                    medianprops=dict(color=C_NOISE, lw=2),
                    boxprops=dict(facecolor="white", edgecolor=C_NOISE, lw=1.2),
                    whiskerprops=dict(color=C_NOISE), capprops=dict(color=C_NOISE))
        ymax = 100 * ok[col].max()
        for i, n in enumerate(levels):
            sub = ok[ok["n_train"] == n]
            ya = 100 * sub.loc[sub["accepted"], col].to_numpy()
            axi.scatter(i + jit[n][:len(ya)], ya, s=14, color=C_NOISE, zorder=3,
                        edgecolors="white", linewidths=0.4)
            yb = 100 * sub.loc[~sub["accepted"], col].to_numpy()
            if len(yb):
                axi.scatter(i + jit[n][100:100 + len(yb)], yb, s=22, marker="x", color=C_REJ, zorder=3)
            tot = int((df["n_train"] == n).sum())
            axi.annotate(f"{len(ya)}/{tot}", (i, 1.0), xycoords=("data", "axes fraction"),
                         xytext=(0, -12), textcoords="offset points", ha="center", fontsize=9)
        axi.set_xticks(pos, [str(n) for n in levels])
        axi.set_yscale("log")
        axi.set_ylim(top=ymax * 3)
        axi.set_xlabel(r"$N_{\mathrm{train}}$")
        axi.set_ylabel(f"Relative error in {label} (%)")
        axi.text(0.03, 0.05, tag, transform=axi.transAxes, va="bottom")

    for ext in ("eps", "pdf"):
        fig.savefig(path_base.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(paths.results_root()),
                   help="Results root that contains hp_vpinn/ (default: <repo>/results).")
    p.add_argument("--list-studies", action="store_true",
                   help="List the sensitivity studies under --root and exit.")
    p.add_argument("--schedule", default="gb")
    p.add_argument("--noise-level", type=float, default=0.01)
    p.add_argument("--subsample", default="stride")
    p.add_argument("--study-dir", default=None, help="Override the sensitivity study dir.")
    p.add_argument("--results", default=None, help="Search results .jsonl (default: from config.json).")
    p.add_argument("--gate", choices=["calibrated", "admissible"], default="calibrated")
    p.add_argument("--physics-metric", choices=["eval", "train"], default="eval",
                   help="Admissibility metric: held-out-quadrature PDE loss (eval, default) "
                        "or training-quadrature physics loss (train).")
    p.add_argument("--obj-threshold", type=float, default=None,
                   help="Calibration threshold on chi2_train + chi2_test (default 4).")
    p.add_argument("--out", default=None, help="Output dir (default: the study dir).")
    p.add_argument("--usetex", action="store_true")
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    if args.list_studies:
        list_studies(root)
        return

    global OBJ_THRESHOLD
    if args.obj_threshold is not None:
        OBJ_THRESHOLD = args.obj_threshold

    noise_tag = f"{round(args.noise_level * 100):01d}pct_noise"
    suffix = "" if args.subsample == "stride" else f"_{args.subsample}"
    tag = f"grenoblesand_{args.schedule}_{noise_tag}{suffix}"
    study_dir = (Path(args.study_dir).expanduser() if args.study_dir
                 else root / "hp_vpinn" / f"sensitivity_{tag}").resolve()
    out = Path(args.out).expanduser().resolve() if args.out else study_dir
    print(f"Root        {root}\nStudy dir   {study_dir}\nOutput dir  {out}")
    out.mkdir(parents=True, exist_ok=True)

    raw = load_runs_raw(study_dir / "runs")
    cfg_path = study_dir / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    winner = cfg.get("winner", {})
    combo = winner.get("combo") or {k: (raw[k].iloc[0] if k == "activation" else int(raw[k].iloc[0]))
                                    for k in HP_COLS}
    res_path = Path(args.results).expanduser() if args.results else Path(winner.get("results_path", ""))
    if not res_path.is_file():
        res_path = root / "hp_vpinn" / f"results_grenoblesand_{args.schedule}_{noise_tag}" / f"hparam_results_{noise_tag}.jsonl"
    init_raw = load_init_seed_raw(res_path, combo)

    frames = {"sensitivity-run": raw}
    if init_raw is not None:
        frames["search-trial"] = init_raw
    phys_col = resolve_physics_col(frames, args.physics_metric)
    print(f"Gate: {phys_col} < {PHYSICS_THRESHOLD:g} and chi2_train+chi2_test < {OBJ_THRESHOLD:g}")

    df = apply_gate(raw, args.gate, phys_col)
    init_seed = apply_gate(init_raw, args.gate, phys_col) if init_raw is not None else None

    trial = int(df["winner_trial"].iloc[0])
    print(f"Frozen config: trial {trial}, {combo}, init seed {int(df['init_seed'].iloc[0])}")
    print(f"Runs: {len(df)}   diverged/failed: {int(df['failed'].sum())}   "
          f"admissible: {int(df['admissible'].sum())}   accepted ({args.gate}): {int(df['accepted'].sum())}")
    if init_seed is not None:
        print(f"Init-seed ensemble (search trials of the combination): {int(init_seed['accepted'].sum())}"
              f"/{len(init_seed)} accepted  -> compare with N and completed runs of this combination "
              f"in tab:si_winners")

    # regression check: r=0 at full density vs the frozen trial itself
    r0 = df[(df["realisation"] == 0) & (df["fraction"] == df["fraction"].max()) & ~df["failed"]]
    if len(r0) and init_seed is not None and (init_seed["trial"] == trial).any():
        t = init_seed[init_seed["trial"] == trial].iloc[0]
        dk = abs(r0.iloc[0]["KsAlpha"] / t["KsAlpha"] - 1)
        dm = abs(r0.iloc[0]["m"] / t["m"] - 1)
        flag = "OK" if max(dk, dm) < 1e-6 else "DIFFERS (acceptable only across hardware/software)"
        print(f"r=0 vs trial {trial}: KsAlpha {r0.iloc[0]['KsAlpha']:.4g} vs {t['KsAlpha']:.4g}, "
              f"m {r0.iloc[0]['m']:.4g} vs {t['m']:.4g} -> {flag}")

    f_max = df["fraction"].max()
    s_noise = summarise(df[df["fraction"] == f_max])
    s_init = summarise(init_seed) if init_seed is not None else None

    dens = []
    for f in sorted(df["fraction"].unique(), reverse=True):
        g = df[df["fraction"] == f]
        s = summarise(g)
        s.update(fraction=f, n_train=int(g["n_train"].iloc[0]))
        dens.append(s)

    (out / f"latex_noise_realisations_{tag}.tex").write_text(latex_noise_table(s_init, s_noise))
    (out / f"latex_data_density_{tag}.tex").write_text(latex_density_table(dens))
    rows = ([dict(ensemble="init_seed", **s_init)] if s_init else []) + \
           [dict(ensemble="noise_realisation", **s_noise)] + \
           [dict(ensemble=f"density_N{s['n_train']}", **s) for s in dens]
    pd.DataFrame(rows).to_csv(out / f"sensitivity_{tag}_summary.csv", index=False)
    keep = ["run_index", "realisation", "noise_seed", "fraction", "n_train", "KsAlpha", "m",
            "err_KsAlpha", "err_m", "chi2_nu_train", "chi2_nu_test", "obj", "final_physics",
            "pde_loss_eval", "physics_value", "obj_gate", "failed", "admissible", "calibrated", "accepted"]
    df[[c for c in keep if c in df]].to_csv(out / f"sensitivity_{tag}_runs.csv", index=False)
    make_figure(df, init_seed, out / f"sensitivity_{tag}", args.usetex)

    cols = ["n_accepted", "n_total", "KsAlpha_med", "m_med", "bias_KsAlpha_med_pct", "bias_m_med_pct",
            "err_KsAlpha_med", "err_m_med", "r_KsAlpha_m", "chi2_train_med_all", "chi2_test_med_all",
            "physics_max_accepted"]
    print("\n", pd.DataFrame(rows).set_index("ensemble")[cols].to_string(float_format=lambda v: f"{v:.4g}"))
    print(f"\nNoise ensemble: {s_noise['n_underestimate_KsAlpha']} of {s_noise['n_accepted']} accepted "
          f"runs underestimate K_s/alpha; max relative error of accepted runs "
          f"K_s/alpha {max(s['err_KsAlpha_max'] for s in dens if np.isfinite(s['err_KsAlpha_max'])):.1f}%, "
          f"m {max(s['err_m_max'] for s in dens if np.isfinite(s['err_m_max'])):.1f}%")
    print(f"Largest {phys_col} among accepted runs (all densities): "
          f"{max(s['physics_max_accepted'] for s in dens if np.isfinite(s['physics_max_accepted'])):.3g}")
    print(f"\nWritten to {out}:")
    for f in sorted(out.glob(f"*{tag}*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()

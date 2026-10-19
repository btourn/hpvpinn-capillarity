#!/usr/bin/env python
"""Post-hoc validation of the EMA/patience early-stopping rule (R2-M08). NumPy/SciPy/Matplotlib only.

Reads the stored histories of the final runs, hp_vpinn/final_runs/<tag>/seeds/seed_<n>/{record.json,history.npz}
(written by final_run_common.py), and, WITHOUT training anything, evaluates:

  1. the rule of the manuscript (EMA of the physics loss, beta_s=0.99, below tau for W consecutive epochs)
     -> reproduces ema_freeze_epoch of every record (consistency check), the projected stop time, and how far
        the hydraulic parameters / chi2 still are from their end-of-training values at the stop epoch;
  2. its sensitivity to (tau, W);
  3. a conjunctive variant: the same rule AND stationarity of the parameters (relative change over the last W
     epochs below eps for every identified parameter).

Any stopping rule that only reads the past is evaluated exactly by truncating a stored history, so the stop
epoch below is the one an online implementation would produce; only the stop *time* is a projection
(t_compile + stop_epoch * per_epoch).

Usage (from the repo root):
  python hp_vpinn/early_stopping_analysis.py --root hp_vpinn/final_runs --out hp_vpinn/Figures/early_stopping
  # optional: --tags grenoblesand_gb_1pct_noise grenoblesand_gb_2pct_noise letd_gb
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
from scipy.signal import lfilter

BETA = 0.99                      # beta_s of eq:ema_physics  (train_vpinn.py: ema_beta + 0.09)
TAU0, W0 = 5e-8, 10_000          # values hard-coded in train_vpinn.py
TAUS = [1e-9, 2e-9, 5e-9, 1e-8, 2e-8, 5e-8, 1e-7, 5e-7, 1e-6]
WS = [1_000, 5_000, 10_000, 20_000]
EPS_GRID = [0.0025, 0.005, 0.01, 0.02, 0.05]
EPS0 = 0.01                      # fixed a priori (well below the seed-to-seed spread of the parameters)
MIN_ABS = 1e-6                   # parameters below this (bound-hugging) are not tested
IGNORE = {"theta_r"}             # parameters not identified by the data (SI sec:comparison_with_fronts_detail); set by --ignore-params
QUAL_OBJ, QUAL_PDE = 4.0, 1e-5   # calibrated / admissible thresholds of sec:hpsearch
TRUTH = {"grenoblesand": {"KsAlpha": 355.783, "m": 0.5096}}   # consistent with rel_errors stored in the records
LABELS = {"grenoblesand_gb_1pct_noise": r"Case~1, \num{1}\% noise", "grenoblesand_gb_2pct_noise": r"Case~1, \num{2}\% noise",
          "letd_gb": "Case~3"}
SHORT = {"grenoblesand_gb_1pct_noise": "Case 1, 1% noise", "grenoblesand_gb_2pct_noise": "Case 1, 2% noise",
         "letd_gb": "Case 3 (LETd)"}


# ----------------------------------------------------------------------------- rules
def ema(phys, beta=BETA):
    """EMA initialised at zero, exactly as in the training step (checked against the stored ema_physics)."""
    return lfilter([1 - beta], [1, -beta], phys)


def run_length(below):
    idx = np.arange(len(below))
    last_false = np.maximum.accumulate(np.where(~below, idx, -1))
    return np.where(below, idx - last_false, 0)


def stop_epoch(e, tau, W):
    """First epoch at which e < tau has held for W consecutive epochs (the freeze of train_vpinn.py)."""
    hit = np.flatnonzero(run_length(e < tau) >= W)
    return int(hit[0]) if hit.size else None


def stationarity(P, W, min_abs=MIN_ABS):
    """max_i |p_i[k]-p_i[k-W]| / |p_i[k]| over the identified parameters; NaN for k<W."""
    keep = [p for p in P.values() if abs(p[-1]) > min_abs]
    s = np.full(len(next(iter(P.values()))), np.nan)
    if not keep:
        return s
    s[W:] = np.max([np.abs(p[W:] - p[:-W]) / np.abs(p[W:]) for p in keep], axis=0)
    return s


def stop_conjunctive(e, P, tau, W, eps):
    armed = run_length(e < tau) >= W
    stat = stationarity(P, W)
    hit = np.flatnonzero(armed & (stat < eps))
    return int(hit[0]) if hit.size else None


def deviation(P, k, ref=None):
    """max / per-parameter relative deviation of the parameters at epoch k from ref (default: end of training)."""
    d = {}
    for n, p in P.items():
        r = p[-1] if ref is None else ref[n]
        d[n] = abs(p[k] - r) / abs(r) if (abs(r) > MIN_ABS and n not in IGNORE) else np.nan
    return d


# ----------------------------------------------------------------------------- loading
def load_tag(tag_dir):
    runs = []
    for sd in sorted((tag_dir / "seeds").glob("seed_*")):
        rp, hp = sd / "record.json", sd / "history.npz"
        if not (rp.exists() and hp.exists()):
            continue
        rec = json.loads(rp.read_text())
        if rec.get("failed"):
            continue
        h = np.load(hp)
        P = {k[len("param_"):]: h[k].astype(float) for k in h.files if k.startswith("param_")}
        runs.append(dict(rec=rec, phys=h["physics_loss"].astype(float), chi=h["data_loss"].astype(float), P=P,
                         ema_stored=h["ema_physics"].astype(float),
                         frozen=np.asarray(h["ema_physics_frozen"]).astype(bool)))
    return runs


def qualified(rec):
    m = rec.get("metrics", {})
    return bool(rec.get("obj", 1e9) < QUAL_OBJ and m.get("pde_loss_eval", 1e9) < QUAL_PDE)


def med(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.median(x)) if x else float("nan")


def rng(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return (float(np.min(x)), float(np.max(x))) if x else (float("nan"), float("nan"))


# ----------------------------------------------------------------------------- per-tag analysis
def analyse(tag, runs):
    case = tag.split("_")[0]
    truth = TRUTH.get(case)
    rows, grid = [], {}
    for r in runs:
        rec, P, phys, chi = r["rec"], r["P"], r["phys"], r["chi"]
        N = len(phys); e = ema(phys)
        f_rec = rec.get("ema_freeze_epoch")
        f = stop_epoch(e, TAU0, W0)
        pre = slice(500, f if f else N)
        ema_err = float(np.max(np.abs(e[pre] - r["ema_stored"][pre]) / r["ema_stored"][pre]))
        t = rec["timing"]; pe = t["per_epoch_ms"]; tc = t["t_compile_s"] or 0.0
        row = dict(seed=rec["seed"], qualified=qualified(rec), freeze_record=f_rec, freeze_recomputed=f,
                   consistent=(f == f_rec), ema_rel_err=ema_err, n_epochs=N, t_fit_s=t["t_fit_s"], ms_per_epoch=pe,
                   obj=rec.get("obj"), chi2_final=chi[-1])
        row["ema_at_freeze"] = e[f] if f else None
        row["transition_end"] = (rec.get("warmup_epochs") or 0) + (rec.get("transition_epochs") or 0)
        row["freeze_in_transition"] = bool(f is not None and f <= row["transition_end"])
        k = f if f else N - 1
        dv = deviation(P, k)
        row.update(t_stop_s=(tc + f * pe / 1e3) if f else None, chi2_at_stop=float(chi[max(k-50, 0):k+50].mean()) if f else None,
                   dev_max=(np.nanmax(list(dv.values())) if f else None))
        for n, p in P.items():
            row[f"{n}_stop"], row[f"{n}_final"], row[f"{n}_dev"] = (p[k] if f else None), p[-1], (dv[n] if f else None)
            if truth and n in truth:
                row[f"{n}_err_stop"] = abs(p[k] - truth[n]) / truth[n] if f else None
                row[f"{n}_err_final"] = abs(p[-1] - truth[n]) / truth[n]
        # sensitivity grid
        for tau in TAUS:
            for W in WS:
                s = stop_epoch(e, tau, W)
                grid.setdefault((tau, W), []).append(
                    dict(stop=s, dev=(np.nanmax(list(deviation(P, s).values())) if s else None), qual=row["qualified"]))
        # conjunctive variant (tau, W as the manuscript; eps grid)
        for eps in EPS_GRID:
            s = stop_conjunctive(e, P, TAU0, W0, eps)
            row[f"conj_stop_{eps}"] = s
            row[f"conj_dev_{eps}"] = float(np.nanmax(list(deviation(P, s).values()))) if s else None
            row[f"conj_t_{eps}"] = (tc + s * pe / 1e3) if s else None
        row["_e"] = e
        rows.append(row)
    return rows, grid


def summarise(tag, rows, grid):
    n = len(rows); fr = [r for r in rows if r["freeze_recomputed"]]
    S = dict(tag=tag, n=n, n_qualified=sum(r["qualified"] for r in rows), n_frozen=len(fr),
             n_frozen_qualified=sum(r["qualified"] for r in fr), all_consistent=all(r["consistent"] for r in rows),
             max_ema_rel_err=max(r["ema_rel_err"] for r in rows),
             t_fit_med=med([r["t_fit_s"] for r in rows]), t_fit_rng=rng([r["t_fit_s"] for r in rows]),
             ms_epoch_med=med([r["ms_per_epoch"] for r in rows]),
             freeze_med=med([r["freeze_recomputed"] for r in fr]), freeze_rng=rng([r["freeze_recomputed"] for r in fr]),
             t_stop_med=med([r["t_stop_s"] for r in fr]), t_stop_rng=rng([r["t_stop_s"] for r in fr]),
             dev_med=med([r["dev_max"] for r in fr]), dev_rng=rng([r["dev_max"] for r in fr]),
             chi2_stop_med=med([r["chi2_at_stop"] for r in fr]), chi2_final_med=med([r["chi2_final"] for r in rows]),
             ema_at_freeze_rng=rng([r["ema_at_freeze"] for r in fr]))
    S["params_dev_med"] = {k[:-4]: med([r[k] for r in fr]) for k in rows[0] if k.endswith("_dev") and k != "dev_max"}
    S["err_stop_med"] = {k[:-9]: med([r[k] for r in fr]) for k in rows[0] if k.endswith("_err_stop")}
    S["err_final_med"] = {k[:-10]: med([r[k] for r in rows]) for k in rows[0] if k.endswith("_err_final")}
    S["n_freeze_in_transition"] = sum(r["freeze_in_transition"] for r in fr)
    S["transition_end"] = rows[0]["transition_end"]
    S["frac_dev_gt_5pct"] = sum(1 for r in fr if r["dev_max"] is not None and r["dev_max"] > 0.05) / max(len(fr), 1)
    S["per_seed"] = [dict(seed=r["seed"], qualified=r["qualified"], freeze=r["freeze_recomputed"], t_stop=r["t_stop_s"],
                          dev_max=r["dev_max"], chi2_stop=r["chi2_at_stop"], chi2_final=r["chi2_final"],
                          dev={k[:-4]: r[k] for k in r if k.endswith("_dev") and k != "dev_max"},
                          conj1=r["conj_stop_0.01"], conj1_dev=r["conj_dev_0.01"], conj2=r["conj_stop_0.02"], conj2_dev=r["conj_dev_0.02"])
                     for r in rows]
    S["conj"] = {}
    for eps in EPS_GRID:
        c = [r for r in rows if r[f"conj_stop_{eps}"]]
        S["conj"][str(eps)] = dict(n_stop=len(c), stop_med=med([r[f"conj_stop_{eps}"] for r in c]),
                                   t_med=med([r[f"conj_t_{eps}"] for r in c]),
                                   dev_med=med([r[f"conj_dev_{eps}"] for r in c]), dev_max=rng([r[f"conj_dev_{eps}"] for r in c])[1])
    S["grid"] = {f"{tau:g}|{W}": dict(n_stop=sum(g["stop"] is not None for g in v), stop_med=med([g["stop"] for g in v]),
                                      dev_med=med([g["dev"] for g in v]))
                 for (tau, W), v in grid.items()}
    return S


# ----------------------------------------------------------------------------- LaTeX
def num(x, fmt=".3g"):
    return "---" if x is None or not np.isfinite(x) else rf"\num{{{format(x, fmt)}}}"


def mr(m, r, fmt=".3g"):
    return f"{num(m, fmt)} ({num(r[0], fmt)}--{num(r[1], fmt)})"


def latex_main(summaries, pinn_lb):
    L = [r"\begin{tabular}{llcccccc}", r"\toprule",
         r"\textbf{Config.} & \textbf{Case} & \textbf{ms/epoch} & \textbf{Full budget [s]} & \textbf{Freeze epoch} & "
         r"\textbf{Stopped} & \textbf{Early stop [s]} & \textbf{$\Delta p$ at stop [\%]} \\", r"\midrule"]
    for i, S in enumerate(summaries):
        first = r"\multirow{%d}{*}{\hpVPINN-gb}" % len(summaries) if i == 0 else ""
        L.append(rf"{first} & {LABELS.get(S['tag'], S['tag'])} & {num(S['ms_epoch_med'], '.2f')} & {mr(S['t_fit_med'], S['t_fit_rng'], '.0f')} & "
                 rf"{mr(S['freeze_med'], S['freeze_rng'], '.0f')} & {S['n_frozen']}/{S['n']} & {mr(S['t_stop_med'], S['t_stop_rng'], '.0f')} & "
                 rf"{num(100 * S['dev_med'], '.1f')} \\")
    if pinn_lb:
        L.append(r"\midrule")
        for i, S in enumerate(summaries):
            first = r"\multirow{%d}{*}{PINN-gb}" % len(summaries) if i == 0 else ""
            L.append(rf"{first} & {LABELS.get(S['tag'], S['tag'])} & --- & $>${pinn_lb:g}\footnotemark[1] & --- & --- & --- & --- \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def latex_si(summaries):
    L = [r"% Sensitivity of the stop epoch (median over seeds; k/n = seeds that stop) and of the parameter deviation at the stop epoch",
         r"\begin{tabular}{l" + "c" * len(WS) + "}", r"\toprule",
         r"$\tau_{\mathrm{stop}}$ & " + " & ".join(rf"$W_{{\mathrm{{stop}}}}={w}$" for w in WS) + r" \\"]
    for S in summaries:
        L += [r"\midrule", rf"\multicolumn{{{len(WS) + 1}}}{{l}}{{{LABELS.get(S['tag'], S['tag'])}}} \\"]
        for tau in TAUS:
            cells = []
            for W in WS:
                g = S["grid"][f"{tau:g}|{W}"]
                cells.append("---" if g["n_stop"] == 0 else f"{num(g['stop_med'] / 1e3, '.0f')}k ({g['n_stop']}/{S['n']}), {num(100 * g['dev_med'], '.0f')}\\%")
            L.append(rf"\num{{{tau:g}}} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", ""]
    L += [r"% Conjunctive rule: EMA below tau_stop for W_stop epochs AND parameter change over the last W_stop epochs below eps",
          r"\begin{tabular}{llcccc}", r"\toprule",
          r"Case & $\varepsilon$ & Stopped & Stop epoch & Time [s] & $\Delta p$ at stop [\%] (median / max) \\"]
    for S in summaries:
        L.append(r"\midrule")
        for eps, c in S["conj"].items():
            L.append(rf"{LABELS.get(S['tag'], S['tag'])} & {float(eps) * 100:g}\% & {c['n_stop']}/{S['n']} & {num(c['stop_med'], '.0f')} & "
                     rf"{num(c['t_med'], '.0f')} & {num(100 * c['dev_med'], '.1f')} / {num(100 * c['dev_max'], '.1f')} \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- figures
def figures(data, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nc = len(data)
    fig, ax = plt.subplots(2, nc, figsize=(4.6 * nc, 6.6), squeeze=False, sharex=True)
    for j, (tag, rows, runs) in enumerate(data):
        for r, run in zip(rows, runs):
            col = "tab:blue" if r["qualified"] else "tab:gray"
            ep = np.arange(0, r["n_epochs"], 50)
            e = r["_e"]
            ax[0, j].semilogy(ep, e[ep], color=col, lw=0.7, alpha=0.8)
            P = run["P"]
            dev = np.max([np.abs(p - p[-1]) / abs(p[-1]) for n, p in P.items() if n not in IGNORE and abs(p[-1]) > MIN_ABS], axis=0)
            ax[1, j].semilogy(ep, np.maximum(dev[ep], 1e-4), color=col, lw=0.7, alpha=0.8)
            f = r["freeze_recomputed"]
            if f:
                ax[0, j].plot(f, e[f], "o", color="tab:red", ms=4)
                ax[1, j].plot(f, max(dev[f], 1e-4), "o", color="tab:red", ms=4)
        ax[0, j].axhline(TAU0, ls="--", color="k", lw=0.8)
        ax[1, j].axhline(EPS0, ls=":", color="k", lw=0.8)
        ax[0, j].set_title(SHORT.get(tag, tag)); ax[1, j].set_xlabel("Epochs")
        ax[0, j].set_ylim(1e-10, 1e-5); ax[1, j].set_ylim(1e-4, 1)
    ax[0, 0].set_ylabel(r"EMA of $\mathcal{L}_{\rm phys}$"); ax[1, 0].set_ylabel(r"$\max_i|p_i-p_i^{\rm final}|/|p_i^{\rm final}|$")
    for a in ax.ravel():
        a.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("eps", "pdf", "png"):
        fig.savefig(out / f"fig_early_stopping.{ext}", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(2, nc, figsize=(4.4 * nc, 7.0), squeeze=False)
    for j, S in enumerate(s for _, _, s in [(t, r, summaries_by_tag[t]) for t, r, _ in data]):
        for i, (key, lab, scale, fmt) in enumerate([("stop_med", "median stop epoch [k]", 1e-3, ".0f"),
                                                    ("dev_med", r"median $\Delta p$ at stop [%]", 100, ".0f")]):
            M = np.array([[S["grid"][f"{tau:g}|{W}"][key] * scale if S["grid"][f"{tau:g}|{W}"]["n_stop"] else np.nan for W in WS] for tau in TAUS])
            im = ax[i, j].imshow(M, aspect="auto", cmap="viridis")
            for a_ in range(len(TAUS)):
                for b_ in range(len(WS)):
                    g = S["grid"][f"{TAUS[a_]:g}|{WS[b_]}"]
                    ax[i, j].text(b_, a_, "none" if g["n_stop"] == 0 else format(M[a_, b_], fmt), ha="center", va="center", color="w", fontsize=7)
            ax[i, j].set_xticks(range(len(WS))); ax[i, j].set_xticklabels(WS, fontsize=7)
            ax[i, j].set_yticks(range(len(TAUS))); ax[i, j].set_yticklabels([f"{t:g}" for t in TAUS], fontsize=7)
            ax[i, j].set_title(f"{SHORT.get(S['tag'], S['tag'])}: {lab}", fontsize=8)
            ax[i, j].add_patch(plt.Rectangle((WS.index(W0) - .5, TAUS.index(TAU0) - .5), 1, 1, fill=False, ec="red", lw=1.5))
            ax[i, j].set_xlabel(r"$W_{\rm stop}$"); ax[i, j].set_ylabel(r"$\tau_{\rm stop}$")
    fig.tight_layout()
    for ext in ("eps", "pdf", "png"):
        fig.savefig(out / f"fig_early_stopping_sensitivity.{ext}", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    global summaries_by_tag
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="hp_vpinn/final_runs")
    ap.add_argument("--out", default="hp_vpinn/Figures/early_stopping")
    ap.add_argument("--tags", nargs="*", default=None, help="default: every <root>/*gb* directory")
    ap.add_argument("--ignore-params", nargs="*", default=["theta_r"], help="parameters excluded from stationarity/deviation")
    ap.add_argument("--pinn-lower-bound", type=float, default=None, help="seconds; adds PINN rows '>x' (informal)")
    a = ap.parse_args()
    global IGNORE
    IGNORE = set(a.ignore_params)
    root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tags = a.tags or sorted(d.name for d in root.iterdir() if d.is_dir() and "_gb" in d.name and not d.name.upper().startswith("OLD"))
    data, summaries, allrows = [], [], []
    for tag in tags:
        runs = load_tag(root / tag)
        if not runs:
            print(f"[skip] {tag}: no completed seeds"); continue
        rows, grid = analyse(tag, runs)
        S = summarise(tag, rows, grid); summaries.append(S); data.append((tag, rows, runs)); allrows += [dict(tag=tag, **{k: v for k, v in r.items() if k != "_e"}) for r in rows]
        print(f"\n{tag}: {S['n']} seeds, {S['n_qualified']} qualified, {S['n_frozen']} froze ({S['n_frozen_qualified']} of them qualified); "
              f"rule reproduces records: {S['all_consistent']} (max EMA rel. err {S['max_ema_rel_err']:.1e})")
        print(f"   freeze epoch median {S['freeze_med']:.0f} {S['freeze_rng']}, t_stop {S['t_stop_med']:.1f}s vs t_fit {S['t_fit_med']:.1f}s, "
              f"param deviation at stop median {100 * S['dev_med']:.1f}% (range {100 * S['dev_rng'][0]:.1f}-{100 * S['dev_rng'][1]:.1f}%)")
        for eps, c in S["conj"].items():
            print(f"   conjunctive eps={float(eps) * 100:g}%: {c['n_stop']}/{S['n']} stop, epoch {c['stop_med']:.0f}, dev {100 * c['dev_med']:.1f}% (max {100 * c['dev_max']:.1f}%)")
    summaries_by_tag = {S["tag"]: S for S in summaries}
    keys = sorted({k for r in allrows for k in r})
    with open(out / "early_stopping_per_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(allrows)
    (out / "early_stopping_summary.json").write_text(json.dumps(summaries, indent=1, default=float))
    (out / "latex_early_stopping_main.tex").write_text(latex_main(summaries, a.pinn_lower_bound))
    (out / "latex_early_stopping_si.tex").write_text(latex_si(summaries))
    figures(data, out)
    print(f"\n[saved] CSV, JSON, LaTeX tables and figures in {out}")


if __name__ == "__main__":
    main()

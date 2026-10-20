"""Post-process Optuna hyperparameter search results and generate summary figures and LaTeX tables.

Two figures per study: A (all cases) chi2_nu_train vs chi2_nu_test with marginal box plots; B (Case 1
only) parameter-recovery scatter, one panel per recovered parameter.

Input: <results>/hp_vpinn/results_<case>_<es|gb>[_<n>pct_noise]/hparam_results_*.jsonl (see
hpvpinn_capillarity.paths). By default figures go to <results>/<model>/Figures, LaTeX tables and the winners
summary to <results>/<model>/tables, and per-run CSVs to the run's own results directory.

--winners builds the winning-architecture summary of every case, configuration and noise level of one model
(non-admissible winners included and flagged): latex_winners.tex, winners.json and winners.csv in
<tables-dir>. The final-run scripts read their configuration from winners.json (see load_winner).

Usage:
    python -m hpvpinn_capillarity.hpo.postprocess --model hp_vpinn --case letd --es-or-gb gb
    python -m hpvpinn_capillarity.hpo.postprocess --model hp_vpinn --case brooks_corey_new --es-or-gb es --noise-level 0.01
    python -m hpvpinn_capillarity.hpo.postprocess --model hp_vpinn --trial-counts
    python -m hpvpinn_capillarity.hpo.postprocess --model hp_vpinn --winners
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from mpl_toolkits.axes_grid1 import make_axes_locatable

from hpvpinn_capillarity import paths


# ---------------------------------------------------------------------------
# LaTeX configuration
# ---------------------------------------------------------------------------
_use_latex = shutil.which("latex") is not None

matplotlib.rcParams.update({
    "text.usetex":            _use_latex,
    "text.latex.preamble":    r"\usepackage{amsbsy}",
    "font.family":            "serif",
    "font.serif":             ["Computer Modern Roman"],
    "axes.labelsize":         11,
    "xtick.labelsize":        9,
    "ytick.labelsize":        9,
    "legend.fontsize":        9,
    "figure.dpi":             200,
})

# ---------------------------------------------------------------------------
# Models and repository layout
# ---------------------------------------------------------------------------

# --model values are also the subdirectory names under the results root, exactly
# as in the search scripts.
MODELS = ("hp_vpinn",)

# Configuration label used in the trial-count table, per model
# (matches the manuscript macros / naming: \hpVPINN-es, PINN-gb, ...).
MODEL_CONFIG_LABEL = {
    "hp_vpinn": r"\hpVPINN-",
}


def model_dir(model: str) -> Path:
    """Root directory of one model's results (<results>/<model>)."""
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}")
    return paths.model_dir(model)


def default_tables_dir(model: str) -> str:
    """Default output directory for LaTeX tables and the winners summary."""
    return str(paths.tables_dir(model))


def default_figures_dir(model: str) -> str:
    """Default output directory for figures and LaTeX tables."""
    return str(model_dir(model) / "Figures")


# ---------------------------------------------------------------------------
# Per-case configuration
# ---------------------------------------------------------------------------

CASES = {
    "grenoblesand": dict(
        label        = "Grenoble Sand",
        label4latex  = "Case 1",
        results_dir  = "results_grenoblesand",
        results_file = "hparam_results_grenoblesand.jsonl",
        true_params  = dict(
            KsAlpha = [r"$K_{\mathrm{s}}/\alpha$", 15.37 / 0.0432],
            m       = [r"$m$", 0.5096],
        ),
        noise_levels = [0.01, 0.02],
    ),
    "vangenuchten": dict(
        label        = "Van Genuchten",
        label4latex  = "Case 2",
        results_dir  = "results_vangenuchten",
        results_file = "hparam_results_vangenuchten.jsonl",
        true_params  = dict(
            KsAlpha = [r"$K_s/\alpha$", 2.079e-6],
            m       = [r"$m$",          0.8764],
            l       = [r"$l$",          2.344],
            theta_r = [r"$\theta_r$",   0.004943],
        ),
    ),
    "letd": dict(
            label        = "LETd",
            label4latex  = "Case 3",
            results_dir  = "results_letd",
            results_file = "hparam_results_letd.jsonl",
            true_params  = dict(
                Dwt     = [r"$D_{wt}$",   1.004e-3],
                L       = [r"$L$",        1.356],
                E       = [r"$E$",        10_010.0],
                T       = [r"$T$",        1.224],
                theta_r = [r"$\theta_r$", 0.00625],
            ),
        ),
    # New Brooks-Corey case: 2 unknowns (KsAlpha = Ks/alpha, n), synthetic data
    # corrupted with 1% noise, same search layout as Grenoble sand.
    # Results: <model>/results_brooks_corey_new_<es|gb>_1pct_noise/
    # KsAlpha "true" value is the one implied by the data file itself
    # (the generator's Ks, alpha and units are unknown); it is used only for
    # the recovery-error columns and plots.
    "brooks_corey_new": dict(
        label        = "Brooks and Corey (new)",
        label4latex  = "Case 4",
        results_dir  = "results_brooks_corey_new",
        results_file = "hparam_results_brooks_corey_new.jsonl",
        true_params  = dict(
            KsAlpha = [r"$K_{\mathrm{s}}/\alpha$", 0.017559],
            n       = [r"$n$",                      1.2876],
        ),
        noise_levels = [0.01],
    ),
    # Original 4-unknown Brooks-Corey case (experimental data). Kept for
    # reference and still runnable with --case brooks_corey, but SKIPPED in the
    # aggregate outputs (trial-counts table, winners summary): the new
    # Brooks-Corey case above takes its place as Case 4.
    "brooks_corey": dict(
        skip_in_summaries = True,
        label        = "Brooks and Corey (previous)",
        label4latex  = "Case 4 (previous)",
        results_dir  = "results_brooks_corey",
        results_file = "hparam_results_brooks_corey.jsonl",
        true_params  = dict(
            KsAlpha = [r"$K_s/\alpha$", 3.983e-6],
            n       = [r"$n$",          0.2837],
            l       = [r"$l$",          4.795],
            theta_r = [r"$\theta_r$",   2.378e-5],
        ),
    )
}

def summary_cases() -> list:
    """Cases included in the aggregate outputs (trial counts, winners),
    ordered by their LaTeX label. Cases flagged skip_in_summaries are left
    out (they remain available through --case)."""
    return sorted(
        (c for c in CASES if not CASES[c].get("skip_in_summaries", False)),
        key=lambda c: CASES[c]["label4latex"],
    )


def has_ground_truth(case: str) -> bool:
    """Cases with synthetic (noisy) data and a known reference solution are
    those that define noise_levels (Grenoble sand, new Brooks-Corey). Recovery
    errors, the recovery figure and the bias/True columns are produced only
    for them."""
    return "noise_levels" in CASES[case]


HPARAM_COLS = ["nn_n_layers", "nn_width", "activation", "n_elements", "poly_degree"]

# ---------------------------------------------------------------------------
# Trial classification and ranking (protocol described in the SI)
# ---------------------------------------------------------------------------
#
#   admissible (physics-consistent):  PHYSICS_METRIC < ADMISSIBLE_PHYSICS
#       PHYSICS_METRIC is the weak-form PDE loss on the HELD-OUT quadrature
#       (pde_loss_eval, Q_EVAL != N_QUAD), not the training-quadrature
#       physics loss, which is biased low by quadrature aliasing. Records
#       without pde_loss_eval (e.g. frontx) fall back to final_physics.
#   calibrated (data not grossly misfit):  obj < CALIBRATED_OBJ
#       with obj = chi2_nu,train + chi2_nu,test, CALIBRATED_OBJ = 4 means a
#       mean chi2_nu below 2. Calibration is REPORTED, it does not gate the
#       ranking: a physics-consistent trial that misfits the data documents
#       model inadequacy, not a training failure.
#   ranking (top-trials tables, winner, identifiability, winners summary):
#       admissible trials first, by obj ascending (tiebreak: physics
#       metric); non-admissible trials after them, by obj. Rank 1 is the
#       winner. Optional OBJ_TIE_TOL > 0: admissible trials within
#       OBJ_TIE_TOL of the best admissible obj are treated as tied and
#       ordered by the physics metric (obj differences below the chi2_nu
#       sampling scatter are not meaningful). Off by default.

CALIBRATED_OBJ     = 4.0             # calibrated: obj < CALIBRATED_OBJ (mean chi2_nu < 2)
ADMISSIBLE_PHYSICS = 1e-5            # admissible: physics metric below this
PHYSICS_METRIC     = "pde_loss_eval" # "pde_loss_eval" (held-out) or "final_physics" (training)
OBJ_TIE_TOL        = 0.0             # tie tolerance on obj among admissible trials
MIN_RUNS           = 10              # a combination needs >= this many qualified (or, in the
                                     # fallback, admissible) runs to be a winner

PHYSICS_METRIC_LABEL = {
    "pde_loss_eval": r"$\mathcal{L}_{\mathrm{pde}}^{\mathrm{eval}}$",
    "final_physics": r"$\mathcal{L}_{\mathrm{physics}}^{\mathrm{train}}$",
}


def physics_values(df: pd.DataFrame) -> pd.Series:
    """Physics metric used for admissibility, per trial.

    PHYSICS_METRIC where available; rows without it (older records, frontx)
    fall back to final_physics.
    """
    if PHYSICS_METRIC in df.columns:
        vals = pd.to_numeric(df[PHYSICS_METRIC], errors="coerce")
        if "final_physics" in df.columns and PHYSICS_METRIC != "final_physics":
            vals = vals.fillna(pd.to_numeric(df["final_physics"], errors="coerce"))
        return vals
    return pd.to_numeric(df["final_physics"], errors="coerce")


def is_admissible(df: pd.DataFrame) -> pd.Series:
    return physics_values(df) < ADMISSIBLE_PHYSICS


def is_calibrated(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df["obj"], errors="coerce") < CALIBRATED_OBJ


def rank_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Return df sorted by the ranking rule above (index reset), with the
    boolean columns 'admissible' and 'calibrated' and the float column
    'physics_metric' added."""
    out = df.copy()
    out["physics_metric"] = physics_values(out)
    out["admissible"] = out["physics_metric"] < ADMISSIBLE_PHYSICS
    out["calibrated"] = is_calibrated(out)
    obj = pd.to_numeric(out["obj"], errors="coerce")

    tier = pd.Series(2, index=out.index)               # non-admissible
    tier[out["admissible"]] = 1                        # admissible
    if OBJ_TIE_TOL > 0 and out["admissible"].any():
        best = obj[out["admissible"]].min()
        tier[out["admissible"] & (obj <= best + OBJ_TIE_TOL)] = 0   # tied at the top
    out["_tier"] = tier
    # tier 0: by physics metric; tiers 1-2: by obj, then physics metric
    out["_k1"] = np.where(tier == 0, out["physics_metric"], obj)
    out["_k2"] = np.where(tier == 0, obj, out["physics_metric"])
    out = out.sort_values(["_tier", "_k1", "_k2"], kind="mergesort")
    return out.drop(columns=["_tier", "_k1", "_k2"]).reset_index(drop=True)


def ranking_description() -> str:
    metric = "held-out PDE loss" if PHYSICS_METRIC == "pde_loss_eval" else "training physics loss"
    txt = (f"admissible ({metric} < {ADMISSIBLE_PHYSICS:.0e}) first, by obj; "
           f"then non-admissible, by obj")
    if OBJ_TIE_TOL > 0:
        txt += f"; admissible trials within obj+{OBJ_TIE_TOL:g} of the best ordered by {metric}"
    return txt

# Readable LaTeX labels for hyperparameter columns
HPARAM_LABELS = {
    "nn_n_layers" : r"$L$",
    "nn_width"    : r"$W$",
    "activation"  : "Activation",
    "n_elements"  : r"$N_{\mathrm{el}}$",
    "poly_degree" : r"$p$",
}

# Readable LaTeX labels for metric columns (top-trials table sub-header)
METRIC_LABELS = {
    "chi2_nu_train"    : r"$\chi^2_{\nu,\mathrm{train}}$",
    "chi2_nu_test"     : r"$\chi^2_{\nu,\mathrm{test}}$",
    "obj"              : r"$\mathrm{obj}$",
    "final_physics"    : r"$\mathcal{L}_{\mathrm{physics}}$",
    "final_pde"        : r"$\mathcal{L}_{\mathrm{pde}}$",
    "final_dirichlet"  : r"$\mathcal{L}_{\mathrm{dir}}$",
    "final_derivative" : r"$\mathcal{L}_{\mathrm{der}}$",
    "pde_loss_eval"    : r"$\mathcal{L}_{\mathrm{pde}}^{\mathrm{eval}}$",
    "strong_form_rms"  : r"$r_{\mathrm{RMS}}^{\mathrm{strong}}$",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _read_jsonl(path: str) -> pd.DataFrame:
    """Read a JSONL results file into a DataFrame, without any filtering."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def load_results(path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load the search trials of a JSONL results file and drop failed trials.

    Pruned trials are kept: they are completed runs whose physics loss did
    not meet the search-time threshold, and they must appear in the
    diagnostics and in the calibrated/admissible counts. The ``pruned``
    column is retained as information only; admissibility is decided by
    is_admissible() from the physics metric, not from this flag.
    """
    df = _read_jsonl(path)
    for flag in ("failed", "pruned"):
        if flag not in df.columns:
            df[flag] = False
        df[flag] = df[flag].fillna(False).astype(bool)

    n_read   = len(df)
    n_failed = int(df["failed"].sum())
    n_pruned = int((df["pruned"] & ~df["failed"]).sum())
    if verbose:
        print(f"Read {n_read} records from {path}; dropped {n_failed} "
              f"failed; kept {n_pruned} pruned.")

    df = (
        df[~df["failed"]]
        .drop(columns=["failed"])
        .reset_index(drop=True)
    )
    # The analysis uses the search trials only. Records of any earlier
    # replication stage (stage == "replication") are ignored.
    if "stage" in df.columns:
        n_rep = int((df["stage"] != "search").sum())
        if n_rep:
            df = df[df["stage"] == "search"].reset_index(drop=True)
            if verbose:
                print(f"Ignored {n_rep} non-search (replication) records.")
    return df


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

def _panel_label(ax, label: str) -> None:
    """Embed a bold panel letter in the bottom-right corner."""
    ax.text(
        0.92, 0.1, label,
        transform=ax.transAxes,
        fontsize=12, fontweight="bold",
        fontfamily="sans-serif",
        va="top", ha="left",
    )


def _draw_chi2_scatter(ax, df: pd.DataFrame,
                       clip_upper: float) -> None:
    """
    chi2_nu_train vs chi2_nu_test scatter for trials with obj < clip_upper,
    colour-coded by obj over [min obj, clip_upper].
    Colorbar is placed as a horizontal inset in the bottom-right quadrant.
    """
    df_filt = df[df["obj"] < clip_upper].copy()
    n_total = len(df)
    n_kept  = len(df_filt)

    if n_kept == 0:
        ax.text(0.5, 0.5, rf"No trials with $\mathrm{{obj}} < {clip_upper:.2g}$",
                ha="center", va="center", transform=ax.transAxes)
        return

    min_obj = df_filt["obj"].min()

    min_test = df_filt["chi2_nu_test"].min()
    max_test = df_filt["chi2_nu_test"].max()
    min_train = df_filt["chi2_nu_train"].min()
    max_train = df_filt["chi2_nu_train"].max()
    lo = min(min_train, min_test)
    hi = max(max_train, max_test)

    sc = ax.scatter(
        df_filt["chi2_nu_train"],
        df_filt["chi2_nu_test"],
        c      = df_filt["obj"],
        cmap   = "viridis_r",
        s      = 10,
        alpha  = 0.8,
        vmin   = min_obj,
        vmax   = clip_upper,
        zorder = 3,
    )

    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8,
            label="train = test", zorder=2)
    ax.axvline(1.0, color="r", linestyle=":", linewidth=0.8, zorder=2)
    ax.axhline(1.0, color="r", linestyle=":", linewidth=0.8,
               label=r"$\chi^2_\nu = 1$", zorder=2)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\chi^2_{\nu,\mathrm{train}}$")
    ax.set_ylabel(r"$\chi^2_{\nu,\mathrm{test}}$")
    #ax.set_box_aspect(1)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    
    # --- Inset colorbar: horizontal, bottom-right, title above ---
    cax = ax.inset_axes([0.55, 0.18, 0.42, 0.06])
    cbar = plt.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=7)
    ticks = [min_obj, (min_obj + clip_upper) / 2, clip_upper]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([rf"${t:.2g}$" for t in ticks])
    cax.set_title(rf"$\mathrm{{obj}} < {clip_upper:.2g}$  " rf"({n_kept}/{n_total} trials)", fontsize=8, pad=2.5)


def _draw_marginals(ax, df: pd.DataFrame, col: str,
                    clip_upper: float = 2.0) -> None:
    """
    Box plot of obj for each value of one hyperparameter, clipped at
    clip_upper (default 2.0) to focus on the meaningful region. The number
    of trials above clip_upper is annotated on each box.
    """
    rng = np.random.default_rng(0)

    unique_vals = sorted(df[col].unique())
    groups      = [df[df[col] == v]["obj"].values for v in unique_vals]
    labels      = [str(v) for v in unique_vals]

    clipped = [g[g <= clip_upper] for g in groups]

    ax.boxplot(
        clipped, tick_labels=labels, patch_artist=True,
        boxprops     = dict(facecolor="lightsteelblue", alpha=0.5),
        medianprops  = dict(color="navy", linewidth=1.5),
        whiskerprops = dict(linestyle="--"),
        showfliers   = False,
    )

    for i, (g, g_clip) in enumerate(zip(groups, clipped)):
        x_center = i + 1
        if len(g_clip) > 0:
            jitter = rng.uniform(-0.15, 0.15, size=len(g_clip))
            ax.scatter(x_center + jitter, g_clip,
                       s=8, alpha=0.5, color="steelblue", zorder=3)
            ax.scatter(x_center, g_clip.min(),
                       marker="*", s=80, color="darkorange", zorder=5,
                       label="min" if i == 0 else "")

        n_above = int((g > clip_upper).sum())
        if n_above > 0:
            ax.text(x_center, clip_upper * 0.97,
                    f"+{n_above}", ha="center", va="top",
                    fontsize=7, color="firebrick")

    #ax.axhline(1.0, color="r", linestyle=":", linewidth=1.0,
    #           label=r"$\chi^2_\nu = 1$")
    ax.set_ylim(0, clip_upper)
    ax.set_xlabel(HPARAM_LABELS.get(col, col))
    ax.set_ylabel(r"$\mathrm{obj}$")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30)


def _draw_recovery(ax, df: pd.DataFrame, param_name: str,
                   true_val: float,
                   clip_upper: float = 10.0) -> None:
    """
    Scatter of recovered parameter value vs chi2_nu_test for trials with
    obj < clip_upper, colour-coded by obj.
    """
    df_filt = df[df["obj"] < clip_upper].copy()
    min_obj = df_filt["obj"].min()
    max_obj = df_filt["obj"].max()
    n_total, n_kept = len(df), len(df_filt)

    if n_kept == 0 or param_name not in df_filt.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return

    cal_filt = df_filt["obj"]

    sc = ax.scatter(
        df_filt["chi2_nu_test"],
        df_filt[param_name].values,
        c=cal_filt, cmap="plasma_r",
        s=20, alpha=0.8,
        vmin=df_filt["obj"].min(), vmax=clip_upper,
    )
    ax.axhline(true_val[1], color="r", linestyle="--", linewidth=1.2,
               label=rf"true = {true_val[1]:.4g}")

    plt.colorbar(sc, ax=ax,
                 label=r"$\mathrm{obj}$", pad=0.02)

    ax.set_xlabel(r"$\chi^2_{\nu,\mathrm{test}}$")
    ax.set_ylabel(true_val[0])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        rf"{true_val[0]}  ({n_kept}/{n_total} trials)", fontsize=9,
    )


# ---------------------------------------------------------------------------
# Figure A: chi2 scatter + marginals  (all cases)
# ---------------------------------------------------------------------------

def plot_figure_A(df: pd.DataFrame, tag: str, figures_dir: str,
                  clip_upper: float = 2.0) -> None:
    """
    Compound figure with two rows:
      Row 1 — chi2_nu_train vs chi2_nu_test scatter  (single panel, left)
              + first two hyperparameter marginals    (right)
      Row 2 — remaining hyperparameter marginals

    Panel letters: a) scatter, b) L, c) W, d) Activation, e) N_el, f) p

    Saved as <figures_dir>/diagnostics_<tag>.eps.
    """
    cols = [c for c in HPARAM_COLS if c in df.columns]
    n_hp = len(cols)

    # Layout: 2 rows. Row 1: scatter + first 2 marginals. Row 2: rest.
    n_row1 = min(n_hp, 2)
    n_row2 = max(n_hp - n_row1, 0)
    total_cols_row1 = 1 + n_row1   # scatter + first marginals

    fig = plt.figure(figsize=(4 * total_cols_row1, 8))
    gs_top = gridspec.GridSpec(
        1, total_cols_row1, figure=fig,
        left=0.06, right=0.97, top=0.95, bottom=0.54,
        wspace=0.38,
    )
    gs_bot = gridspec.GridSpec(
        1, max(n_row2, 1), figure=fig,
        left=0.06, right=0.97, top=0.47, bottom=0.07,
        wspace=0.38,
    )

    letters = "abcdefghij"
    letter_idx = 0

    # --- (a) chi2 scatter ---
    ax_scatter = fig.add_subplot(gs_top[0, 0])
    _draw_chi2_scatter(ax_scatter, df, clip_upper)
    _panel_label(ax_scatter, f"{letters[letter_idx]})")
    letter_idx += 1

    # --- (b), (c) first two marginals in top row ---
    for i in range(n_row1):
        ax = fig.add_subplot(gs_top[0, 1 + i])
        _draw_marginals(ax, df, cols[i], clip_upper)
        if i == 0:
            ax.legend(fontsize=7, loc="lower left")
        _panel_label(ax, f"{letters[letter_idx]})")
        letter_idx += 1

    # --- remaining marginals in bottom row ---
    for i in range(n_row2):
        ax = fig.add_subplot(gs_bot[0, i])
        _draw_marginals(ax, df, cols[n_row1 + i], clip_upper)
        _panel_label(ax, f"{letters[letter_idx]})")
        letter_idx += 1

    path = os.path.join(figures_dir, f"diagnostics_{tag}.eps")
    fig.savefig(path, format="eps", bbox_inches="tight")
    print(f"[saved] {path}")
    plt.close(fig)


def plot_architecture_occurrences(df: pd.DataFrame, tag: str,
                                  figures_dir: str) -> None:
    """Plot repeated architectures among trials meeting fixed metric limits."""
    required = HPARAM_COLS + ["obj", "pde_loss_eval"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        print("[skip] architecture-occurrence plot; missing columns: "
              + ", ".join(missing))
        return

    if "stage" in df.columns:
        df = df.loc[df["stage"] == "search"]

    qualified = df.loc[
        (pd.to_numeric(df["obj"], errors="coerce") < 4)
        & (pd.to_numeric(df["pde_loss_eval"], errors="coerce") < ADMISSIBLE_PHYSICS)
    ]
    summary = (
        qualified.groupby(HPARAM_COLS, dropna=False)
        .agg(
            count=("obj", "size"),
            mean_obj=("obj", "mean"),
            std_obj=("obj", "std"),
        )
    )
    summary = summary[summary["count"] > 1]
    # completed runs per combination (denominator of the fraction)
    n_runs = combination_summary(df)["n_runs"]
    summary["n_runs"] = n_runs.reindex(summary.index).to_numpy()
    summary["frac"] = summary["count"] / summary["n_runs"]
    summary = summary.sort_values(["count", "frac"])
    # winning combination (same rule as the winners summary)
    _, win_combo, _ = _select_winning_architecture(df)
    if summary.empty:
        print(f"[skip] no repeated architectures with obj < 4 and "
              f"pde_loss_eval < {ADMISSIBLE_PHYSICS}")
        return

    labels = [
        rf"$L={layers}$, $W={width}$, {activation}, $N_{{\mathrm{{el}}}}={elements}$, $p={degree}$"
        for layers, width, activation, elements, degree
        in summary.index
    ]
    values = summary["count"].to_numpy()
    fig_height = max(3.0, 0.38 * len(labels) + 1.0)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    colors = ["darkorange" if tuple(idx) == tuple(win_combo) else "steelblue"
              for idx in summary.index]
    bars = ax.barh(labels, values, color=colors)
    ax.bar_label(bars, labels=[f"{int(c)}/{int(n)}" for c, n
                               in zip(summary["count"], summary["n_runs"])],
                 padding=3)
    for bar, (_, stats) in zip(bars, summary.iterrows()):
        ax.annotate(
            f"mean={stats['mean_obj']:.3g}, std={stats['std_obj']:.3g}",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(45, 0), textcoords="offset points",
            va="center", fontsize=8,
        )
    ax.set_xlabel("Number of qualified trials (label: qualified / completed runs)")
    ax.set_ylabel("Architecture")
    ax.set_title(r"Repeated search architectures ($\mathrm{obj} < 4$, "
                 rf"$\mathcal{{L}}_{{\mathrm{{pde}}}}^{{\mathrm{{eval}}}} < {ADMISSIBLE_PHYSICS:.0e}$)")
    ax.set_xticks(np.arange(0, int(values.max()) + 1))
    ax.set_xlim(0, int(values.max()) + max(8, int(values.max() * 0.6)))
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    path = os.path.join(figures_dir, f"architecture_occurrences_{tag}.eps")
    fig.savefig(path, format="eps", bbox_inches="tight")
    print(f"[saved] {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure B: parameter recovery  (Case 1 only)
# ---------------------------------------------------------------------------

def plot_figure_B(df: pd.DataFrame, true_params: dict,
                  tag: str, figures_dir: str,
                  clip_upper: float = 10.0) -> None:
    """
    One panel per recovered parameter, arranged in a single row.
    Panel letters start from a). Saved as <figures_dir>/recovery_<tag>.eps.
    """
    params_present = [n for n in true_params if n in df.columns]
    if not params_present:
        print("No parameter columns found — skipping Figure B.")
        return

    n = len(params_present)
    fig, axes = plt.subplots(
        1, n, figsize=(5 * n, 4.5),
        squeeze=False,
    )
    axes = axes[0]

    letters = "abcdefghij"
    for i, (ax, param) in enumerate(zip(axes, params_present)):
        _draw_recovery(ax, df, param, true_params[param], clip_upper)
        _panel_label(ax, f"{letters[i]})")

    plt.tight_layout()
    path = os.path.join(figures_dir, f"recovery_{tag}.eps")
    fig.savefig(path, format="eps", bbox_inches="tight")
    print(f"[saved] {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_top_trials(df: pd.DataFrame, true_params: dict,
                     case: str, n: int = 20,
                     results_dir: str = None) -> None:
    all_n = len(df)
    param_cols = [p for p in true_params if p in df.columns]
    err_cols   = []
    if has_ground_truth(case):
        err_cols = [f"err_{p}" for p in true_params if f"err_{p}" in df.columns]

    candidate_cols = (
        ["trial", "nn_n_layers", "nn_width", "activation",
         "n_elements", "poly_degree", "init_seed",
         "chi2_nu_train", "chi2_nu_test", "obj",
         "final_physics", "final_pde", "final_dirichlet", "final_derivative",
         "pde_loss_eval", "strong_form_rms",
         "elapsed_s"]
        + param_cols + err_cols
    )
    select_cols = [c for c in candidate_cols if c in df.columns]

    ranked = rank_trials(df)
    select_cols = select_cols + ["admissible", "calibrated"]
    top = ranked.head(n)[select_cols]

    print(f"\nTop {n} trials ({ranking_description()}):")
    print(top.to_string(index=False, float_format="{:.4g}".format))

    if results_dir is not None:
        path = os.path.join(results_dir, "top_trials.csv")
        export_cols = [c for c in ranked.columns if c not in ("noise_level",)]
        ranked.head(all_n)[export_cols].to_csv(path, index=False)
        print(f"[saved] {path}")


def print_top_trials_latex(
    df: pd.DataFrame,
    true_params: dict,
    case: str,
    n: int = 20,
    tag: str = None,
    tables_dir: str = "./tables",
) -> None:
    """
    Print (or save) a LaTeX table of the top-n trials ranked by
    rank_trials(): admissible trials first, by obj; then non-admissible
    trials, by obj. Non-admissible trials are marked with ^dagger on the
    trial number. Only search trials are considered.

    Recovery error columns are included only for Case 1 (Grenoble sand),
    where the true parameters are known.  For Cases 2 and 3, the table
    omits recovered parameter values since no ground truth is available.

    Header layout (below the group-level header and its cmidrules) uses two
    stacked rows:
      * row 1 -- column name ("Trial", "Time", metric symbol, ...);
      * row 2 -- column qualifier ("No.", "[s]", or the common scale factor
                 $\times 10^{k}$ of a scaled metric column).
    For the scaled metric columns (training losses, evaluation loss and
    strong-form residual), k is the lowest decimal exponent among the
    non-zero entries of that column in the displayed top-n trials, and each
    cell shows only the leading number v / 10^k with two decimals.

    Parameters
    ----------
    df          : DataFrame of non-failed trials.
    true_params : Case configuration dict (keys = parameter names).
    case        : Case identifier ('grenoblesand', 'vangenuchten', 'letd').
    n           : Number of top trials to include (default: 20).
    tag         : If given, write the LaTeX table to
                  <tables_dir>/latex_top20_<tag>.tex.
                  If None, print to stdout.
    tables_dir : Output directory for the .tex file (default: ./tables).
    """
    import io

    # --- search trials only ---
    if "stage" in df.columns:
        df = df[df["stage"] == "search"].copy()
    else:
        df = df.copy()

    # --- Recovery error columns (Case 1 only) ---
    if has_ground_truth(case):
        err_cols = [
            f"err_{p}" for p in true_params
            if f"err_{p}" in df.columns
        ]
    else:
        err_cols = []

    # --- Ranking (admissible first; see rank_trials) ---
    top = rank_trials(df).head(n).copy()

    # --- Convert errors to percentages ---
    for col in err_cols:
        top[col] = top[col] * 100

    # --- Rank number column ---
    top.insert(0, "rank_no", range(1, len(top) + 1))

    # --- Combined architecture string  (W x L) ---
    top["W_L"] = (
        top["nn_width"].astype(int).astype(str)
        + r"$\times$"
        + top["nn_n_layers"].astype(int).astype(str)
    )

    # --- Determine which HP columns are present ---
    has_elements = "n_elements" in top.columns
    has_degree   = "poly_degree" in top.columns

    hp_cols  = ["W_L", "activation"]
    hp_heads = [r"W$\times$L", "Act."]
    if has_elements:
        hp_cols.append("n_elements")
        hp_heads.append(r"$N_{\mathrm{el}}$")
    if has_degree:
        hp_cols.append("poly_degree")
        hp_heads.append(r"$p$")

    n_hp = len(hp_cols)

    metric_cols  = [
        c for c in [
            "chi2_nu_train", "chi2_nu_test", "obj", "final_physics",
            "final_pde", "pde_loss_eval", "strong_form_rms",
        ]
        if c in df.columns
    ]
    n_metrics    = len(metric_cols)

    final_cols = (
        ["rank_no", "trial", "elapsed_s"]
        + hp_cols
        + metric_cols
        + err_cols
    )

    n_err = len(err_cols)

    # Column totals for cmidrule calculation
    # Col 1: No.  Col 2: Trial  Col 3: Time
    # Cols 4 .. 3+n_hp: HPs
    # Cols 4+n_hp .. 3+n_hp+n_metrics: Metrics
    # Cols 4+n_hp+n_metrics .. end: Errors (Case 1 only)
    hp_start  = 4
    hp_end    = 3 + n_hp
    met_start = hp_end + 1
    met_end   = hp_end + n_metrics
    err_start = met_end + 1
    err_end   = met_end + n_err

    # --- Metric header labels ---
    # Final (training-quadrature) loss terms carry the "train" superscript,
    # to distinguish them from the independently evaluated pde_loss_eval.
    _TRAIN_LOSS_LABELS = {
        "final_physics"    : r"$\mathcal{L}_{\mathrm{physics}}^{\mathrm{train}}$",
        "final_pde"        : r"$\mathcal{L}_{\mathrm{pde}}^{\mathrm{train}}$",
        "final_dirichlet"  : r"$\mathcal{L}_{\mathrm{dir}}^{\mathrm{train}}$",
        "final_derivative" : r"$\mathcal{L}_{\mathrm{der}}^{\mathrm{train}}$",
    }

    def _metric_label(col: str) -> str:
        return _TRAIN_LOSS_LABELS.get(col, METRIC_LABELS[col])

    # --- Per-column scale factors for the scaled metric columns ---
    _SCALED_COLS = {
        "final_physics", "final_pde", "final_dirichlet", "final_derivative",
        "pde_loss_eval", "strong_form_rms",
    }
    _SCALED_DECIMALS = 2

    def _decimal_exponent(v: float) -> int:
        # Exponent as it would be displayed with _SCALED_DECIMALS decimals
        # in the mantissa, so that e.g. 9.996e-5 is treated as 1.00e-4.
        return int(f"{v:.{_SCALED_DECIMALS}e}".split("e")[1])

    col_exponent = {}
    for col in metric_cols:
        if col not in _SCALED_COLS:
            continue
        vals = pd.to_numeric(top[col], errors="coerce").to_numpy(dtype=float)
        nz = np.abs(vals[np.isfinite(vals) & (vals != 0.0)])
        col_exponent[col] = _decimal_exponent(nz.min()) if nz.size else 0

    def _scale_head(col: str) -> str:
        k = col_exponent.get(col)
        if k is None or k == 0:
            return ""
        return rf"$\times 10^{{{k}}}$"

    # --- Build LaTeX lines ---
    out = io.StringIO()

    def w(line=""):
        out.write(line + "\n")

    # Column alignment
    align = "lcc " + "c" * n_hp + " " + "c" * n_metrics
    if n_err:
        align += " " + "c" * n_err

    w(r"\begin{tabular}{" + align + "}")
    w(r"\toprule")

    # Top-level (group) headers
    hp_span  = rf"\multicolumn{{{n_hp}}}{{c}}{{Optimal HPs}}"
    met_span = rf"\multicolumn{{{n_metrics}}}{{c}}{{Metrics}}"
    row1 = f"& & & {hp_span} & {met_span}"
    if n_err:
        err_span = rf"\multicolumn{{{n_err}}}{{c}}{{Recovery error}}"
        row1 += f" & {err_span}"
    row1 += r" \\"
    w(row1)

    # Cmidrules
    rules = (
        rf"\cmidrule(lr){{{hp_start}-{hp_end}}}"
        rf"\cmidrule(lr){{{met_start}-{met_end}}}"
    )
    if n_err:
        rules += rf"\cmidrule(lr){{{err_start}-{err_end}}}"
    w(rules)

    # Sub-header, row A: column names
    sub_a = [" ", "Trial", "Time"] + hp_heads
    sub_a += [_metric_label(c) for c in metric_cols]
    for col in err_cols:
        param = col.replace("err_", "")
        spec  = true_params.get(param)
        # Use the LaTeX label from the case config when available
        # (e.g. [r"$K_{\mathrm{s}}/\alpha$", value]); fall back to the key.
        if isinstance(spec, (list, tuple)) and isinstance(spec[0], str):
            sym = spec[0].strip("$")
        else:
            sym = param
        sub_a.append(rf"$\varepsilon_{{{sym}}}$")
    w(" & ".join(sub_a).lstrip() + r" \\")

    # Sub-header, row B: qualifiers / scale factors
    sub_b = ["No.", "No.", "[s]"] + [""] * n_hp
    sub_b += [_scale_head(c) for c in metric_cols]
    sub_b += [r"[\%]"] * n_err
    w(" & ".join(sub_b).rstrip() + r" \\")
    w(r"\midrule")

    # Data rows
    for _, row in top.iterrows():
        cells = []
        for col in final_cols:
            val = row[col]
            if pd.isna(val):
                cells.append("---")
            elif col == "elapsed_s":
                cells.append(f"{val:.1f}")
            elif col == "trial":
                mark = "" if bool(row["admissible"]) else r"$^\dagger$"
                cells.append(str(int(val)) + mark)
            elif col == "rank_no":
                cells.append(str(int(val)))
            elif col in ("n_elements", "poly_degree"):
                cells.append(str(int(val)))
            elif col == "activation":
                cells.append(str(val))
            elif col == "W_L":
                cells.append(str(val))
            elif col in col_exponent:
                if val == 0.0:
                    cells.append("0")
                else:
                    scaled = val / 10.0 ** col_exponent[col]
                    cells.append(f"{scaled:.{_SCALED_DECIMALS}f}")
            elif isinstance(val, float):
                if val == 0.0:
                    cells.append(r"\num{0}")
                elif abs(val) < 1e-3:
                    cells.append(rf"\num{{{val:.3e}}}")
                else:
                    cells.append(rf"\num{{{val:.4g}}}")
            else:
                cells.append(str(val))
        w("  " + " & ".join(cells) + r" \\")

    w(r"\bottomrule")
    w(r"\end{tabular}")

    result = out.getvalue()

    if tag is not None:
        output_path = os.path.join(tables_dir, f"latex_top20_{tag}.tex")
        with open(output_path, "w") as f:
            f.write(result)
        print(f"[saved] {output_path}")
    else:
        print(result)


# ---------------------------------------------------------------------------
# Multi-seed identifiability statistics
# ---------------------------------------------------------------------------

def _param_label(true_params: dict, name: str) -> str:
    """LaTeX label of a physical parameter, falling back to its key."""
    spec = true_params.get(name)
    if isinstance(spec, (list, tuple)) and isinstance(spec[0], str):
        return spec[0]
    return name


def _param_value(true_params: dict, name: str):
    """Reference value of a physical parameter, or None if not available."""
    spec = true_params.get(name)
    if isinstance(spec, (list, tuple)):
        return spec[1] if len(spec) > 1 else None
    return spec


def winner_is_valid(winner) -> bool:
    """True if the selected combination is reproducibly admissible
    (selection basis 'qualified' or 'admissible')."""
    return bool(winner["admissible"]) and bool(winner.get("reproducible", True))


def combination_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per hyperparameter combination among the search trials of df
    (completed runs; failed runs are dropped by load_results).

    n_runs        completed runs of the combination
    n_admissible  runs with physics metric < ADMISSIBLE_PHYSICS
    n_qualified   runs that are admissible AND calibrated (obj < CALIBRATED_OBJ)
    frac_qualified, frac_admissible  = n_qualified / n_runs, n_admissible / n_runs
    median_obj_qualified, median_obj_admissible  median obj over those runs
    """
    cols = [c for c in HPARAM_COLS if c in df.columns]
    search = df[df["stage"] == "search"] if "stage" in df.columns else df
    r = rank_trials(search)
    r["_obj"] = pd.to_numeric(r["obj"], errors="coerce")
    r["_qual"] = r["admissible"] & r["calibrated"]
    g = r.groupby(cols, dropna=False)
    out = pd.DataFrame({
        "n_runs":       g.size(),
        "n_admissible": g["admissible"].sum().astype(int),
        "n_qualified":  g["_qual"].sum().astype(int),
        "median_obj_qualified":  g.apply(lambda x: x.loc[x["_qual"], "_obj"].median()),
        "median_obj_admissible": g.apply(lambda x: x.loc[x["admissible"], "_obj"].median()),
    })
    out["frac_qualified"] = out["n_qualified"] / out["n_runs"]
    out["frac_admissible"] = out["n_admissible"] / out["n_runs"]
    return out


def _representative_trial(runs: pd.DataFrame) -> pd.Series:
    """The run whose obj is closest to the median obj of `runs`."""
    obj = pd.to_numeric(runs["obj"], errors="coerce")
    return runs.loc[(obj - obj.median()).abs().idxmin()]


def _select_winning_architecture(df: pd.DataFrame):
    """
    Winning hyperparameter COMBINATION of a study (search trials only), selected
    by reproducibility rather than by a single trial:

      1. qualified runs = admissible and calibrated. Among combinations
         with >= MIN_RUNS qualified runs: most qualified runs;
         ties -> higher qualified fraction (n_qualified / n_runs); then
         lower median obj of the qualified runs.         basis "qualified"
      2. otherwise, the same on admissible runs only (calibration
         dropped; the winner is flagged not calibrated). basis "admissible"
      3. otherwise, no reproducible physics-consistent combination: the
         rank-1 trial under rank_trials() (not admissible).  basis "none"

    Returns (winner, arch_tuple, arch_cols). `winner` is the representative
    run of the combination (qualified [or admissible] run with the median
    obj), with the extra fields selection_basis, n_runs, n_admissible,
    n_qualified, frac_qualified and median_obj.
    """
    cols = [c for c in HPARAM_COLS if c in df.columns]
    search = df[df["stage"] == "search"] if "stage" in df.columns else df
    ranked = rank_trials(search)
    summ = combination_summary(search)

    winner, basis, combo = None, "none", None
    for basis_try, n_col, f_col, m_col, flag in (
        ("qualified",  "n_qualified",  "frac_qualified",  "median_obj_qualified",  "q"),
        ("admissible", "n_admissible", "frac_admissible", "median_obj_admissible", "a"),
    ):
        cand = summ[summ[n_col] >= MIN_RUNS]
        if cand.empty:
            continue
        cand = cand.sort_values([n_col, f_col, m_col],
                                ascending=[False, False, True], kind="mergesort")
        combo = cand.index[0] if isinstance(cand.index[0], tuple) else (cand.index[0],)
        mask = pd.Series(True, index=ranked.index)
        for c, v in zip(cols, combo):
            mask &= ranked[c] == v
        runs = ranked[mask]
        runs = runs[runs["admissible"] & runs["calibrated"]] if flag == "q" else runs[runs["admissible"]]
        winner = _representative_trial(runs).copy()
        basis = basis_try
        break

    if winner is None:
        winner = ranked.iloc[0].copy()
        combo = tuple(winner[c] for c in cols)

    stats = summ.loc[combo if len(cols) > 1 else combo[0]]
    winner["selection_basis"] = basis
    # A combination without >= MIN_RUNS admissible runs is not a
    # reproducible winner, even if its representative trial happens to be
    # admissible: it is not replicated, analysed or selected.
    winner["reproducible"] = basis != "none"
    for k in ("n_runs", "n_admissible", "n_qualified"):
        winner[k] = int(stats[k])
    winner["frac_qualified"] = float(stats["frac_qualified"])
    winner["median_obj"] = float(stats["median_obj_qualified"] if basis == "qualified"
                                 else stats["median_obj_admissible"])
    return winner, tuple(winner[c] for c in cols), cols


def compute_identifiability_stats(
    df: pd.DataFrame,
    true_params: dict,
    case: str,
    results_dir: str = None,
):
    """
    Statistics over the repeated runs of the winning combination, and the
    correlation structure between the recovered physical parameters.

    The winning combination is selected by _select_winning_architecture
    (search trials only). The statistics are computed over ALL its
    qualified runs (admissible and calibrated) among the search trials, so
    N differs between studies. If the combination was selected on the
    "admissible" basis (no combination has enough qualified runs), the
    statistics are over its admissible runs instead. If no combination has
    >= MIN_RUNS runs on either basis, nothing is computed.

    The N runs share the fixed physical-parameter initial guess and differ
    in the NN-weight seed and in the TPE-driven history of the search; the
    combination is chosen from the same trials, so the statistics are
    slightly optimistic (selection effect). The fresh-seed final runs are
    the unbiased check. This function measures reproducibility under
    repeated independent training. It is not on its own evidence of
    identifiability, which requires varying the physical-parameter
    starting guess (multi-start campaign).

    Parameters
    ----------
    df          : DataFrame of non-failed search trials (load_results).
    true_params : Case configuration dict (keys = parameter names).
    case        : Case identifier.
    results_dir : If given, save the per-trial subset, the summary
                  statistics and the correlation matrices as CSVs there.

    Returns
    -------
    result : dict with keys 'winner', 'arch_cols', 'arch_tuple', 'sub',
             'param_cols', 'summary', 'corr_pearson', 'corr_spearman',
             'basis', 'n_runs'; or None if the analysis is not applicable.
    """
    if df.empty:
        print(f"\n[{case}] No search trials -- skipping identifiability.")
        return None

    winner, arch_tuple, arch_cols = _select_winning_architecture(df)
    arch_str = ", ".join(f"{c}={v}" for c, v in zip(arch_cols, arch_tuple))
    print(f"\n[{case}] Winning combination ({winner['selection_basis']}: "
          f"{winner['n_qualified']} qualified / {winner['n_admissible']} admissible / "
          f"{winner['n_runs']} runs; representative trial "
          f"{int(winner['trial'])}): {arch_str}")

    if not winner_is_valid(winner):
        print(f"  No combination has >= {MIN_RUNS} admissible runs "
              f"({PHYSICS_METRIC} < {ADMISSIBLE_PHYSICS:.0e}). Skipping statistics.")
        return None

    mask = pd.Series(True, index=df.index)
    for col, val in zip(arch_cols, arch_tuple):
        mask &= (df[col] == val)
    matching = df[mask]

    basis = str(winner["selection_basis"])
    keep = is_admissible(matching)
    if basis == "qualified":
        keep &= is_calibrated(matching)
    sub = matching[keep].copy()
    n_sub = len(sub)
    n_runs = len(matching)
    print(f"  {n_runs} runs of the combination; statistics over the {n_sub} "
          f"{basis} runs ({PHYSICS_METRIC} < {ADMISSIBLE_PHYSICS:.0e}"
          f"{f', obj < {CALIBRATED_OBJ:g}' if basis == 'qualified' else ''}).")

    if "init_seed" in sub.columns:
        dup = sub["init_seed"].duplicated(keep=False)
        if dup.any():
            seeds = sorted(sub.loc[dup, "init_seed"].unique().tolist())
            print(f"  WARNING: repeated init_seed values in the subset: {seeds}")

    if n_sub < 2:
        print("  Fewer than 2 runs -- statistics not computed.")
        return None

    # --- Marginal statistics ---
    param_cols = [p for p in true_params if p in sub.columns]
    err_cols   = [f"err_{p}" for p in true_params if f"err_{p}" in sub.columns]
    for col in err_cols:
        sub[col] = sub[col] * 100            # relative errors in %
    stat_cols  = ["obj", "chi2_nu_train", "chi2_nu_test"] + param_cols + err_cols
    stat_cols  = [c for c in stat_cols if c in sub.columns]

    summary = pd.DataFrame({
        "mean": sub[stat_cols].mean(),
        "std":  sub[stat_cols].std(),
        "cv_%": 100 * sub[stat_cols].std() / sub[stat_cols].mean().abs(),
        "min":  sub[stat_cols].min(),
        "max":  sub[stat_cols].max(),
    })
    print(f"\n  Marginal statistics over {n_sub} {basis} runs "
          f"(err_* in %):")
    print(summary.to_string(float_format="{:.4g}".format))

    # --- Joint correlation structure ---
    corr_pearson  = None
    corr_spearman = None
    if len(param_cols) >= 2:
        corr_pearson  = sub[param_cols].corr(method="pearson")
        corr_spearman = sub[param_cols].corr(method="spearman")
        print(f"\n  Parameter correlation (Pearson):")
        print(corr_pearson.to_string(float_format="{:.4g}".format))
        print(f"\n  Parameter correlation (Spearman):")
        print(corr_spearman.to_string(float_format="{:.4g}".format))

    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        sub_path = os.path.join(results_dir, "identifiability_trials.csv")
        sub.to_csv(sub_path, index=False)
        print(f"[saved] {sub_path}")

        stats_path = os.path.join(results_dir, "identifiability_stats.csv")
        summary.to_csv(stats_path)
        print(f"[saved] {stats_path}")

        if corr_pearson is not None:
            corr_path = os.path.join(results_dir, "identifiability_correlation.csv")
            with open(corr_path, "w") as f:
                f.write("# Pearson\n")
                corr_pearson.to_csv(f)
                f.write("\n# Spearman\n")
                corr_spearman.to_csv(f)
            print(f"[saved] {corr_path}")

    return dict(
        winner        = winner,
        arch_cols     = arch_cols,
        arch_tuple    = arch_tuple,
        sub           = sub,
        param_cols    = param_cols,
        summary       = summary,
        corr_pearson  = corr_pearson,
        corr_spearman = corr_spearman,
        basis         = basis,
        n_runs        = n_runs,
    )


def print_identifiability_latex(
    result: dict,
    true_params: dict,
    case: str,
    tag: str = None,
    tables_dir: str = "./tables",
) -> None:
    """
    Print (or save) a LaTeX table summarising the multi-seed statistics of
    the recovered physical parameters for the winning architecture.

    Layout (parameters in rows):
      * a header row with the winning configuration and N (runs used out of
        the runs of the combination);
      * Parameter | [True] | Mean | Std | CV [%] | Min | Max | [Bias [%]]
        (True and Bias -- signed relative bias of the ensemble mean -- only
        for Case 1, where the true parameters are known);
      * a correlation block: Pearson below the diagonal, Spearman above.

    Parameters
    ----------
    result      : dict returned by compute_identifiability_stats.
    true_params : Case configuration dict.
    case        : Case identifier.
    tag         : If given, write the table to
                  <tables_dir>/latex_identifiability_<tag>.tex.
                  If None, print to stdout.
    tables_dir : Output directory for the .tex file (default: ./tables).
    """
    import io

    has_truth  = has_ground_truth(case)
    winner     = result["winner"]
    sub        = result["sub"]
    params     = result["param_cols"]
    summary    = result["summary"]
    basis      = result["basis"]
    n_runs     = result["n_runs"]
    corr_p     = result["corr_pearson"]
    corr_s     = result["corr_spearman"]
    n_total    = len(sub)

    def num(v, fmt=".4g"):
        if pd.isna(v):
            return "---"
        return rf"\num{{{v:{fmt}}}}"

    # --- Columns ---
    heads_a = ["Parameter"]
    heads_b = [""]
    if has_truth:
        heads_a.append("True");  heads_b.append("")
    heads_a += ["Mean", "Std", "CV", "Min", "Max"]
    heads_b += ["", "", r"[\%]", "", ""]
    if has_truth:
        heads_a.append("Bias");  heads_b.append(r"[\%]")
    n_cols = len(heads_a)

    # --- Configuration header ---
    arch = []
    if "nn_width" in winner.index and "nn_n_layers" in winner.index:
        arch.append(f"{int(winner['nn_width'])}$\\times${int(winner['nn_n_layers'])}")
    if "activation" in winner.index:
        arch.append(str(winner["activation"]))
    if "n_elements" in winner.index:
        arch.append(rf"$N_{{\mathrm{{el}}}}={int(winner['n_elements'])}$")
    if "poly_degree" in winner.index:
        arch.append(rf"$p={int(winner['poly_degree'])}$")
    n_str = (
        rf"$N={n_total}$ {basis} runs of {n_runs}"
        + ("" if basis == "qualified" else r"$^\ddagger$")
    )
    config_line = ", ".join(arch) + r" \quad " + n_str

    out = io.StringIO()

    def w(line=""):
        out.write(line + "\n")

    w(r"\begin{tabular}{l" + "c" * (n_cols - 1) + "}")
    w(r"\toprule")
    w(rf"\multicolumn{{{n_cols}}}{{l}}{{{config_line}}} \\")
    w(r"\midrule")
    w(" & ".join(heads_a) + r" \\")
    w(" & ".join(heads_b).rstrip() + r" \\")
    w(r"\midrule")

    # --- Parameter rows ---
    for p in params:
        st = summary.loc[p]
        cells = [_param_label(true_params, p)]
        if has_truth:
            true_val = _param_value(true_params, p)
            cells.append(num(true_val))
        cells += [
            num(st["mean"]), num(st["std"]), num(st["cv_%"], ".3g"),
            num(st["min"]),  num(st["max"]),
        ]
        if has_truth:
            true_val = _param_value(true_params, p)
            bias = (
                100 * (st["mean"] - true_val) / abs(true_val)
                if true_val not in (None, 0) else float("nan")
            )
            cells.append(
                "---" if pd.isna(bias)
                else rf"\num[retain-explicit-plus]{{{bias:+.3g}}}"
            )
        w("  " + " & ".join(cells) + r" \\")

    # --- Correlation block: Pearson (lower), Spearman (upper) ---
    if corr_p is not None:
        pad = [""] * (n_cols - 1 - len(params))
        w(r"\midrule")
        w(rf"\multicolumn{{{n_cols}}}{{l}}"
          r"{Correlation: Pearson (lower), Spearman (upper)} \\")
        w(" & ".join([""] + [_param_label(true_params, p) for p in params]
                     + pad) + r" \\")
        for i, pi in enumerate(params):
            cells = [_param_label(true_params, pi)]
            for j, pj in enumerate(params):
                if i == j:
                    cells.append("1")
                elif i > j:
                    cells.append(num(corr_p.loc[pi, pj], ".2f"))
                else:
                    cells.append(num(corr_s.loc[pi, pj], ".2f"))
            w("  " + " & ".join(cells + pad) + r" \\")

    w(r"\bottomrule")
    w(r"\end{tabular}")

    result_tex = out.getvalue()

    if tag is not None:
        output_path = os.path.join(tables_dir, f"latex_identifiability_{tag}.tex")
        with open(output_path, "w") as f:
            f.write(result_tex)
        print(f"[saved] {output_path}")
    else:
        print(result_tex)


# ---------------------------------------------------------------------------
# Trial counts across all cases
# ---------------------------------------------------------------------------

def resolve_run(model: str, case: str, es_or_gb: str = None,
                noise_level: float = None, results_dir: str = None):
    """
    Return (results_dir, results_path, tag) for one post-processing run.

    The run directory is resolved under <repo>/<model>/. results_dir
    overrides the case's default base directory name (a relative value is
    taken relative to <repo>/<model>/, an absolute one is used as is); the
    _es/_gb and _<n>pct_noise suffixes are appended as usual.

    The tag (used in output file names) does not include the model, so the
    output names are unchanged from the per-model version of this script;
    models are kept apart by their separate Figures directories.
    """
    cfg  = CASES[case]
    base = str(model_dir(model) / (results_dir or cfg["results_dir"]))
    tag  = case
    if es_or_gb is not None:
        base = f"{base}_{es_or_gb}"
        tag  = f"{tag}_{es_or_gb}"

    if "noise_levels" in cfg:
        pct       = int(round(float(noise_level) * 100))
        run_dir   = f"{base}_{pct}pct_noise"
        run_file  = f"hparam_results_{pct}pct_noise.jsonl"
        tag       = f"{tag}_{pct}pct_noise"
    else:
        run_dir   = base
        run_file  = cfg["results_file"]

    return run_dir, os.path.join(run_dir, run_file), tag


def count_trials(results_path: str) -> dict:
    """
    Trial counts for one results file, as reported in the trial-count table
    (search trials only).

    completed             : non-failed trials (pruned trials included)
    calibrated            : completed trials with obj < CALIBRATED_OBJ
    admissible            : completed trials with physics metric < ADMISSIBLE_PHYSICS
    admissible_calibrated : admissible trials that are also calibrated
    winner_ok             : whether a combination with >= MIN_RUNS
                            qualified (or admissible) runs exists
    basis, n_qualified, n_runs : selection basis and run counts of the
                            winning combination
    """
    search = load_results(results_path, verbose=False)
    is_cal = is_calibrated(search)
    is_adm = is_admissible(search)

    winner_ok, basis, n_q, n_r, n_a = False, "none", 0, 0, 0
    if len(search):
        winner, _, _ = _select_winning_architecture(search)
        winner_ok = winner_is_valid(winner)
        basis = str(winner["selection_basis"])
        n_q, n_r = int(winner["n_qualified"]), int(winner["n_runs"])
        n_a = int(winner["n_admissible"])

    return dict(
        completed             = int(len(search)),
        admissible            = int(is_adm.sum()),
        admissible_calibrated = int((is_adm & is_cal).sum()),
        calibrated            = int(is_cal.sum()),
        winner_ok             = winner_ok,
        basis                 = basis,
        n_qualified           = n_q,
        n_admissible_win      = n_a,
        n_runs                = n_r,
    )


def print_trial_counts_latex(
    model: str,
    tables_dir: str = None,
    configs: tuple = ("es", "gb"),
    save: bool = True,
) -> None:
    """
    Build the trial-count table for all cases,
    configurations and noise levels of one model, and write it to
    <tables_dir>/latex_trial_counts.tex (or print it if save is False).
    tables_dir defaults to <results>/<model>/tables.

    Only the tabular is generated; the caption and label stay in the SI.
    Cases are ordered by their label4latex. A missing results file gives a
    row of '---'. Columns: Completed (non-failed trials), Calibrated
    (obj < CALIBRATED_OBJ), Admissible (physics metric <
    ADMISSIBLE_PHYSICS), Adm. & calibrated (admissible and calibrated),
    and Qual./runs: qualified / completed runs of the winning combination
    (the number of runs behind the winner's statistics). A ^ddagger marks
    a combination selected on admissible runs only (no combination has
    enough qualified runs); '---' means no combination has MIN_RUNS
    admissible runs.
    """
    import io

    tables_dir   = tables_dir or default_tables_dir(model)
    config_label = MODEL_CONFIG_LABEL[model]

    out = io.StringIO()

    def w(line=""):
        out.write(line + "\n")

    w(r"\begin{tabular}{llccccc}")
    w(r"\toprule")
    w(r"& & \multicolumn{4}{c}{All trials} & \multicolumn{1}{c}{Winning combination} \\")
    w(r"\cmidrule(lr){3-6} \cmidrule(lr){7-7}")
    w(r"\textbf{Case} & \textbf{Config.} & \textbf{Completed} & \textbf{Calibrated} & "
      r"\textbf{Admissible} & \textbf{Adm.\ \& calibrated} & \textbf{Qual./runs} \\")
    w(r"\midrule")

    for case in summary_cases():
        cfg = CASES[case]
        case_label = cfg["label4latex"].replace(" ", "~")
        noise_levels = cfg.get("noise_levels", [None])

        for nl in noise_levels:
            group_label = case_label
            if nl is not None:
                group_label += rf", \num{{{int(round(nl * 100))}}}\% noise"

            for k, conf in enumerate(configs):
                _, path, _ = resolve_run(model, case, conf, nl)
                first = group_label if k == 0 else ""
                cells = [first, f"{config_label}{conf}"]

                if not os.path.exists(path):
                    print(f"[trial counts] missing: {path}")
                    cells += ["---"] * 5
                else:
                    c = count_trials(path)
                    if not c["winner_ok"]:
                        win = "---"
                    else:
                        n_win = (c["n_qualified"] if c["basis"] == "qualified"
                                 else c["n_admissible_win"])
                        win = f"{n_win}/{c['n_runs']}"
                        if c["basis"] != "qualified":
                            win += r"$^\ddagger$"
                    cells += [
                        str(c["completed"]), str(c["calibrated"]), str(c["admissible"]),
                        str(c["admissible_calibrated"]), win,
                    ]
                w("  " + " & ".join(cells) + r" \\")

    w(r"\bottomrule")
    w(r"\end{tabular}")

    result = out.getvalue()
    if save:
        os.makedirs(tables_dir, exist_ok=True)
        path = os.path.join(tables_dir, "latex_trial_counts.tex")
        with open(path, "w") as f:
            f.write(result)
        print(f"[saved] {path}")
    else:
        print(result)


# ---------------------------------------------------------------------------
# Winning-architecture summary across all cases (input to the final runs)
# ---------------------------------------------------------------------------

WINNERS_FILE_STEM = "winners"   # <results>/<model>/tables/winners.{json,csv}

# Metrics copied from the winning trial's record (when present)
_WINNER_METRICS = [
    "obj", "chi2_nu_train", "chi2_nu_test",
    "final_physics", "final_pde", "final_dirichlet", "final_derivative",
    "pde_loss_eval", "strong_form_rms", "elapsed_s",
]


def _py(v):
    """NumPy/pandas scalar -> plain Python (JSON-serialisable); NaN -> None."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def winners_paths(model: str, out_dir: str = None) -> tuple:
    """(json_path, csv_path) of the winners summary for one model.
    Default directory: <results>/<model>/tables (same as the LaTeX tables)."""
    base = Path(out_dir) if out_dir else Path(default_tables_dir(model))
    return (str(base / f"{WINNERS_FILE_STEM}.json"),
            str(base / f"{WINNERS_FILE_STEM}.csv"))


# Metrics summarised in log space (geometric mean, multiplicative std): they
# span orders of magnitude, so the arithmetic mean/std is dominated by the
# largest run. Display only; winner selection never uses these statistics.
_LOG_METRICS = ("physics", "strong_form_rms")


def _geo_stats(x: pd.Series) -> dict:
    """Geometric mean and multiplicative geometric std (log10-space std,
    ddof=1) of the positive finite entries of x."""
    v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size < 2:
        return dict(gmean=None, gsd=None)
    lg = np.log10(v)
    return dict(gmean=_py(10.0 ** lg.mean()), gsd=_py(10.0 ** lg.std(ddof=1)))


def collect_winners(model: str, configs: tuple = ("es", "gb")) -> list:
    """
    One entry per (case, configuration, noise level) of `model`, with the
    winning combination selected by _select_winning_architecture (most
    qualified runs among the search trials; see there) -- the same
    rule as the identifiability analysis. The reported
    trial is the representative run of that combination.

    If no combination is reproducibly admissible, the rank-1 trial is
    still reported, flagged admissible=False (the '---' rows of
    the trial-count table), so every study has a reported configuration.
    Calibration (obj < CALIBRATED_OBJ) is reported as a flag. Missing
    results files give status="missing".

    Each entry carries everything needed to retrain the winner: the
    hyperparameters, the NN-weight seed (init_seed, == Optuna trial number in
    the search), the schedule tag and the noise level.
    """
    rows = []
    for case in summary_cases():
        cfg = CASES[case]
        for nl in cfg.get("noise_levels", [None]):
            for conf in configs:
                _, path, tag = resolve_run(model, case, conf, nl)
                row = dict(
                    tag=tag, model=model, case=case,
                    case_label=cfg["label4latex"], config=conf,
                    noise_level=nl, results_path=path,
                )
                if not os.path.exists(path):
                    row["status"] = "missing"
                    rows.append(row)
                    continue

                search = load_results(path, verbose=False)
                if search.empty:
                    row["status"] = "no_search_trials"
                    rows.append(row)
                    continue

                winner, arch_tuple, arch_cols = _select_winning_architecture(search)
                calibrated = bool(winner["calibrated"])
                admissible = winner_is_valid(winner)

                # Runs behind the winner's statistics (compute_identifiability_stats):
                # qualified runs of the combination, or admissible ones on the
                # admissible basis.
                basis = str(winner["selection_basis"])
                n_stats = int(winner["n_qualified"] if basis == "qualified"
                              else winner["n_admissible"]) if admissible else 0

                # Mean and std of the reported metrics over the same runs
                m_mask = pd.Series(True, index=search.index)
                for c, v in zip(arch_cols, arch_tuple):
                    m_mask &= search[c] == v
                runs = search[m_mask]
                keep = is_admissible(runs)
                if basis == "qualified":
                    keep &= is_calibrated(runs)
                runs = runs[keep] if admissible else runs.iloc[0:0]
                _series = {m: pd.to_numeric(runs[m], errors="coerce")
                           for m in ("chi2_nu_train", "chi2_nu_test", "obj")
                           if m in runs.columns}
                _series["physics"] = physics_values(runs)
                if "strong_form_rms" in runs.columns:
                    _series["strong_form_rms"] = pd.to_numeric(
                        runs["strong_form_rms"], errors="coerce")
                metrics_stats = {}
                for m, x in _series.items():
                    d = dict(mean=_py(x.mean()), std=_py(x.std()))
                    if m in _LOG_METRICS:
                        d.update(_geo_stats(x))
                    metrics_stats[m] = d

                seed = winner.get("init_seed", winner["trial"])
                if pd.isna(seed):
                    seed = winner["trial"]

                row.update(
                    status          = "ok",
                    trial           = int(winner["trial"]),
                    init_seed       = int(seed),
                    n_completed     = int(len(search)),
                    calibrated      = calibrated,
                    admissible      = admissible,
                    physics_converged = admissible,   # alias read by final_runs/common.py
                    physics_metric  = dict(name=PHYSICS_METRIC,
                                           value=_py(winner["physics_metric"])),
                    n_admissible_in_search = int(is_admissible(search).sum()),
                    n_admissible_same_arch = int(winner["n_admissible"]),
                    n_stats        = n_stats,
                    metrics_stats  = metrics_stats,
                    combination     = dict(
                        basis          = str(winner["selection_basis"]),
                        n_runs         = int(winner["n_runs"]),
                        n_admissible   = int(winner["n_admissible"]),
                        n_qualified    = int(winner["n_qualified"]),
                        frac_qualified = float(winner["frac_qualified"]),
                        median_obj     = _py(winner["median_obj"]),
                    ),
                    hparams         = {c: (str(v) if c == "activation" else int(v))
                                       for c, v in zip(arch_cols, arch_tuple)},
                    metrics         = {m: _py(winner[m]) for m in _WINNER_METRICS
                                       if m in winner.index},
                    recovered       = {p: _py(winner[p]) for p in cfg["true_params"]
                                       if p in winner.index},
                )
                if "noise_levels" in cfg:   # ground truth known (Case 1)
                    row["recovery_error"] = {
                        p: _py(winner[f"err_{p}"]) for p in cfg["true_params"]
                        if f"err_{p}" in winner.index
                    }
                # Schedule settings stored by some backends (frontx records them)
                for k in ("warmup_epochs", "transition_epochs"):
                    if k in winner.index and pd.notna(winner[k]):
                        row[k] = int(winner[k])
                rows.append(row)
    return rows


def group_key(row: dict) -> str:
    """Study group within which -es and -gb compete: the case, plus the
    noise level for cases that have one (e.g. 'grenoblesand_1pct_noise')."""
    key = row["case"]
    if row.get("noise_level") is not None:
        key += f"_{int(round(row['noise_level'] * 100))}pct_noise"
    return key


def select_configurations(rows: list) -> dict:
    """
    Choose, per study group (case [+ noise level]), which configuration
    (-es or -gb) is carried forward to the final runs. Adds 'selected' (bool)
    and 'selection_reason' (str) to every row in place and returns
    {group_key: selected tag or None}.

    Rule, on the winning combinations (see _select_winning_architecture):
      * selection basis first: "qualified" > "admissible" > "none";
      * then more qualified (resp. admissible) runs, higher qualified
        fraction, lower median obj;
      * basis "none" for all -> lower physics metric (flagged, with a warning);
      * only one study available -> that one (flagged if not admissible).
    Calibration is not part of the rule (it is reported per entry).
    """
    def _phys(r):
        v = r.get("physics_metric", {}).get("value")
        return float("inf") if v is None else v

    groups = {}
    for r in rows:
        r["selected"], r["selection_reason"] = False, None
        r["physics_converged"] = bool(r.get("status") == "ok" and r.get("admissible"))
        groups.setdefault(group_key(r), []).append(r)

    selected = {}
    for key, members in groups.items():
        ok = [r for r in members if r.get("status") == "ok"]
        if not ok:
            selected[key] = None
            continue
        conv = [r for r in ok if r["physics_converged"]]
        _order = {"qualified": 0, "admissible": 1, "none": 2}

        def _key(r):
            c = r["combination"]
            n = c["n_qualified"] if c["basis"] == "qualified" else c["n_admissible"]
            m = c["median_obj"] if c["median_obj"] is not None else float("inf")
            return (_order[c["basis"]], -n, -c["frac_qualified"], m)

        if len(conv) == 1:
            best, reason = conv[0], "only configuration with a reproducible admissible combination"
        elif len(conv) > 1:
            best = min(conv, key=_key)
            b = best["combination"]
            reason = (f"both admissible; {b['basis']} basis, most qualified runs "
                      f"({b['n_qualified']}/{b['n_runs']}), then fraction, then median obj")
        else:
            best = min(ok, key=_phys)
            reason = f"NONE admissible; lowest {PHYSICS_METRIC}"
            print(f"  WARNING [{key}]: no configuration has an admissible winner "
                  f"({PHYSICS_METRIC} < {ADMISSIBLE_PHYSICS:.0e}); selected "
                  f"{best['tag']} (lowest {PHYSICS_METRIC}).")
        if len(ok) == 1:
            reason = "only available study" + ("" if best["physics_converged"]
                                               else "; NOT admissible")
        best["selected"], best["selection_reason"] = True, reason
        selected[key] = best["tag"]
    return selected


def save_winners(rows: list, model: str, out_dir: str = None) -> tuple:
    """
    Write the winners summary as JSON (machine-readable, keyed by tag) and
    CSV (one flat row per study). Returns (json_path, csv_path).

    JSON layout:
        {"model": ..., "selection_rule": ..., "thresholds": {...},
         "winners": {"<tag>": {...entry from collect_winners...}, ...}}
    The tag is the same one used in the figure/table file names, e.g.
    "grenoblesand_gb_1pct_noise" or "letd_es".
    """
    json_path, csv_path = winners_paths(model, out_dir)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    payload = dict(
        model          = model,
        selection_rule = (
            "combination with the most qualified runs (admissible and "
            f"calibrated; >= {MIN_RUNS} required); ties: higher "
            "qualified fraction, then lower median obj; fallback: same on "
            "admissible runs only; reported trial = run with the median obj. "
            "Failed trials excluded, pruned trials included."),
        thresholds     = dict(calibrated_obj=CALIBRATED_OBJ,
                              admissible_physics=ADMISSIBLE_PHYSICS,
                              physics_metric=PHYSICS_METRIC,
                              min_qualified_runs=MIN_RUNS),
        selection_rule_configuration = (
            "per case [+ noise level]: the -es/-gb configuration with a "
            "reproducible admissible combination; if both, qualified basis "
            "first, then more qualified runs, higher fraction, lower median obj"),
        selected       = {group_key(r): r["tag"] for r in rows if r.get("selected")},
        winners        = {r["tag"]: r for r in rows},
    )
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[saved] {json_path}")

    flat = []
    for r in rows:
        d = {k: v for k, v in r.items()
             if k not in ("hparams", "metrics", "recovered", "recovery_error", "combination")}
        d.update({f"combo_{k}": v for k, v in r.get("combination", {}).items()})
        d.update(r.get("hparams", {}))
        d.update(r.get("metrics", {}))
        d.update({f"rec_{k}": v for k, v in r.get("recovered", {}).items()})
        d.update({f"err_{k}": v for k, v in r.get("recovery_error", {}).items()})
        flat.append(d)
    pd.DataFrame(flat).convert_dtypes().to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")
    return json_path, csv_path


def load_winner(tag: str, model: str = "hp_vpinn", path: str = None) -> dict:
    """Read one entry of the winners summary (used by the final-run scripts)."""
    json_path = path or winners_paths(model)[0]
    with open(json_path) as f:
        winners = json.load(f)["winners"]
    if tag not in winners:
        raise KeyError(f"{tag!r} not in {json_path}; available: {sorted(winners)}")
    return winners[tag]


def load_selected(case: str, noise_level: float = None, model: str = "hp_vpinn",
                  path: str = None) -> dict:
    """Winners entry of the configuration selected for the final runs of a
    case (and noise level, for Case 1), e.g. load_selected('grenoblesand', 0.01)."""
    json_path = path or winners_paths(model)[0]
    with open(json_path) as f:
        payload = json.load(f)
    key = group_key(dict(case=case, noise_level=noise_level))
    tag = payload.get("selected", {}).get(key)
    if tag is None:
        raise KeyError(f"No selected configuration for {key!r} in {json_path}")
    return payload["winners"][tag]


def _stacked_geo(st, fmt=".2e"):
    """Geometric mean over 'x/ gsd' (multiplicative std), stacked. Falls back
    to the arithmetic mean/std for entries without log statistics."""
    if not st:
        return "---"
    if st.get("gmean") is None or st.get("gsd") is None:
        if st.get("mean") is None or st.get("std") is None:
            return "---"
        return (r"\shortstack{\num{%s}\\{\scriptsize$\pm$\num{%s}}}"
                % (format(st["mean"], fmt), format(st["std"], fmt)))
    return (r"\shortstack{\num{%s}\\{\scriptsize$\times\!/\!\div$\num{%.2f}}}"
            % (format(st["gmean"], fmt), st["gsd"]))


def print_winners_latex(rows: list, model: str, tables_dir: str = None,
                        save: bool = True) -> None:
    """
    LaTeX tabular of the winning configuration per case, configuration and
    noise level (<tables_dir>/latex_winners.tex). Non-admissible winners
    are marked with ^* on the trial number, as in the trial-count table;
    non-calibrated winners (obj >= CALIBRATED_OBJ) with ^dagger on obj.
    The physics column is the admissibility metric (PHYSICS_METRIC).
    The configuration selected for the final runs (select_configurations)
    is set in bold.
    N is the number of runs of the winning combination behind its
    statistics (qualified runs; admissible runs, marked ^ddagger, when the
    combination was selected on the admissible basis). Runs out of the
    combination's completed runs are in the trial-count table. Caption and
    label stay in the SI.
    """
    import io

    tables_dir   = tables_dir or default_tables_dir(model)
    config_label = MODEL_CONFIG_LABEL[model]
    has_hp_vpinn_cols = any(
        "n_elements" in r.get("hparams", {}) for r in rows if r.get("status") == "ok"
    )

    met_keys  = ["chi2_nu_train", "chi2_nu_test", "obj", "physics"]
    met_heads = [METRIC_LABELS["chi2_nu_train"], METRIC_LABELS["chi2_nu_test"],
                 METRIC_LABELS["obj"],
                 PHYSICS_METRIC_LABEL.get(PHYSICS_METRIC, PHYSICS_METRIC),
                 METRIC_LABELS["strong_form_rms"]]
    n_hp = 2 if has_hp_vpinn_cols else 1     # "W x L, act." and "N_el, p"
    n_met = len(met_heads)
    hp_start, hp_end = 3, 2 + n_hp
    met_start, met_end = hp_end + 1, hp_end + n_met

    def stacked(st, fmt, sci=False):
        """mean over +- std, stacked (no extra package needed)."""
        if not st or st.get("mean") is None or st.get("std") is None:
            return "---"
        f = ".2e" if sci else fmt
        m, sd = format(st["mean"], f), format(st["std"], f)
        return r"\shortstack{\num{%s}\\{\scriptsize$\pm$\num{%s}}}" % (m, sd)

    out = io.StringIO()

    def w(line=""):
        out.write(line + "\n")

    w(r"\begin{tabular}{ll " + "c" * n_hp + " " + "c" * n_met + " c}")
    w(r"\toprule")
    w(rf"& & \multicolumn{{{n_hp}}}{{c}}{{Winning HPs}} & "
      rf"\multicolumn{{{n_met}}}{{c}}{{Metrics (mean $\pm$ std)}} & \\")
    w(rf"\cmidrule(lr){{{hp_start}-{hp_end}}}\cmidrule(lr){{{met_start}-{met_end}}}")
    hp_heads = [r"W$\times$L, act."] + ([r"$N_{\mathrm{el}}, p$"] if has_hp_vpinn_cols else [])
    w(r"\textbf{Case} & \textbf{Config.} & " + " & ".join(hp_heads)
      + " & " + " & ".join(met_heads) + r" & $N$ \\")
    w(r"\midrule")

    prev_group = None
    for r in rows:
        group = r["case_label"].replace(" ", "~")
        if r["noise_level"] is not None:
            group += rf", \num{{{int(round(r['noise_level'] * 100))}}}\% noise"
        first = group if group != prev_group else ""
        if prev_group is not None and group != prev_group:
            w(r"\addlinespace")
        prev_group = group
        conf_cell = f"{config_label}{r['config']}"
        if r.get("selected"):
            conf_cell = rf"\textbf{{{conf_cell}}}"
        cells = [first, conf_cell]

        if r.get("status") != "ok":
            cells += ["---"] * (n_hp + n_met + 1)
            w("  " + " & ".join(cells) + r" \\")
            continue

        hp, ms = r["hparams"], r.get("metrics_stats", {})
        cells.append(f"{int(hp['nn_width'])}$\\times${int(hp['nn_n_layers'])}, {hp['activation']}")
        if has_hp_vpinn_cols:
            cells.append(f"{hp.get('n_elements', '---')}, {hp.get('poly_degree', '---')}")
        cells += [
            stacked(ms.get("chi2_nu_train"), ".2f"),
            stacked(ms.get("chi2_nu_test"), ".2f"),
            stacked(ms.get("obj"), ".2f"),
            _stacked_geo(ms.get("physics")),
            _stacked_geo(ms.get("strong_form_rms")),
            ("---" if not r["admissible"] else
             str(r["n_stats"]) + ("" if r["combination"]["basis"] == "qualified"
                                   else r"$^\ddagger$")),
        ]
        w("  " + " & ".join(cells) + r" \\")

    w(r"\bottomrule")
    w(r"\end{tabular}")

    result = out.getvalue()
    if save:
        os.makedirs(tables_dir, exist_ok=True)
        path = os.path.join(tables_dir, "latex_winners.tex")
        with open(path, "w") as f:
            f.write(result)
        print(f"[saved] {path}")
    else:
        print(result)


def build_winners_summary(model: str, tables_dir: str = None,
                          out_dir: str = None,
                          configs: tuple = ("es", "gb")) -> list:
    """Collect, print, and save (LaTeX + JSON + CSV) the winners of one model."""
    rows = collect_winners(model, configs)
    out_dir = out_dir or tables_dir   # all outputs together unless overridden
    selected = select_configurations(rows)

    print(f"\nWinning configurations -- model={model} "
          f"(* = not admissible: {PHYSICS_METRIC} >= {ADMISSIBLE_PHYSICS:.0e}; "
          f"+ = not calibrated: obj >= {CALIBRATED_OBJ:.2g})")
    for r in rows:
        if r.get("status") != "ok":
            print(f"  {r['tag']:32s} {r['status']}")
            continue
        hp = ", ".join(f"{k}={v}" for k, v in r["hparams"].items())
        m = r["metrics"]
        print(f"  {r['tag']:32s} trial {r['trial']:>4d}{'' if r['admissible'] else '*'}  "
              f"{hp}  obj={m.get('obj', float('nan')):.3g}{'' if r['calibrated'] else '+'}  "
              f"{PHYSICS_METRIC}={r['physics_metric']['value'] or float('nan'):.2e}  "
              f"N={r['n_stats']}  "
              f"[{r['combination']['basis']}: {r['combination']['n_qualified']} qualified / "
              f"{r['combination']['n_admissible']} admissible / {r['combination']['n_runs']} runs]")

    print(f"\nSelected configuration per study (admissible: {PHYSICS_METRIC} < {ADMISSIBLE_PHYSICS:.0e}):")
    by_tag = {r["tag"]: r for r in rows}
    for key, tag in selected.items():
        reason = by_tag[tag]["selection_reason"] if tag else "no results"
        print(f"  {key:28s} -> {tag or '---':32s} ({reason})")

    print_winners_latex(rows, model, tables_dir=tables_dir)
    save_winners(rows, model, out_dir=out_dir)
    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by the per-run mode and the --trial-counts mode."""
    parser.add_argument(
        "--model", required=True, choices=MODELS,
        help="Which model's results to process (also the subdirectory name).",
    )
    parser.add_argument(
        "--figures-dir", default=None,
        help="Output directory for figures (default: <results>/<model>/Figures).",
    )
    parser.add_argument(
        "--tables-dir", default=None,
        help="Output directory for LaTeX tables and the winners summary "
             "(default: <results>/<model>/tables).",
    )
    parser.add_argument(
        "--calibrated-obj", type=float, default=None,
        help=f"Override the calibration threshold obj < X "
             f"(default: {CALIBRATED_OBJ}). Needed if the objective's scale "
             f"changes.",
    )
    parser.add_argument(
        "--admissible-physics", type=float, default=None,
        help=f"Override the admissibility threshold: physics metric < X "
             f"(default: {ADMISSIBLE_PHYSICS:.0e}).",
    )
    parser.add_argument(
        "--physics-metric", choices=["eval", "train"], default=None,
        help="Physics metric for admissibility: 'eval' = held-out-quadrature "
             "PDE loss pde_loss_eval (default), 'train' = training-quadrature "
             "physics loss final_physics (the previous rule).",
    )
    parser.add_argument(
        "--min-runs", type=int, default=None,
        help=f"Minimum number of qualified (fallback: admissible) runs of a "
             f"hyperparameter combination for it to be a winner "
             f"(default: {MIN_RUNS}).",
    )
    parser.add_argument(
        "--obj-tie-tol", type=float, default=None,
        help=f"Treat admissible trials within this obj of the best as tied and "
             f"order them by the physics metric (default: {OBJ_TIE_TOL:g} = off).",
    )


def _apply_threshold_overrides(args) -> None:
    """Module-level thresholds are read by several functions; override them
    once here, before any of those functions runs."""
    global CALIBRATED_OBJ, ADMISSIBLE_PHYSICS, PHYSICS_METRIC, OBJ_TIE_TOL, MIN_RUNS
    if args.calibrated_obj is not None:
        CALIBRATED_OBJ = args.calibrated_obj
    if args.admissible_physics is not None:
        ADMISSIBLE_PHYSICS = args.admissible_physics
    if getattr(args, "physics_metric", None) is not None:
        PHYSICS_METRIC = {"eval": "pde_loss_eval", "train": "final_physics"}[args.physics_metric]
    if getattr(args, "min_runs", None) is not None:
        MIN_RUNS = args.min_runs
    if getattr(args, "obj_tie_tol", None) is not None:
        OBJ_TIE_TOL = args.obj_tie_tol
    print(f"[classification] admissible: {PHYSICS_METRIC} < {ADMISSIBLE_PHYSICS:.0e} | "
          f"calibrated: obj < {CALIBRATED_OBJ:g} | min runs: {MIN_RUNS} | ranking: {ranking_description()}")


def main() -> None:
    # --trial-counts: build the trial-count table for all cases of one model and exit
    parser_tc = argparse.ArgumentParser(add_help=False)
    parser_tc.add_argument("--trial-counts", action="store_true")
    args_tc, _ = parser_tc.parse_known_args()
    if args_tc.trial_counts:
        parser = argparse.ArgumentParser(
            description="Build the trial-count table for all cases of one model."
        )
        parser.add_argument("--trial-counts", action="store_true")
        _add_common_args(parser)
        args = parser.parse_args()
        _apply_threshold_overrides(args)
        print_trial_counts_latex(args.model, tables_dir=args.tables_dir)
        return

    # --winners: winning-architecture summary for all cases of one model
    parser_w = argparse.ArgumentParser(add_help=False)
    parser_w.add_argument("--winners", action="store_true")
    args_w, _ = parser_w.parse_known_args()
    if args_w.winners:
        parser = argparse.ArgumentParser(
            description="Build the winning-architecture summary (LaTeX, JSON, "
                        "CSV) for all cases of one model."
        )
        parser.add_argument("--winners", action="store_true")
        _add_common_args(parser)
        parser.add_argument(
            "--winners-dir", default=None,
            help="Output directory for winners.json/.csv (default: same as "
                 "--tables-dir, i.e. <results>/<model>/tables).",
        )
        args = parser.parse_args()
        _apply_threshold_overrides(args)
        build_winners_summary(args.model, tables_dir=args.tables_dir,
                              out_dir=args.winners_dir)
        return

    parser_tmp = argparse.ArgumentParser(add_help=False)
    parser_tmp.add_argument("--case", required=True, choices=list(CASES))
    args_tmp, _ = parser_tmp.parse_known_args()
    cfg_tmp = CASES[args_tmp.case]

    parser = argparse.ArgumentParser(
        description="Post-process Optuna results and generate SI figures."
    )
    _add_common_args(parser)
    parser.add_argument("--case", required=True, choices=list(CASES))

    if "noise_levels" in cfg_tmp:
        valid_noise_strs = [str(s) for s in cfg_tmp["noise_levels"]]
        parser.add_argument(
            "--noise-level", type=str, required=True,
            choices=valid_noise_strs,
            metavar=f"{{{','.join(valid_noise_strs)}}}",
        )

    parser.add_argument(
        "--results-dir", default=None,
        help="Override the case's base results-directory name (relative to "
             "<repo>/<model>/, or absolute).",
    )
    parser.add_argument("--es-or-gb", choices=["es", "gb"], default=None)
    parser.add_argument("--n-top", type=int, default=20)
    parser.add_argument(
        "--clip-upper", type=float, default=None,
        help="Y-axis clip for marginal box plots and obj filter in the "
             "scatter panels (default: the calibration threshold).",
    )
    parser.add_argument(
        "--trial-counts", action="store_true",
        help="Only build the trial-count table for all cases of --model "
             "(latex_trial_counts.tex) and exit; --case is then not needed.",
    )
    args = parser.parse_args()
    _apply_threshold_overrides(args)

    cfg         = CASES[args.case]
    true_params = cfg["true_params"]
    label       = cfg["label"]
    clip_upper  = args.clip_upper if args.clip_upper is not None else CALIBRATED_OBJ

    # Paths and file tag, built from the CLI arguments so that output names
    # do not depend on the results directory name.
    results_dir, results_path, tag = resolve_run(
        args.model, args.case, args.es_or_gb,
        getattr(args, "noise_level", None),
        results_dir=args.results_dir,
    )

    figures_dir = args.figures_dir or default_figures_dir(args.model)
    tables_dir = args.tables_dir or default_tables_dir(args.model)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    counts = count_trials(results_path)
    df = load_results(results_path)
    print(f"Model: {args.model}  |  Case: {label}  |  "
          f"{counts['completed']} completed, "
          f"{counts['admissible']} admissible ({PHYSICS_METRIC} < {ADMISSIBLE_PHYSICS:.0e}), "
          f"{counts['admissible_calibrated']} of them calibrated (obj < {CALIBRATED_OBJ:g})"
          f"{'' if counts['winner_ok'] else f' -- no combination with >= {MIN_RUNS} admissible runs'}.")

    df_search = df   # load_results keeps the search trials only

    print_top_trials_latex(
        df_search, true_params, args.case,
        n=args.n_top,
        tag=tag,
        tables_dir=tables_dir,
    )

    print_top_trials(df_search, true_params, args.case,
                     n=args.n_top, results_dir=results_dir)

    plot_figure_A(df_search, tag, figures_dir, clip_upper=clip_upper)
    plot_architecture_occurrences(df_search, tag, figures_dir)

    if has_ground_truth(args.case):
        plot_figure_B(df_search, true_params, tag, figures_dir,
                      clip_upper=clip_upper)

    ident = compute_identifiability_stats(
        df, true_params, args.case,
        results_dir=results_dir,
    )
    if ident is not None:
        print_identifiability_latex(
            ident, true_params, args.case,
            tag=tag,
            tables_dir=tables_dir,
        )


if __name__ == "__main__":
    main()

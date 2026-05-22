"""
hparam_postprocess.py
=====================
Post-process Optuna hyperparameter search results and generate
publication-quality compound figures for the Supplementary Information.

Two figures are produced per study:

  Figure A  (all cases)   — chi2_nu_train vs chi2_nu_test  (left)
                            + marginal box plots             (right)

  Figure B  (Case 1 only) — parameter recovery scatter,
                            one panel per recovered parameter

Usage
-----
    python hparam_postprocess.py --case letd --es-or-gb gb
    python hparam_postprocess.py --case grenoblesand --es-or-gb gb --noise-level 0.01
"""

import argparse
import json
import os
import shutil

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from mpl_toolkits.axes_grid1 import make_axes_locatable


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
# Per-case configuration
# ---------------------------------------------------------------------------

CASES = {
    "grenoblesand": dict(
        label           = "Grenoble Sand",
        label4latex     = "Case 1",
        results_subdir  = "results_grenoblesand",
        results_file    = "hparam_results_grenoblesand.jsonl",
        true_params     = dict(
            KsAlpha = [r"$K_{\mathrm{s}}/\alpha$", 15.37 / 0.0432],
            m       = [r"$m$", 0.5096],
        ),
        noise_levels = [0.01, 0.02],
    ),
    "letd": dict(
        label           = "LETd",
        label4latex     = "Case 2",
        results_subdir  = "results_letd",
        results_file    = "hparam_results_letd.jsonl",
        true_params     = dict(
            Dwt     = 1.004e-3,
            L       = 1.356,
            E       = 10_010.0,
            T       = 1.224,
            theta_r = 0.00625,
        ),
    ),
    "vangenuchten": dict(
        label           = "Van Genuchten",
        label4latex     = "Case 3",
        results_subdir  = "results_vangenuchten",
        results_file    = "hparam_results_vangenuchten.jsonl",
        true_params     = dict(
            KsAlpha = 2.079e-6,
            m       = 0.8764,
            l       = 2.344,
            theta_r = 0.004943,
        ),
    ),
}

HPARAM_COLS = ["nn_n_layers", "nn_width", "activation", "n_elements", "poly_degree"]

# Readable LaTeX labels for hyperparameter columns
HPARAM_LABELS = {
    "nn_n_layers" : r"$L$",
    "nn_width"    : r"$W$",
    "activation"  : "Activation",
    "n_elements"  : r"$N_{\mathrm{el}}$",
    "poly_degree" : r"$p$",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: str) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    if "pruned" not in df.columns:
        df["pruned"] = False

    mask = ~df["failed"]
    df = (
        df[mask]
        .drop(columns=["failed", "pruned"], errors="ignore")
        .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

def _panel_label(ax, label: str) -> None:
    """Embed a bold panel letter in the top-left corner."""
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
    chi2_nu_train vs chi2_nu_test scatter for trials with obj < clip_obj.
    Colorbar is placed as a horizontal inset in the bottom-right quadrant.
    """
    df_filt = df[df["obj"] < clip_upper].copy()
    min_obj = df_filt["obj"].min()
    max_obj = df_filt["obj"].max()
    n_total = len(df)
    n_kept  = len(df_filt)

    if n_kept == 0:
        ax.text(0.5, 0.5, "No trials with obj < {clip_obj:.2g}",
                ha="center", va="center", transform=ax.transAxes)
        return

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
        vmin   = 0.0,
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
    cbar.set_ticks([0, clip_upper/2, clip_upper])
    cbar.set_ticklabels(
        [rf"${min_obj:.2g}$", rf"${(clip_upper-min_obj)/2:.2g}$", rf"${clip_upper:.2g}$"]
    )
    cax.set_title(rf"$\mathrm{{obj}} < {clip_upper:.2g}$  " rf"({n_kept}/{n_total} trials)", fontsize=8, pad=2.5)


def _draw_marginals(ax, df: pd.DataFrame, col: str,
                    clip_upper: float = 2.0) -> None:
    """
    Box plot of chi2_nu_test for each value of one hyperparameter,
    clipped at clip_upper (default 2.0) to focus on the meaningful region.
    """
    rng = np.random.default_rng(0)

    unique_vals = sorted(df[col].unique())
    groups      = [df[df[col] == v]["obj"].values for v in unique_vals]
    labels      = [str(v) for v in unique_vals]

    clipped = [g[g <= clip_upper] for g in groups]

    ax.boxplot(
        clipped, labels=labels, patch_artist=True,
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
    Scatter of recovered parameter value vs chi2_nu_test,
    colour-coded by |chi2_nu_train - 1|.
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

def plot_figure_A(df: pd.DataFrame, 
                  figure_name: str,
                  figure_dir: str,
                  clip_upper: float = 2.0) -> None:
    """
    Compound figure with two rows:
      Row 1 — chi2_nu_train vs chi2_nu_test scatter  (single panel, left)
              + first two hyperparameter marginals    (right)
      Row 2 — remaining hyperparameter marginals

    Panel letters: a) scatter, b) L, c) W, d) Activation, e) N_el, f) p
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

    path = os.path.join(figure_dir, f"diagnostics_{figure_name}.eps")
    fig.savefig(path, format="eps", bbox_inches="tight")
    print(f"[saved] {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure B: parameter recovery  (Case 1 only)
# ---------------------------------------------------------------------------

def plot_figure_B(df: pd.DataFrame, true_params: dict,
                  figure_name: str,
                  figure_dir: str,
                  clip_upper: float = 10.0) -> None:
    """
    One panel per recovered parameter, arranged in a single row.
    Panel letters start from a).
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
    path = os.path.join(figure_dir, f"recovery_{figure_name}.eps")
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
    if case == "grenoblesand":
        err_cols = [f"err_{p}" for p in true_params if f"err_{p}" in df.columns]

    candidate_cols = (
        ["trial", "nn_n_layers", "nn_width", "activation",
         "n_elements", "poly_degree", "init_seed",
         "chi2_nu_train", "chi2_nu_test", "obj",
         "final_physics", "elapsed_s"]
        + param_cols + err_cols
    )
    select_cols = [c for c in candidate_cols if c in df.columns]

    df = df.copy()
    df["_rank_primary"]   = df["obj"]
    df["_rank_secondary"] = df["final_physics"]
    top = (
        df.sort_values(
            ["_rank_primary", "_rank_secondary"],
            ascending=[True, True],
        )
        .head(n)[select_cols]
    )

    print(f"\nTop {n} trials (ranked by obj, then final_physics):")
    print(top.to_string(index=False, float_format="{:.4g}".format))

    if results_dir is not None:
        path = os.path.join(results_dir, "top_trials.csv")
        all_cols_df = df.sort_values(
            ["_rank_primary", "_rank_secondary"], ascending=[True, True]
        ).head(all_n)
        export_cols = [
            c for c in all_cols_df.columns
            if c not in ("noise_level",) and not c.startswith("_rank")
        ]
        all_cols_df[export_cols].to_csv(path, index=False)
        print(f"[saved] {path}")


def print_top_trials_latex(
    df: pd.DataFrame,
    true_params: dict,
    case: str,
    n: int = 20,
    results_dir: str = None,
) -> None:
    """
    Print (or save) a LaTeX table of the top-n trials ranked by the
    two-stage criterion (obj, then final_physics).

    Recovery error columns are included only for Case 1 (Grenoble sand),
    where the true parameters are known.  For Cases 2 and 3, the table
    omits recovered parameter values since no ground truth is available.

    Parameters
    ----------
    df          : DataFrame of non-failed trials.
    true_params : Case configuration dict (keys = parameter names).
    case        : Case identifier ('grenoblesand', 'vangenuchten', 'letd').
    n           : Number of top trials to include (default: 20).
    output_path : If given, write the LaTeX table to this .tex file.
                  If None, print to stdout.
    """
    import io

    df = df.copy()

    # --- Recovery error columns (Case 1 only) ---
    if case == "grenoblesand":
        err_cols = [
            f"err_{p}" for p in true_params
            if f"err_{p}" in df.columns
        ]
    else:
        err_cols = []

    # --- Ranking ---
    df["_rank_primary"]   = df["obj"]
    df["_rank_secondary"] = df["final_physics"]
    top = (
        df.sort_values(
            ["_rank_primary", "_rank_secondary"],
            ascending=[True, True],
        )
        .head(n)
        .copy()
    )

    # --- Convert errors to percentages ---
    for col in err_cols:
        top[col] = top[col] * 100

    # --- Rank number column ---
    top.insert(0, "rank_no", range(1, len(top) + 1))

    # --- Combined architecture string  (W x L) ---
    top["L_W"] = (
        top["nn_width"].astype(int).astype(str)
        + r"$\times$"
        + top["nn_n_layers"].astype(int).astype(str)
    )

    # --- Determine which HP columns are present ---
    has_elements = "n_elements" in top.columns
    has_degree   = "poly_degree" in top.columns

    hp_cols  = ["L_W", "activation"]
    hp_heads = [r"L$\times$W", "Act."]
    if has_elements:
        hp_cols.append("n_elements")
        hp_heads.append(r"$N_{\mathrm{el}}$")
    if has_degree:
        hp_cols.append("poly_degree")
        hp_heads.append(r"$p$")

    n_hp = len(hp_cols)

    metric_cols  = ["chi2_nu_train", "chi2_nu_test", "obj", "final_physics"]
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

    # Top-level headers
    hp_span  = rf"\multicolumn{{{n_hp}}}{{c}}{{Optimal HPs}}"
    met_span = rf"\multicolumn{{{n_metrics}}}{{c}}{{Metrics}}"
    row1 = (
        r"& \multicolumn{1}{c}{\raisebox{-1.5ex}{Trial}} "
        r"& \multicolumn{1}{c}{\raisebox{-1.5ex}{Time [s]}} "
        f"& {hp_span} & {met_span}"
    )
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

    # Sub-headers
    sub = r"No. & No. & & " + " & ".join(hp_heads)
    sub += (
        r" & $\chi^2_{\nu,\mathrm{train}}$"
        r" & $\chi^2_{\nu,\mathrm{test}}$"
        r" & $\mathrm{obj}$"
        r" & $\mathcal{L}_{\mathrm{physics}}$"
    )
    for col in err_cols:
        param = col.replace("err_", "")
        sub += rf" & $\varepsilon_{{{param}}}~[\%]$"
    sub += r" \\"
    w(sub)
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
            elif col in ("rank_no", "trial"):
                cells.append(str(int(val)))
            elif col in ("n_elements", "poly_degree"):
                cells.append(str(int(val)))
            elif col == "activation":
                cells.append(str(val))
            elif col == "L_W":
                cells.append(str(val))
            elif isinstance(val, float):
                if val == 0.0:
                    cells.append(r"\num{0}")
                elif "physics" in col or abs(val) < 1e-3:
                    cells.append(rf"\num{{{val:.3e}}}")
                else:
                    cells.append(rf"\num{{{val:.4g}}}")
            else:
                cells.append(str(val))
        w("  " + " & ".join(cells) + r" \\")

    w(r"\bottomrule")
    w(r"\end{tabular}")

    result = out.getvalue()

    if results_dir is not None:
        output_path = os.path.join("./Figures", f"latex_top20_{results_dir[8:]}.tex")
        with open(output_path, "w") as f:
            f.write(result)
        print(f"[saved] {output_path}")
    else:
        print(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser_tmp = argparse.ArgumentParser(add_help=False)
    parser_tmp.add_argument("--case", required=True, choices=list(CASES))
    args_tmp, _ = parser_tmp.parse_known_args()
    cfg_tmp = CASES[args_tmp.case]

    parser = argparse.ArgumentParser(
        description="Post-process Optuna results and generate SI figures."
    )
    parser.add_argument("--case", required=True, choices=list(CASES))

    if "noise_levels" in cfg_tmp:
        valid_noise_strs = [str(s) for s in cfg_tmp["noise_levels"]]
        parser.add_argument(
            "--noise-level", type=str, required=True,
            choices=valid_noise_strs,
            metavar=f"{{{','.join(valid_noise_strs)}}}",
        )

    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--figures-dir", default="./figures")
    parser.add_argument("--es-or-gb", choices=["es", "gb"], default=None)
    parser.add_argument("--n-top", type=int, default=20)
    parser.add_argument(
        "--clip-upper", type=float, default=2.0,
        help="Y-axis clip for marginal box plots (default: 2.0).",
    )
    parser.add_argument(
        "--filter-percentile", type=float, default=25.0,
        help="Percentile of |chi2_nu_train - 1| used to filter "
             "calibrated trials (default: 25).",
    )
    args = parser.parse_args()

    cfg         = CASES[args.case]
    true_params = cfg["true_params"]
    label       = cfg["label"]

    base_results_dir = args.results_dir
    base_results_subdir = cfg["results_subdir"]
    if args.es_or_gb is not None:
        base_results_subdir = f"{base_results_subdir}_{args.es_or_gb}"

    if "noise_levels" in cfg:
        noise_level  = float(args.noise_level)
        noise_pct    = int(noise_level * 100)
        results_subdir  = f"{base_results_subdir}_{noise_pct}pct_noise"
        results_file = f"hparam_results_{noise_pct}pct_noise.jsonl"
    else:
        results_subdir  = base_results_subdir
        results_file = cfg["results_file"]

    results_dir = os.path.join(base_results_dir, results_subdir)
    results_path = os.path.join(base_results_dir, results_subdir, results_file)

    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    df = load_results(results_path)
    print(f"Case: {label}  |  Loaded {len(df)} non-failed trials.")

    print_top_trials_latex(
        df, true_params, args.case,
        n=args.n_top,
        #results_dir=results_dir,
    )

    print_top_trials(df, true_params, args.case,
                     n=args.n_top, results_dir=results_dir)

    plot_figure_A(df, results_subdir[8:], args.figures_dir,
                  clip_upper=args.clip_upper)

    if args.case == "grenoblesand":
        plot_figure_B(df, true_params, results_subdir[8:], args.figures_dir,
                      clip_upper=args.clip_upper)


if __name__ == "__main__":
    main()

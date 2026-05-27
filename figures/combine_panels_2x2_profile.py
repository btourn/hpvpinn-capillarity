"""
combine_panels_2x2_profile.py
==============================
Build the 2×2 wetting profile compound figure (Fig. 2 in the manuscript)
from pre-computed CSV files.

Layout
------
    (a) Case 1, 1% noise,  hpVPINN-gb  — Grenoble sand
    (b) Case 1, 2% noise,  hpVPINN-gb  — Grenoble sand
    (c) Case 2, hpVPINN-es — Whatman, van Genuchten (with zoom inset)
    (d) Case 3, hpVPINN-gb — Whatman, LETd

Input format
------------
Two-column CSV files (phi, theta) with an optional '#' header line.
Each panel is configured via a dict; see the panel configuration
section below for accepted keys.

Panel configuration keys
------------------------
    sol        : path to the solution CSV (hp-VPINN prediction). Required.
    test       : path to the test data CSV. Optional.
    true       : path to the noise-free reference CSV. Optional.
    label      : panel letter, e.g. 'a)'. Optional.
    sol_legend : legend label for the solution curve.
    zoom       : ((x0,y0),(x1,y1)) zoom region for inset. Optional.
    inset_pos  : [left, bottom, width, height] in axes-fraction coords.

Usage
-----
    uv run python figures/combine_panels_2x2_profile.py

Output
------
    figures/fig2.eps
"""

import shutil

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# LaTeX configuration
# ---------------------------------------------------------------------------
_use_latex = shutil.which("latex") is not None

matplotlib.rcParams.update({
    "text.usetex":         _use_latex,
    "text.latex.preamble": r"\usepackage{amsbsy}",
    "font.family":         "serif",
    "font.serif":          ["Computer Modern Roman"],
    "axes.labelsize":      14,
    "xtick.labelsize":     12,
    "ytick.labelsize":     12,
    "legend.fontsize":     11,
    "figure.dpi":          300,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path):
    """Load a two-column (phi, theta) CSV with an optional '#' header."""
    return np.loadtxt(path, delimiter=",", comments="#")


# ---------------------------------------------------------------------------
# Single-panel plotter
# ---------------------------------------------------------------------------

def plot_panel(ax, cfg):
    """
    Draw one wetting-profile panel on *ax* from the configuration dict *cfg*.
    """
    sol   = load_csv(cfg["sol"])
    train = None
    if "train" in cfg and cfg["train"] is not None:
        train = load_csv(cfg["train"])
    test = None
    if "test" in cfg and cfg["test"] is not None:    
        test  = load_csv(cfg["test"])
    true  = load_csv(cfg["true"]) if cfg.get("true") else None

    phi_sol,   theta_sol   = sol[:, 0],   sol[:, 1]

    # --- noise-free reference ---
    if true is not None:
        ax.plot(
            true[:, 0], true[:, 1],
            color     = "black",
            linewidth = 1.2,
            linestyle = "--",
            marker    = "s",
            markersize= 3.5,
            markevery = 5,
            label     = "Noise-free solution",
            zorder    = 2,
        )

    # --- predicted profile ---
    ax.plot(
        phi_sol, theta_sol,
        color     = "blue",
        linewidth = 1.8,
        label     = cfg.get("sol_legend", r"\textit{hp}-VPINN"),
        zorder    = 3,
    )

    # --- training data ---
    if train is not None:
        phi_train, theta_train = train[:, 0], train[:, 1]
        ax.scatter(
            phi_train, theta_train,
            color     = "red",
            s         = 18,
            linewidths= 0.5,
            zorder    = 4,
            label     = cfg.get("train_legend", "Train"),
        )

    # --- test data ---
    if test is not None:
        phi_test,  theta_test  = test[:, 0],  test[:, 1]
        ax.scatter(
            phi_test, theta_test,
            facecolors= "none",
            edgecolors= "red",
            linewidths= 0.8,
            s         = 18,
            zorder    = 4,
            label     = cfg.get("test_legend", "Test"),
        )

    # --- axes ---
    phi_max = cfg.get("phi_max", 1.0)
    ax.set_xlim(0.0, phi_max)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$\tilde{\phi}$")
    ax.set_ylabel(r"$\tilde{\theta}$")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(
        loc       = cfg.get("legend_loc", "lower left"),
        framealpha= 0.85,
        edgecolor = "0.8",
    )

    # --- panel label ---
    label = cfg.get("label")
    if label:
        ax.text(
            0.90, 0.95, label,
            transform  = ax.transAxes,
            fontsize   = 13,
            fontweight = "bold",
            fontfamily = "sans-serif",
            va="top", ha="left",
        )

    # --- optional zoom inset ---
    zoom = cfg.get("zoom")
    if zoom is not None:
        (x0, y0), (x1, y1) = zoom
        inset_pos = cfg.get("inset_pos", [0.30, 0.35, 0.30, 0.55])

        ax_ins = ax.inset_axes(
            inset_pos,
            xlim=(x0, x1),
            ylim=(y0, y1),
        )

        if true is not None:
            ax_ins.plot(
                true[:, 0], true[:, 1],
                color="black", linewidth=1.2, linestyle="--",
                marker="s", markersize=3.5, markevery=5, zorder=2,
            )
        ax_ins.plot(phi_sol, theta_sol,
                    color="blue", linewidth=1.8, zorder=3)
        #ax_ins.scatter(phi_train, theta_train,
        #               color="red", s=10, linewidths=0.4, zorder=4)
        ax_ins.scatter(phi_test, theta_test,
                       facecolors="none", edgecolors="red",
                       linewidths=0.6, s=10, zorder=4)

        ax_ins.tick_params(labelsize=7)
        ax_ins.set_xlabel(r"$\tilde{\phi}$", fontsize=8, labelpad=2)
        ax_ins.set_ylabel(r"$\tilde{\theta}$", fontsize=8, labelpad=2)

        ax.indicate_inset_zoom(
            ax_ins, edgecolor="0.4", linewidth=0.8, alpha=0.8,
        )


# ---------------------------------------------------------------------------
# Master figure
# ---------------------------------------------------------------------------

def generate_profile_figure(panels, output_name="fig2.eps"):
    """
    Build a 2×2 compound figure from a list of 4 panel configuration dicts.

    Parameters
    ----------
    panels      : list of 4 dicts (see module docstring for keys)
    output_name : output file path (.eps or .pdf)
    """
    assert len(panels) == 4, "Exactly 4 panel configurations are required."

    fig = plt.figure(figsize=(10, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for cfg, (row, col) in zip(panels, positions):
        ax = fig.add_subplot(gs[row, col])
        plot_panel(ax, cfg)

    fig.savefig(output_name, format=output_name.split(".")[-1],
                bbox_inches="tight")
    print(f"[saved] {output_name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    BASE = "./results/results_grenoblesand"

    panels = [
        # --- (a) Grenoble sand, 1 % noise, hpVPINN-gb ---
        dict(
            sol         = f"{BASE}_gb_1pct_noise/profile_sol_trial226.csv",
            #train       = f"{BASE}_gb_1pct_noise/profile_train_trial226.csv",
            test        = f"{BASE}_gb_1pct_noise/profile_test_trial226.csv",
            true        = f"{BASE}_gb_1pct_noise/profile_true.csv",
            sol_legend  = r"\textit{hp}-VPINN-gb",
            train_legend= r"Train (1\% noise)",
            test_legend = r"Test (independent 1\% noise)",
            label       = "a)",
            zoom        = None,
        ),
        # --- (b) Grenoble sand, 2 % noise, hpVPINN-gb ---
        dict(
            sol         = f"{BASE}_gb_2pct_noise/profile_sol_trial236.csv",
            #train       = f"{BASE}_gb_2pct_noise/profile_train_trial236.csv",
            test        = f"{BASE}_gb_2pct_noise/profile_test_trial236.csv",
            true        = f"{BASE}_gb_2pct_noise/profile_true.csv",
            sol_legend  = r"\textit{hp}-VPINN-gb",
            train_legend= r"Train (2\% noise)",
            test_legend = r"Test (independent 2\% noise)",
            label       = "b)",
            zoom        = None,
        ),
        # --- (c) Whatman, van Genuchten, hpVPINN-es  (zoom inset) ---
        dict(
            sol         = "./results/results_vangenuchten_es/profile_sol_trial182.csv",
            #train       = "./results/results_vangenuchten_es/profile_train_trial182.csv",
            test        = "./results/results_vangenuchten_es/profile_test_trial182.csv",
            true        = None,
            sol_legend  = r"\textit{hp}-VPINN-es",
            label       = "c)",
            legend_loc  = "lower left",
            zoom        = ((0.80, 0.10), (0.90, 0.80)),
            inset_pos   = [0.30, 0.35, 0.30, 0.55],
        ),
        # --- (d) Whatman, LETd, hpVPINN-gb ---
        dict(
            sol         = "./results/results_letd_gb/profile_sol_trial72.csv",
            #train       = "./results/results_letd_gb/profile_train_trial72.csv",
            test        = "./results/results_letd_gb/profile_test_trial72.csv",
            true        = None,
            sol_legend  = r"\textit{hp}-VPINN-gb",
            label       = "d)",
        ),
    ]

    generate_profile_figure(panels, output_name="./figures/fig2.eps")

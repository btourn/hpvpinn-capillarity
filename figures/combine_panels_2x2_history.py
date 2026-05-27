"""
combine_panels_2x2_history.py
==============================
Build the 2×2 convergence history figure (Fig. 4 in the manuscript) for
the hpVPINN-gb configurations on Case 1 (Grenoble sand) at 1% and 2%
noise.

Layout
------
    Row 1: physics loss L_phys and its EMA L_phys_hat vs epochs, with
            the convergence threshold (dashed) and freeze epoch (dotted).
    Row 2: recovered parameter trajectories KsAlpha and m vs epochs
            (twin-axis), with true values (dashed) and freeze epoch (dotted).

Input format
------------
Parquet files with one row per training epoch. Column indices used:
    2 : physics loss
    5 : EMA of physics loss
    6 : EMA freeze flag (bool)
    7 : KsAlpha trajectory
    8 : m trajectory

Usage
-----
    uv run python figures/combine_panels_2x2_history.py

Output
------
    figures/fig4.eps
"""

import shutil
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# --- LaTeX Configuration (from your original script) ---
_use_latex = shutil.which("latex") is not None

matplotlib.rcParams.update({
    "text.usetex":            _use_latex,
    "text.latex.preamble":    r"\usepackage{amsbsy}",
    "font.family":            "serif",
    "font.serif":             ["Computer Modern Roman"],
    "axes.labelsize":         14,
    "xtick.labelsize":        12,
    "ytick.labelsize":        12,
    "legend.fontsize":        12,
    "figure.dpi":             300,
})


def plot_losses(ax, history_csv, tol, panel_label=None):

    # Load the history data
    data = pd.read_parquet(history_csv).to_numpy()
    epochs = np.arange(1, len(data) + 1)
    loss_phys = data[:, 2]
    loss_phys_ema = data[:, 5]
    loss_phys_ema_frozen = data[:, 6]

    intercept = np.where(loss_phys_ema_frozen)[0][0]

    # Plotting
    ax.semilogy(epochs, loss_phys, label=r'$\mathcal{L}_{\mathrm{phys}}$', color='black')
    ax.semilogy(epochs, loss_phys_ema, label=r'$\hat{\mathcal{L}}_{\mathrm{phys}}\,(\mathrm{EMA})$', color='green')
    ax.axhline(tol, color='gray', linestyle='--')
    ax.axvline(intercept, color='black', linestyle=':', linewidth=1)

    ax.set_xlim(0, 200000)
    ax.set_xticks([50000, 100000, 150000, 200000])
    ax.set_xticklabels(['50k', '100k', '150k', '200k'])

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    if panel_label:
        ax.text(0.08, 0.9, panel_label, transform=ax.transAxes, fontsize=13, fontweight='bold', va='top')

    ax.legend(loc="upper right")


def plot_parameters(ax, history_csv, true_params, panel_label=None):
    # 1. Load the history data
    # Note: Using indices 7 and 8 as per your snippet
    data = pd.read_parquet(history_csv).to_numpy()
    epochs = np.arange(1, len(data) + 1)
    loss_phys_ema_frozen = data[:, 6]
    KsAlpha = data[:, 7]
    m = data[:, 8]

    intercept = np.where(loss_phys_ema_frozen)[0][0]

    # 2. Plot KsAlpha on the original (left) axis
    ln1 = ax.plot(epochs, KsAlpha, label=r'$K_s/\alpha$', color='black', linewidth=1.2)
    ax.axhline(true_params[0], color='gray', linestyle='--')
    ax.set_xlabel('Epochs')
    ax.set_ylabel(r'$K_s/\alpha$ (cm$^2$/h)', color='black')
    ax.tick_params(axis='y', labelcolor='black')

    # 3. Create a twin axis for m (right side)
    ax_right = ax.twinx()
    ln2 = ax_right.plot(epochs, m, label=r'$m$', color='blue', linewidth=1.2)
    ax_right.axhline(true_params[1], color='blue', linestyle='--')
    ax_right.set_ylabel(r'$m$', color='blue')
    ax_right.tick_params(axis='y', labelcolor='blue')

    # 4. Handle the Legend
    # Since we have two axes, we need to combine the handles for one legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="center right", framealpha=0.85)

    ax.axvline(intercept, color='black', linestyle=':', linewidth=1)

    # 5. Panel Labeling and Grid
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if panel_label:
        ax.text(0.08, 0.9, panel_label, transform=ax.transAxes, 
                fontsize=13, fontweight='bold', va='top')
    
    # Optional: Set x-ticks to match your other plots (e.g., 50k intervals)
    ax.set_xlim(0, 200000)
    ax.set_xticks([50000, 100000, 150000, 200000])
    ax.set_xticklabels(['50k', '100k', '150k', '200k'])


def generate_master_figure(files_history, output_name="master_figure.eps"):

    # 1. Setup the Figure and GridSpec
    # 2 rows, 2 columns
    fig = plt.figure(figsize=(10, 8)) # Total size roughly 4x4 per panel
    gs = gridspec.GridSpec(2, 2, figure=fig)
    
    letters = [['a)', 'b)'], ['c)', 'd)']]

    tols = [5e-8, 5e-8]
    KsAlpha_true = 15.37 / 0.0432  # cm^2/h
    m_true = 0.5096
    true_params = [KsAlpha_true, m_true]

    # 2. Row 1: Losses
    for i, file in enumerate(files_history):
        ax = fig.add_subplot(gs[0, i])
        plot_losses(ax, file, tols[i], panel_label=letters[0][i])
        # Hide internal labels to save space if desired
        #if i > 0: ax.set_ylabel("") 

    # 3. Row 2: Lambda Phys
    for i, file in enumerate(files_history):
        ax = fig.add_subplot(gs[1, i])
        plot_parameters(ax, file, true_params, panel_label=letters[1][i])
        #if i > 0: ax.set_ylabel("")


    # 5. Global Adjustments
    plt.tight_layout()
    fig.savefig(output_name, format='eps', bbox_inches='tight')
    print(f"Saved master figure: {output_name}")



if __name__ == "__main__":

    # Use your existing file lists 
    histories = [
        './results/results_grenoblesand_gb_1pct_noise/history_trial226.parquet',
        './results/results_grenoblesand_gb_2pct_noise/history_trial236.parquet'
    ]

    generate_master_figure(histories, output_name="./figures/fig4.eps")
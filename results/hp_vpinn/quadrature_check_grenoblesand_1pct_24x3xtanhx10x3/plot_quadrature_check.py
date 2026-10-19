"""
plot_quadrature_check.py
========================
Figure for the SI quadrature convergence check (sec:quadrature_check).

Three side-by-side panels, one per training quadrature order Q_train. Each
panel shows the weak-form PDE loss of the trained, frozen models against the
evaluation order Q_eval:
  * Q_eval != Q_train : geometric mean over seeds with +/- 1 standard
                        deviation of log10(loss) as error bars (no lines);
  * Q_eval == Q_train : a large red dot (geometric mean over seeds).

Input: one JSON per Q_train holding a list of per-seed run records (as
written by quadrature_check_grenoblesand.py and merged), e.g.
    qtrain10.json  qtrain16.json  qtrain24.json

Usage
-----
    python plot_quadrature_check.py qtrain10.json qtrain16.json qtrain24.json
    python plot_quadrature_check.py qtrain*.json --out Figures/quadrature_check --no-latex
"""

import argparse
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HELD_OUT = "#2a78d6"     # held-out evaluation orders
TRAIN    = "#d62728"     # evaluation at the training order


def load(path):
    runs = json.load(open(path))
    runs = [r for r in runs if not r.get("failed")]
    q_train = {r["q_train"] for r in runs}
    assert len(q_train) == 1, f"{path}: mixed Q_train values {q_train}"
    return q_train.pop(), runs


def log_stats(runs, q):
    v = np.log10([r["eval_losses"][str(q)] for r in runs])
    return v.mean(), (v.std(ddof=1) if v.size > 1 else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--out", default="quadrature_check", help="Output path without extension.")
    p.add_argument("--no-latex", action="store_true")
    a = p.parse_args()

    use_tex = (not a.no_latex) and shutil.which("latex") is not None
    matplotlib.rcParams.update({
        "text.usetex": use_tex, "font.family": "serif",
        "font.serif": ["Computer Modern Roman"] if use_tex else ["DejaVu Serif"],
        "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8, "figure.dpi": 200,
    })

    data = sorted(load(f) for f in a.files)
    fig, axes = plt.subplots(1, len(data), figsize=(4.2 * len(data), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    letters = "abcdefgh"

    for k, (ax, (qt, runs)) in enumerate(zip(axes, data)):
        q_evals = sorted(int(q) for q in runs[0]["eval_losses"])
        held = [q for q in q_evals if q != qt]

        mu, sd = zip(*(log_stats(runs, q) for q in held))
        mu, sd = np.array(mu), np.array(sd)
        geo = 10 ** mu
        ax.errorbar(held, geo, yerr=[geo - 10 ** (mu - sd), 10 ** (mu + sd) - geo],
                    fmt="o", ms=4, color=HELD_OUT, ecolor=HELD_OUT, elinewidth=1,
                    capsize=2.5, lw=0, label=r"$Q_{\mathrm{eval}} \neq Q_{\mathrm{train}}$")

        mu_t, _ = log_stats(runs, qt)
        ax.plot(qt, 10 ** mu_t, "o", ms=11, color=TRAIN, mec="white", mew=1.2, zorder=5,
                label=r"$Q_{\mathrm{eval}} = Q_{\mathrm{train}}$")

        ax.set_yscale("log")
        ax.set_xlabel(r"$Q_{\mathrm{eval}}$")
        ax.set_title(rf"$Q_{{\mathrm{{train}}}} = {qt}$  ({len(runs)} seeds)", fontsize=10)
        ax.grid(True, which="major", alpha=0.3)
        ax.set_xlim(0, max(q_evals) + 2)
        ax.text(0.92, 0.92, f"{letters[k]})", transform=ax.transAxes,
                fontsize=12, fontweight="bold", fontfamily="sans-serif", va="top")
        if k == 0:
            ax.set_ylabel(r"$\mathcal{L}_{\mathrm{pde}}$ (frozen model)")
            ax.legend(loc="upper center", frameon=False)

    fig.tight_layout()
    fig.savefig(f"{a.out}.eps", format="eps", bbox_inches="tight")
    fig.savefig(f"{a.out}.png", dpi=200, bbox_inches="tight")
    print(f"[saved] {a.out}.eps, {a.out}.png")


if __name__ == "__main__":
    main()

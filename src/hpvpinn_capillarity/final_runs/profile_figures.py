"""Wetting-profile drawing shared by final_runs/common.py (one figure per final
run) and combined_profiles.py (2x2 figure of the four case studies).
NumPy + Matplotlib only.

Seed band (not drawn by default; the draw_* functions ignore the `band` argument): it is not a confidence
interval but the pointwise envelope of the trained profiles over the qualified fresh-seed runs, i.e. the
network-initialisation spread only (the data are fixed). Two envelopes:
    "minmax"  pointwise min--max over the seeds (default)
    "p5-95"   pointwise 5th--95th percentiles (robust to a single outlier)
It is drawn only if its largest width exceeds BAND_MIN_VISIBLE (in normalised theta units).
"""
import numpy as np

BAND_COLOR = "#c6dbef"        # opaque light blue (PostScript has no transparency)
BAND_MIN_VISIBLE = 5e-3       # normalised theta

# Same thresholds as hpo/postprocess.py (overridden by read_constant there)
ADMISSIBLE_PHYSICS = 1e-5
CALIBRATED_OBJ = 4.0

# Zoomed inset (axes-fraction rectangle of the inset, data limits of the zoom),
# per case; used where a close-up of the wetting front is informative.
ZOOM_BY_CASE = {
    # xlim/ylim: data window magnified. The inset keeps the on-screen aspect
    # ratio of that window (undistorted close-up) for the given height, down to
    # a minimum width (min_width); if the ratio would give a narrower inset it
    # is widened to min_width (slightly stretched). Placement in axes fractions:
    # right edge (right), bottom edge (bottom), height (height).
    "vangenuchten": dict(xlim=(0.75, 0.90), ylim=(-0.02, 0.90),
                         right=0.71, bottom=0.27, height=0.56, min_width=0.22),
}


def _inset_rect(ax, zoom):
    """Inset rectangle [x0, y0, w, h] in axes fractions (aspect-preserving)."""
    x0d, x1d = ax.get_xlim(); y0d, y1d = ax.get_ylim()
    (zx0, zx1), (zy0, zy1) = zoom["xlim"], zoom["ylim"]
    h = zoom["height"]
    # pixel aspect (w/h) of the window as seen in the main axes
    bb = ax.get_window_extent()
    win_ratio = ((zx1 - zx0) / (x1d - x0d) * bb.width) / ((zy1 - zy0) / (y1d - y0d) * bb.height)
    w = max(h * win_ratio * bb.height / bb.width, zoom["min_width"])
    return [zoom["right"] - w, zoom["bottom"], w, h]


def refit_zoom(ax):
    """Recompute the inset position after the final layout (call after
    fig.tight_layout()); no-op for axes without an inset."""
    ins = getattr(ax, "_zoom_inset", None)
    if ins is None:
        return
    from matplotlib.transforms import Bbox, TransformedBbox
    axins, zoom = ins
    rect = _inset_rect(ax, zoom)
    # locators return boxes in figure-fraction coordinates
    axins.set_axes_locator(
        lambda _a, _r: TransformedBbox(
            TransformedBbox(Bbox.from_bounds(*rect), ax.transAxes),
            ax.figure.transFigure.inverted()))


def is_qualified(rec: dict, adm: float = ADMISSIBLE_PHYSICS,
                 cal: float = CALIBRATED_OBJ):
    """(ok, physics_key) for one seed record: admissible on the held-out PDE
    loss (falls back to final_physics if absent) and calibrated (obj < cal)."""
    mt = rec.get("metrics") or {}
    key, phys = "pde_loss_eval", mt.get("pde_loss_eval")
    if phys is None or not np.isfinite(phys):
        key, phys = "final_physics", mt.get("final_physics")
    obj = rec.get("obj")
    ok = (phys is not None and np.isfinite(phys) and phys < adm
          and obj is not None and np.isfinite(obj) and obj < cal)
    return bool(ok), key


def band_from_pool(th_pool, kind: str = "minmax"):
    """Envelope dict(lo, hi, n, label) of an (n_seeds, n_phi) array, or None."""
    th_pool = np.asarray(th_pool)
    if th_pool.ndim != 2 or len(th_pool) < 2:
        return None
    if kind == "p5-95":
        lo, hi = np.percentile(th_pool, [5, 95], axis=0)
        label = "5--95\\%" if _usetex() else "5--95%"
    else:
        lo, hi = th_pool.min(0), th_pool.max(0)
        label = "min--max"
    return dict(lo=lo, hi=hi, n=int(len(th_pool)), label=label)


def _usetex() -> bool:
    import matplotlib
    return bool(matplotlib.rcParams["text.usetex"])


def _draw_content(ax, ref, rep_curve, band, schedule, noise_level,
                  labels=True, ms=1.0, curve_label=None):
    pct = r"\%" if _usetex() else "%"
    phi = rep_curve["phi_nd"]
    visible = False     # seed band no longer drawn (kept in the signature for compatibility)
    lab = (lambda s: s) if labels else (lambda s: None)
    ax.plot(phi, rep_curve["theta_nd"], "b-", linewidth=2.2, zorder=3,
            label=lab(curve_label or rf"$hp$-VPINN-{schedule}"))
    if "phi_clean_nd" in ref:      # drawn on top of the solution, thinner
        ax.plot(ref["phi_clean_nd"], ref["theta_clean_nd"], "k--", linewidth=1.0, zorder=4,
                label=lab("Noise-free solution"))
    if noise_level is None:
        tr_lab, te_lab = "Train", "Test"
    else:
        ns = f"{int(round(noise_level * 100))}{pct} noise"
        tr_lab, te_lab = f"Train ({ns})", f"Test (independent {ns})"
    ax.plot(ref["phi_train_nd"], ref["theta_train_nd"], "o", color="r", markersize=4.5 * ms,
            linestyle="none", zorder=5, label=lab(tr_lab))
    ax.plot(ref["phi_test_nd"], ref["theta_test_nd"], "o", markerfacecolor="none",
            markeredgecolor="r", markersize=5 * ms, linestyle="none", zorder=6,
            label=lab(te_lab))
    return visible


def draw_profile(ax, ref, rep_curve, band, schedule, noise_level, *, zoom=None,
                 legend=True, legend_fontsize=9, legend_loc="lower left",
                 panel_label=None, label_fontsize=12, curve_label=None):
    """Draw one wetting-profile panel on ax. zoom: dict(xlim, ylim, rect) adds
    a magnified inset of the region xlim x ylim (rect: inset position in axes
    fractions). panel_label (e.g. 'a)') is placed at the top right."""
    visible = _draw_content(ax, ref, rep_curve, band, schedule, noise_level,
                            curve_label=curve_label)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"$\tilde{\phi}$")
    ax.set_ylabel(r"$\tilde{\theta}$")
    ax.grid(True, alpha=0.4)
    if legend:
        ax.legend(loc=legend_loc, fontsize=legend_fontsize).set_zorder(10)  # above zoom connectors
    if zoom:
        ax.figure.canvas.draw()          # axes extent needed for the aspect ratio
        axins = ax.inset_axes(_inset_rect(ax, zoom))
        _draw_content(axins, ref, rep_curve, band, schedule, noise_level,
                      labels=False, ms=0.7, curve_label=curve_label)
        axins.set_xlim(*zoom["xlim"])
        axins.set_ylim(*zoom["ylim"])
        axins.grid(True, alpha=0.4)
        axins.set_xticks(np.arange(np.ceil(zoom['xlim'][0] / 0.05) * 0.05, zoom['xlim'][1] + 1e-9, 0.05))
        axins.tick_params(labelsize=6.5)
        ax.indicate_inset_zoom(axins, edgecolor="0.35", linewidth=0.8)
        ax._zoom_inset = (axins, zoom)
    if panel_label:
        ax.text(0.97, 0.96, panel_label, transform=ax.transAxes, fontsize=label_fontsize,
                ha="right", va="top")
    return visible

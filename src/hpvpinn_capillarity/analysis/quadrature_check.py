"""
quadrature_check_grenoblesand.py
================================
Quadrature convergence / aliasing check (SI, sec:quadrature_check) for the
hp-VPINN on Case 1 (Grenoble sand), using the CURRENT pipeline: the same
data preparation, model construction, training schedule and precision as
hparam_search_grenoblesand.py.

For each training quadrature order Q_train and each seed:
  1. train one hp-VPINN with the fixed architecture below (gb schedule,
     200k steps, same settings as the HPO);
  2. freeze it and evaluate the weak-form PDE loss at every Q_eval in
     Q_EVAL_LIST, plus at Q_train itself (training-order value);
  3. record the "honest floor" at Q_REF = max(Q_EVAL_LIST), which is never a
     training order, the ratio floor / training-order value, the strong-form
     residual RMS on a dense grid, chi2 (train/test) and parameter recovery.

Results are written as one JSON per (Q_train, seed) run, so runs can be
split across SLURM jobs and resumed; `--summarize` aggregates whatever runs
exist (geometric means across seeds, since losses span orders of magnitude).

Place at the repo root, next to hparam_search_grenoblesand.py.

Usage
-----
    # everything, serially (slow: see README note in the job commands)
    python quadrature_check_grenoblesand.py

    # one Q_train, all seeds (one SLURM job per Q_train)
    python quadrature_check_grenoblesand.py --q-train 16

    # smoke test (seconds): one run, few steps
    python quadrature_check_grenoblesand.py --q-train 10 --seeds 1 --epochs 2000 --out-dir /tmp/qc_smoke

    # aggregate existing runs
    python quadrature_check_grenoblesand.py --summarize
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import jax

import hparam_search_grenoblesand as hs   # constants + shared data prep + model builder

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Architecture under test (Stage A winner, Grenoble sand, gb, 1% noise)
ARCH = dict(nn_n_layers=2, nn_width=16, activation="tanh", n_elements=10, poly_degree=3)

NOISE_LEVEL = 0.01
WARMUP_EPOCHS, TRANSITION_EPOCHS = 0, 20_000          # gb schedule, as in the HPO

Q_TRAIN_LIST = [24]                            # 10 = production N_QUAD
Q_EVAL_LIST  = list(range(2, 49, 2))                   # 2, 4, ..., 40
Q_REF        = max(Q_EVAL_LIST)                        # honest-floor reference order

# Ten fixed, arbitrary seeds (drawn once; not tied to any HPO trial number)
SEEDS = [8113, 20459, 33127, 47701, 51389, 62903, 70117, 81611, 90073, 99991]

assert Q_REF not in Q_TRAIN_LIST, "Q_REF must never be a training order"

DEFAULT_OUT_DIR = hs.REPO_ROOT / "hp_vpinn" / "quadrature_check_grenoblesand_1pct"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weak_loss(trained_pinn, q):
    """Weak-form PDE loss of a frozen model evaluated with q Gauss points per element."""
    loss = hs.VPINNLoss(
        phi_min=0.0, phi_max=1.0,
        phi_ref=trained_pinn.pde.phi_ref, scaling=trained_pinn.pde.scaling,
        n_elements=ARCH["n_elements"], poly_degree=ARCH["poly_degree"],
        n_quad=q, element_clustering=1.0,
    )
    return float(loss(trained_pinn)[0])


def _run_path(out_dir, q_train, seed):
    return Path(out_dir) / f"qtrain{q_train}_seed{seed}.json"


# ---------------------------------------------------------------------------
# One (Q_train, seed) run
# ---------------------------------------------------------------------------

def run_one(q_train, seed, data, epochs, phi_dense):
    import optax

    pinn = hs._vpinn_build_pinn(
        ARCH["nn_n_layers"], ARCH["nn_width"], ARCH["activation"],
        data["phi_ref"], data["scaling"], seed,
    )
    lr_schedule = optax.exponential_decay(
        init_value=hs.LR_INIT, transition_steps=hs.LR_STEPS,
        decay_rate=hs.LR_END / hs.LR_INIT, end_value=hs.LR_END,
    )
    trainer = hs.VPINNTrainer(
        pinn               = pinn,
        measurements_train = data["measurements_train"],
        measurements_val   = None,                     # no early stopping, as in the HPO
        n_params           = hs.N_PARAMS,
        lambda_data        = 1.0,
        lambda_bc          = hs.LAMBDA_BC,
        lr_schedule_fn     = lr_schedule,
        n_elements         = ARCH["n_elements"],
        poly_degree        = ARCH["poly_degree"],
        n_quad             = q_train,
        use_lbfgs          = False,
    )

    t0 = time.perf_counter()
    trained, history = trainer.fit(
        epochs=epochs, warmup_epochs=WARMUP_EPOCHS, transition_epochs=TRANSITION_EPOCHS,
    )
    elapsed = time.perf_counter() - t0

    tail = min(10_000, epochs)
    final_physics = float(np.mean(history.physics_loss[-tail:]))
    final_data    = float(np.mean(history.data_loss[-tail:]))
    row = dict(q_train=q_train, seed=seed, epochs=epochs, elapsed_s=elapsed,
               precision=("float64" if jax.config.jax_enable_x64 else "float32"),
               noise_level=NOISE_LEVEL, **ARCH)

    if not (np.isfinite(final_physics) and np.isfinite(final_data)):
        row.update(failed=True)
        return row

    diff = trained.pde.diffusivity
    n_trainable = sum(1 for n in diff.__dataclass_fields__ if hasattr(getattr(diff, n), "hat"))
    recovered = {n: float(getattr(diff, n).value()) for n in hs.TRUE_PARAMS}
    errors = {f"err_{n}": abs(recovered[n] - hs.TRUE_PARAMS[n]) / abs(hs.TRUE_PARAMS[n])
              for n in hs.TRUE_PARAMS}

    eval_losses = {int(q): _weak_loss(trained, q) for q in Q_EVAL_LIST}
    train_order_loss = eval_losses.get(q_train, _weak_loss(trained, q_train))
    floor = eval_losses[Q_REF]

    row.update(
        failed=False,
        final_physics=final_physics,                     # training-order physics (incl. BC), last 10k mean
        chi2_nu_train=float(data["measurements_train"].chi2_nu(trained.nn_model, n_params=n_trainable)),
        chi2_nu_test=float(data["measurements_test"].chi2_nu(trained.nn_model, n_params=n_trainable)),
        pde_loss_train_order=train_order_loss,           # weak PDE loss at Q_eval = Q_train
        pde_loss_floor=floor,                            # weak PDE loss at Q_eval = Q_REF
        floor_over_train_order=(floor / train_order_loss) if train_order_loss > 0 else float("inf"),
        q_ref=Q_REF,
        eval_losses=eval_losses,
        strong_form_rms=float(hs.evaluate_strong_form_rms_vpinn(trained, phi_dense)),
        **recovered, **errors,
    )
    return row


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _geo(vals):
    v = np.asarray([x for x in vals if np.isfinite(x) and x > 0], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    lg = np.log10(v)
    return float(10 ** lg.mean()), float(lg.std(ddof=1)) if v.size > 1 else 0.0, int(v.size)


def summarize(out_dir):
    rows = [json.loads(p.read_text()) for p in sorted(Path(out_dir).glob("qtrain*_seed*.json"))]
    ok = [r for r in rows if not r.get("failed")]
    print(f"\n{len(rows)} runs found in {out_dir} ({len(rows) - len(ok)} failed)")
    if not ok:
        return

    q_trains = sorted({r["q_train"] for r in ok})
    q_evals  = sorted({int(q) for r in ok for q in r["eval_losses"]})

    # --- Per-Q_train summary ---
    print("\nPer Q_train (geometric mean across seeds; sd = std of log10):")
    hdr = (f"{'Q_train':>7} {'n':>3} {'L(Q_train)':>12} {'L(Q_ref)':>12} {'ratio':>10} "
           f"{'r_strong':>10} {'chi2_tr':>8} {'chi2_te':>8} {'errKs%':>13} {'errm%':>13}")
    print(hdr)
    summary = {}
    for qt in q_trains:
        sub = [r for r in ok if r["q_train"] == qt]
        g_self, _, n = _geo([r["pde_loss_train_order"] for r in sub])
        g_floor, sd_floor, _ = _geo([r["pde_loss_floor"] for r in sub])
        g_ratio, _, _ = _geo([r["floor_over_train_order"] for r in sub])
        g_strong, _, _ = _geo([r["strong_form_rms"] for r in sub])
        c_tr = np.mean([r["chi2_nu_train"] for r in sub])
        c_te = np.mean([r["chi2_nu_test"] for r in sub])
        eK = np.array([100 * r["err_KsAlpha"] for r in sub])
        em = np.array([100 * r["err_m"] for r in sub])
        print(f"{qt:7d} {n:3d} {g_self:12.3e} {g_floor:12.3e} {g_ratio:10.3g} "
              f"{g_strong:10.3e} {c_tr:8.3f} {c_te:8.3f} "
              f"{eK.mean():6.1f}±{eK.std(ddof=1) if len(eK) > 1 else 0:5.1f} "
              f"{em.mean():6.1f}±{em.std(ddof=1) if len(em) > 1 else 0:5.1f}")
        summary[qt] = dict(n=n, loss_train_order_geo=g_self, loss_floor_geo=g_floor,
                           loss_floor_log10_sd=sd_floor, ratio_geo=g_ratio,
                           strong_form_rms_geo=g_strong, chi2_train_mean=float(c_tr),
                           chi2_test_mean=float(c_te),
                           err_KsAlpha_pct_mean=float(eK.mean()), err_m_pct_mean=float(em.mean()))

    # --- Weak loss vs Q_eval, one column per Q_train (for the SI table/figure) ---
    print("\nWeak PDE loss vs Q_eval (geometric mean across seeds); * marks Q_eval = Q_train:")
    print(f"{'Q_eval':>7} " + " ".join(f"{'Qt=' + str(qt):>13}" for qt in q_trains))
    curve = {}
    for qe in q_evals:
        cells = []
        for qt in q_trains:
            g, _, _ = _geo([r["eval_losses"].get(str(qe), r["eval_losses"].get(qe, np.nan))
                            for r in ok if r["q_train"] == qt])
            curve.setdefault(qt, {})[qe] = g
            cells.append(f"{g:12.3e}{'*' if qe == qt else ' '}")
        print(f"{qe:7d} " + " ".join(cells))

    out = Path(out_dir) / "summary.json"
    out.write_text(json.dumps(dict(per_q_train=summary, loss_vs_q_eval=curve), indent=2))
    print(f"\n[saved] {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Quadrature convergence / aliasing check, Grenoble sand.")
    p.add_argument("--q-train", type=int, nargs="+", default=Q_TRAIN_LIST)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--epochs", type=int, default=hs.MAX_STEPS)
    p.add_argument("--precision", choices=["float64", "float32"], default="float64",
                   help="Must match the HPO runs being described (default: float64).")
    p.add_argument("--data", default=str(hs.REPO_ROOT / "data_grenoblesand.dat"))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--overwrite", action="store_true", help="Re-run even if the JSON exists.")
    p.add_argument("--summarize", action="store_true", help="Only aggregate existing runs.")
    a = p.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.summarize:
        summarize(out_dir)
        return

    bad = [q for q in a.q_train if q == Q_REF]
    if bad:
        raise SystemExit(f"Q_train={Q_REF} coincides with the reference order Q_REF.")

    # Precision must be fixed before the backend builds any array.
    jax.config.update("jax_enable_x64", a.precision == "float64")
    hs._load_vpinn_backend()
    import jax.numpy as jnp
    phi_dense = jnp.linspace(1e-4, 1.0, hs.N_GRID)

    shared = hs.prepare_shared_data(a.data, NOISE_LEVEL)   # same noise/split as the HPO
    data = hs._vpinn_prepare(shared)

    print(f"Architecture: {ARCH} | precision: {a.precision} | epochs: {a.epochs}")
    print(f"Q_train: {a.q_train} | Q_eval: {Q_EVAL_LIST[0]}..{Q_REF} | seeds: {a.seeds}")
    print(f"Output: {out_dir}")

    for qt in a.q_train:
        for seed in a.seeds:
            path = _run_path(out_dir, qt, seed)
            if path.exists() and not a.overwrite:
                print(f"[skip] {path.name} exists")
                continue
            print(f"\n=== Q_train={qt}  seed={seed} ===", flush=True)
            row = run_one(qt, seed, data, a.epochs, phi_dense)
            path.write_text(json.dumps(row, indent=2))
            if row.get("failed"):
                print("  diverged (non-finite losses)")
                continue
            print(f"  t={row['elapsed_s']:.0f}s  chi2 tr/te={row['chi2_nu_train']:.3f}/{row['chi2_nu_test']:.3f}"
                  f"  KsAlpha={row['KsAlpha']:.4g} ({100*row['err_KsAlpha']:.1f}%)"
                  f"  m={row['m']:.4f} ({100*row['err_m']:.1f}%)", flush=True)
            print(f"  L_pde(Q_train={qt})={row['pde_loss_train_order']:.3e}"
                  f"  L_pde(Q_ref={Q_REF})={row['pde_loss_floor']:.3e}"
                  f"  ratio={row['floor_over_train_order']:.3g}"
                  f"  r_strong={row['strong_form_rms']:.3e}", flush=True)

    summarize(out_dir)


if __name__ == "__main__":
    main()

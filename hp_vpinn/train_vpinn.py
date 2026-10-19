"""
train_vpinn.py
==============
Self-contained hp-VPINN trainer for the capillarity Boltzmann ODE.

Design principles
-----------------
* No dependency on train.py.  All training logic lives here.
* The loss function keeps a minimal signature.
* The JIT-compiled scan loop uses the same warm-up / transition / adaptive
  gradient-balancing structure as train_pinn.py.
* The spike-score and residual-cutoff logic from the strong-form PINN is
  intentionally omitted.
* Validation-based early stopping is preserved.
* An optional L-BFGS refinement stage follows the Adam stage.

History tuple returned by fit()
--------------------------------
Each entry is a 1-D array of length `epochs`:

    (total_loss, data_loss, physics_loss, val_loss, lambda_phys, diffusivity)

    index 0 : total loss
    index 1 : data loss
    index 2 : physics loss (weak-form PDE + BC)
    index 3 : validation data loss  (same as data_loss if no val set)
    index 4 : current value of lambda_phys used in the step
    index 5 : current diffusivity snapshot
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from jaxopt import LBFGS

from vpinn_loss import VPINNLoss

from typing import NamedTuple

class TrainingHistory(NamedTuple):
    total_loss: jax.Array
    data_loss: jax.Array
    physics_loss: jax.Array
    val_loss: jax.Array
    lambda_phys: jax.Array
    ema_physics: jax.Array
    ema_physics_frozen: jax.Array
    diffusivity: any
    pde_loss: jax.Array
    dirichlet_loss: jax.Array
    derivative_loss: jax.Array


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def make_loss_fn(static_model, n_params, lambda_data: float, lambda_bc: float, vpinn_loss: VPINNLoss):
    """
    Build and return the training loss function.

    The returned functions have signatures:

        data_loss_only(trainable_model, phi_m, theta_m, sigma_m)
        physics_loss_only(trainable_model)
        loss_fn(trainable_model, phi_m, theta_m, sigma_m, lambda_phys)
            -> (total_loss, (data_loss, physics_loss, all_A, all_B))
    """

    def data_loss_only(trainable_model, measurements):
        full_model = eqx.combine(trainable_model, static_model)
        return lambda_data * measurements.chi2_nu(full_model.nn_model, n_params=n_params)

    def physics_loss_only(trainable_model):
        full_model = eqx.combine(trainable_model, static_model)
        pde_loss, _, _, _ = vpinn_loss(full_model)

        # BC loss: enforce dtheta/dphi(1) = 0 (far-field condition)
        theta_1  = full_model.nn_model(1.0)
        dtheta_1 = jax.grad(full_model.nn_model)(1.0)
        dirichlet_loss = theta_1 ** 2
        derivative_loss = dtheta_1 ** 2
        bc_loss  = lambda_bc * (dirichlet_loss + derivative_loss)
        physics_loss = pde_loss + bc_loss
        return physics_loss, (pde_loss, dirichlet_loss, derivative_loss)

    def loss_fn(trainable_model, measurements, lambda_phys):
        full_model = eqx.combine(trainable_model, static_model)

        # ---- Data loss (sigma-weighted MSE = chi^2 / N) ----
        data_loss = measurements.chi2_nu(full_model.nn_model, n_params=n_params)

        # ---- Weak-form physics loss ----
        pde_loss, all_R, all_A, all_B = vpinn_loss(full_model)

        # ---- BC loss: enforce dtheta/dphi(1) = 0 ----
        theta_1  = full_model.nn_model(1.0)
        dtheta_1 = jax.grad(full_model.nn_model)(1.0)
        bc_loss  = lambda_bc * (theta_1 ** 2 + dtheta_1 ** 2)
        physics_loss = pde_loss + bc_loss
        total_loss = lambda_data * data_loss + lambda_phys * physics_loss

        return total_loss, (data_loss, physics_loss, all_R, all_A, all_B)

    return loss_fn, data_loss_only, physics_loss_only


# ---------------------------------------------------------------------------
# JIT-compiled Adam training loop
# ---------------------------------------------------------------------------

@eqx.filter_jit
def _train_vpinn(
    trainable_model,
    static_model,
    optimizer,
    opt_state,
    loss_fn,
    data_loss_only,
    physics_loss_only,
    lambda_data,
    lambda_bc,
    measurements_train,
    measurements_val,
    lambda_phys_arr,
    epochs,
    validate_every,
    patience,
    min_delta,
    warmup_epochs: int = 0,
    transition_epochs: int = 20_000,
    ema_beta: float = 0.9,
    lambda_min: float = 1e-6,
    lambda_max: float = 1e6,
    has_validation: bool = True,   # static — controls which branch is compiled
):
    def compute_val_loss(trainable_model):
        full_model = eqx.combine(trainable_model, static_model)
        val_loss = measurements_val.chi2_nu(full_model.nn_model, n_params=4)
        return val_loss

    def tree_abs_stats(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        flat = jnp.concatenate([jnp.abs(x).reshape(-1) for x in leaves])
        return jnp.max(flat), jnp.mean(flat)

    def step_fn(carry, step):
        (
            trainable_model,
            opt_state,
            active,
            best_val_loss,
            best_model,
            patience_counter,
            lambda_phys_state,
            ema_physics,
            ema_physics_frozen,
            phys_pat_counter
        ) = carry

        # Warm-up schedule value for this epoch
        lambda_phys_sched = lambda_phys_arr[step]

        # ------------------------------------------------------------
        # 1) Compute gradient statistics for adaptive balancing
        # ------------------------------------------------------------
        data_loss, data_grads = jax.value_and_grad(data_loss_only)(
            trainable_model, measurements_train
        )

        (physics_loss, (pde_loss_val, dirichlet_loss_val, derivative_loss_val)), phys_grads = (
            jax.value_and_grad(physics_loss_only, has_aux=True)(trainable_model)
        )

        physics_threshold  = 5e-8   # same threshold used in the gate criterion
        phys_patience      = 10_000  # epochs EMA must stay below threshold to confirm

        # 1. Candidate EMA update (computed regardless of frozen state)
        ema_physics_candidate = (
            (ema_beta+0.09) * ema_physics
            + (1.0 - (ema_beta+0.09)) * physics_loss
        )

        # 2. Use candidate only if not yet frozen; otherwise hold the frozen value
        ema_physics = jax.lax.select(
            ema_physics_frozen,
            ema_physics,              # frozen: do not update
            ema_physics_candidate,    # active: update normally
        )

        # 3. Patience counter: increments while EMA < threshold, resets otherwise
        #    Also freezes at zero once frozen (no point counting further)
        phys_pat_counter = jax.lax.select(
            ema_physics_frozen,
            phys_pat_counter,         # frozen: hold value
            jax.lax.select(
                ema_physics < physics_threshold,
                phys_pat_counter + 1, # below threshold: increment
                jnp.array(0),         # above threshold: reset
            ),
        )

        # 4. Freeze once patience is exceeded while below threshold
        #    Once True it stays True (latching behaviour)
        ema_physics_frozen = (
            ema_physics_frozen
            | ((ema_physics < physics_threshold) & (phys_pat_counter >= phys_patience))
        )

        max_grad_data, _ = tree_abs_stats(data_grads)
        _, mean_grad_phys = tree_abs_stats(phys_grads)

        lambda_phys_target = lambda_data * max_grad_data / (mean_grad_phys + 1e-8)
        lambda_phys_target = jnp.clip(lambda_phys_target, lambda_min, lambda_max)


        # ------------------------------------------------------------
        # 2) Warm-up / transition / adaptive use
        # ------------------------------------------------------------
        warmup_epochs_j = jnp.asarray(warmup_epochs)
        transition_epochs_j = jnp.asarray(jnp.maximum(transition_epochs, 1))

        in_warmup = step < warmup_epochs_j
        in_transition = (
            (step >= warmup_epochs_j)
            & (step < (warmup_epochs_j + transition_epochs_j))
        )

        # EMA update of the adaptive state
        lambda_phys_state = (
            ema_beta * lambda_phys_state
            + (1.0 - ema_beta) * lambda_phys_target
        )

        blend = jnp.clip(
            (step - warmup_epochs_j) / transition_epochs_j,
            0.0,
            1.0,
        )

        lambda_phys_used = jnp.where(
            in_warmup,
            lambda_phys_sched,
            jnp.where(
                in_transition,
                (1.0 - blend) * lambda_phys_sched + blend * lambda_phys_state,
                lambda_phys_state,
            ),
        )

        # ------------------------------------------------------------
        # 3) Loss and gradients
        # ------------------------------------------------------------
        loss = data_loss + lambda_phys_used * physics_loss
        grads = jax.tree_util.tree_map(
            lambda gd, gp: gd + lambda_phys_used * gp,
            data_grads,
            phys_grads,
        )

        # Mask gradients if inactive
        grads = jax.tree_util.tree_map(
            lambda g: jnp.where(active, g, jnp.zeros_like(g)),
            grads,
        )

        # ------------------------------------------------------------
        # 4) Validation and early stopping
        # ------------------------------------------------------------
        if has_validation:
            do_validate = (step % validate_every) == 0
            val_loss = jax.lax.cond(
                do_validate,
                lambda _: compute_val_loss(trainable_model),
                lambda _: best_val_loss,
                operand=None,
            )
            improved = val_loss < (best_val_loss - min_delta)
        else:
            val_loss = jnp.array(jnp.inf)
            improved = jnp.array(True)

        best_val_loss = jax.lax.select(improved, val_loss, best_val_loss)
        best_model = jax.tree_util.tree_map(
            lambda new, old: jnp.where(improved, new, old),
            trainable_model, best_model,
        )

        if has_validation:
            patience_counter = jax.lax.select(
                improved,
                jnp.array(0),
                patience_counter + do_validate.astype(jnp.int32),
            )
            active = active & (patience_counter < patience)

        # ------------------------------------------------------------
        # 5) History
        # ------------------------------------------------------------
        history = TrainingHistory(
            total_loss=loss,
            data_loss=data_loss,
            physics_loss=physics_loss,
            val_loss=val_loss,
            lambda_phys=lambda_phys_used,
            ema_physics=ema_physics,
            ema_physics_frozen=ema_physics_frozen,
            diffusivity=trainable_model.pde.diffusivity,
            pde_loss=pde_loss_val,
            dirichlet_loss=dirichlet_loss_val,
            derivative_loss=derivative_loss_val,
        )
        
        # ------------------------------------------------------------
        # 6) Optimizer update
        # ------------------------------------------------------------
        updates, opt_state = optimizer.update(grads, opt_state, trainable_model)
        trainable_model = eqx.apply_updates(trainable_model, updates)

        return (
            trainable_model,
            opt_state,
            active,
            best_val_loss,
            best_model,
            patience_counter,
            lambda_phys_state,
            ema_physics,
            ema_physics_frozen,
            phys_pat_counter
        ), history

    lambda_phys_state_init = lambda_phys_arr[jnp.minimum(warmup_epochs, epochs - 1)]
    init_carry = (
        trainable_model,
        opt_state,
        jnp.array(True),
        jnp.array(jnp.inf),
        trainable_model,
        jnp.array(0),
        lambda_phys_state_init,
        jnp.array(0.0),
        jnp.array(False),
        jnp.array(0),
    )

    (final_model, _, _, _, best_model, _, _, _, _, _), history = jax.lax.scan(
        step_fn,
        init_carry,
        jnp.arange(epochs),
    )

    trained_model = jax.lax.cond(
        jnp.asarray(validate_every) <= jnp.asarray(epochs),
        lambda: best_model,
        lambda: final_model,
    )

    return trained_model, history


# ---------------------------------------------------------------------------
# JIT-compiled L-BFGS refinement stage
# ---------------------------------------------------------------------------
@eqx.filter_jit
def _train_lbfgs(
    trainable_model,
    static_model,
    loss_fn,
    measurements,
    lambda_phys,
    maxiter: int = 500,
):
    """
    L-BFGS refinement starting from the result of the Adam stage.
 
    Drives solver.update manually inside a jax.lax.scan loop so that
    per-step loss components are recorded.
 
    Returns
    -------
    final_params : trainable pytree
    lbfgs_history : tuple
        (total_loss, data_loss, physics_loss, diffusivity)
        Each entry has a leading axis of length `maxiter`.
    """
 
    # objective must be a pure function of params (closure over the rest)
    def objective(model):
        total_loss, _ = loss_fn(model, measurements, lambda_phys)
        return total_loss
 
    # maxiter=1 so each solver.update executes exactly one quasi-Newton step
    solver = LBFGS(fun=objective, maxiter=1, tol=1e-8, history_size=20)
    init_state = solver.init_state(trainable_model)
 
    def step(carry, _):
        params, state = carry
        params, state = solver.update(params, state)
 
        # one extra forward pass to decompose the loss
        total, (data_loss, physics_loss, _, _) = loss_fn(
            params, measurements, lambda_phys
        )
        full_model = eqx.combine(params, static_model)
        snapshot = (total, data_loss, physics_loss, full_model.pde.diffusivity)
        return (params, state), snapshot
 
    (final_params, _), lbfgs_history = jax.lax.scan(
        step, (trainable_model, init_state), None, length=maxiter
    )
 
    return final_params, lbfgs_history


# --------------------------------------------------1.0,-------------------------
# VPINNTrainer
# ---------------------------------------------------------------------------

class VPINNTrainer:
    """
    Trainer for the hp-VPINN formulation of the capillarity Boltzmann ODE.

    Responsibilities
    ----------------
    1. Build the VPINNLoss quadrature grid once at construction time.
    2. Assemble the loss function and optimizer.
    3. Run the Adam training loop via _train_vpinn.
    4. Optionally run an L-BFGS refinement stage.
    5. Return the trained PINN and the loss history.
    """

    def __init__(
        self,
        pinn,
        measurements_train,
        measurements_val,
        n_params,
        lambda_data: float = 1.0,
        lambda_bc: float = 1.0,
        lr_schedule_fn=1e-3,
        # hp-VPINN grid
        n_elements: int = 20,
        poly_degree: int = 3,
        n_quad: int = 8,
        element_clustering: float = 1.0,
        # optional stages
        use_lbfgs: bool = False,
        lbfgs_maxiter: int = 500,
        l2_reg: bool = False,
        weight_decay: float = 1e-3,
    ):
        self.pinn = pinn
        self.measurements_train = measurements_train
        self.measurements_val = measurements_val
        self.n_params = n_params
        self.lambda_data = lambda_data
        self.lambda_bc = lambda_bc
        self.use_lbfgs = use_lbfgs
        self.lbfgs_maxiter = lbfgs_maxiter

        # ---- Partition model into trainable and static leaves ----
        self.trainable_model, self.static_model = eqx.partition(
            pinn, eqx.is_array
        )

        # ---- Optimizer ----
        if l2_reg:
            # Weight decay on network weights only; exclude diffusivity parameters from L2 regularization.
            mask = jax.tree_util.tree_map(lambda _: True, self.trainable_model)
            diff = self.trainable_model.pde.diffusivity
            for field_name in diff.__dataclass_fields__:
                param = getattr(diff, field_name)
                if hasattr(param, "hat"):
                    mask = eqx.tree_at(
                        lambda t, fn=field_name: getattr(t.pde.diffusivity, fn).hat,
                        mask,
                        replace=False,
                    )
            self.optimizer = optax.adamw(
                learning_rate=lr_schedule_fn, weight_decay=weight_decay, mask=mask
            )
        else:
            self.optimizer = optax.adam(learning_rate=lr_schedule_fn)

        # ---- Build hp-VPINN quadrature grid ----
        self.vpinn_loss = VPINNLoss(
            phi_min=0.0,
            phi_max=1.0,
            phi_ref=pinn.pde.phi_ref,
            scaling=pinn.pde.scaling,
            n_elements=n_elements,
            poly_degree=poly_degree,
            n_quad=n_quad,
            element_clustering=element_clustering,
        )

        Ne = n_elements
        p = poly_degree
        nq = n_quad
        print(
            f"[VPINNTrainer] grid: {Ne} elements × {p} test fns × {nq} quad pts"
            f"  →  {Ne * p} weak residuals,  {Ne * nq} network evaluations per step"
        )

    # ------------------------------------------------------------------

    def fit(
        self,
        epochs: int,
        validate_every: int = 1_000,
        patience: int = 50,
        min_delta: float = 1e-6,
        warmup_epochs: int = 0,  #20_000
        transition_epochs: int = 20_000,  #10_000
        ema_beta: float = 0.9,
        lambda_min: float = 1e-6,
        lambda_max: float = 1e9,
    ):
        """
        Run the Adam training loop (and optional L-BFGS refinement).
        """
        
        # ---- Validation data ----
        use_validation = self.measurements_val is not None
        if use_validation:
            measurements_val = self.measurements_val
            has_validation   = True
        else:
            measurements_val = self.measurements_train  # never evaluated, just needs valid structure
            has_validation   = False
            validate_every   = epochs + 1
            patience         = epochs + 1
            min_delta        = 0.0

        # ---- Initial data loss (used to set the scale of lambda_phys) ----
        full_model = eqx.combine(self.trainable_model, self.static_model)
        L0_data = self.measurements_train.chi2_nu(full_model.nn_model, n_params=self.n_params)

        # ---- Warm-up schedule for lambda_phys ----
        steps = jnp.arange(epochs)
        lambda_phys_arr = self.lambda_data * L0_data * 10 ** (-2 + steps / 100_000)

        # ---- Assemble loss function ----
        loss_fn, data_loss_only, physics_loss_only = make_loss_fn(
            self.static_model,
            self.n_params,
            self.lambda_data,
            self.lambda_bc,
            self.vpinn_loss,
        )

        opt_state = self.optimizer.init(self.trainable_model)

        # ---- Adam loop ----
        trained_model, history = _train_vpinn(
            self.trainable_model,
            self.static_model,
            self.optimizer,
            opt_state,
            loss_fn,
            data_loss_only,
            physics_loss_only,
            self.lambda_data,
            self.lambda_bc,
            self.measurements_train, 
            measurements_val, 
            lambda_phys_arr,
            epochs,
            validate_every,
            patience,
            min_delta,
            warmup_epochs=warmup_epochs,
            transition_epochs=transition_epochs,
            ema_beta=ema_beta,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            has_validation=has_validation,
        )

        # ---- Optional L-BFGS stage ----
        if self.use_lbfgs:
            def _print_status(label, model):
                lp = float(history[4][-1])
                total, (data, phys, _, _) = loss_fn(
                    model, self.measurements_train, lp
                )
                full = eqx.combine(model, self.static_model)
                diff = full.pde.diffusivity
                param_strs = "  ".join(
                    f"{name}={float(getattr(diff, name).value()):.4g}"
                    for name in diff.__dataclass_fields__
                    if hasattr(getattr(diff, name), "value")
                )
                print(
                    f"[L-BFGS {label}]"
                    f"  total={float(total):.4g}"
                    f"  data={float(data):.4g}"
                    f"  phys={float(phys):.4g}"
                    f"  |  {param_strs}"
                )
 
            _print_status("before", trained_model)
 
            lp = history[4][-1]   # lambda_phys frozen at the last Adam value
            trained_model, lbfgs_raw = _train_lbfgs(
                trained_model,
                self.static_model,
                loss_fn,
                self.measurements_train,
                lp,
                maxiter=self.lbfgs_maxiter,
            )
            _print_status("after ", trained_model)
 
            # lbfgs_raw = (total, data, phys, diffusivity)  each (maxiter, ...)
            # Pad to match the 6-tuple structure of the Adam history so the two
            # histories can be concatenated transparently:
            #   index 0 total_loss | 1 data_loss | 2 physics_loss
            #   index 3 val_loss   | 4 lambda_phys | 5 diffusivity
            last_val = history[3][-1]
            lbfgs_padded = (
                lbfgs_raw[0],                                       # total
                lbfgs_raw[1],                                       # data
                lbfgs_raw[2],                                       # phys
                jnp.full(self.lbfgs_maxiter, last_val),             # val (frozen)
                jnp.full(self.lbfgs_maxiter, lp),                   # lambda (frozen)
                lbfgs_raw[3],                                       # diffusivity
            )
 
            # Concatenate along the leading (epoch) axis for every leaf
            history = jax.tree_util.tree_map(
                lambda a, b: jnp.concatenate([a, b]),
                history,
                lbfgs_padded,
            )

        full_trained_pinn = eqx.combine(trained_model, self.static_model)
        #_, all_R, _, _ = self.vpinn_loss(full_trained_pinn)
        #element_centroids = jnp.mean(self.vpinn_loss.quad_pts, axis=1)  # shape (Ne,)
        #residual_rms = jnp.sqrt(jnp.mean(all_R**2, axis=1))
        return full_trained_pinn, history

# Applying the framework to a new dataset

The recommendations below summarise the protocol of the manuscript and the
evidence in its Discussion and Supplementary Information (SI). They describe
what was done and observed for the four cases; the thresholds were fixed
empirically and their transfer to other data was not tested. The simplest
route is to copy `hpo/search_<case>.py` and `final_runs/<case>.py` of the
closest case and adapt the data, the constants and the model.

**Scope.** The framework applies to one-dimensional horizontal flow with
constant boundary and initial conditions, where the Boltzmann transformation
reduces the problem to an ODE in the similarity variable. It does not apply
directly to gravity-driven or multidimensional flow, time-dependent boundary
conditions, or heterogeneous media. The data loss assumes independent
measurement errors with known standard deviations.

**1. Choose the constitutive model.**
`constitutive_model.py` provides van Genuchten, LETd and Brooks and Corey
diffusivities, with parameters declared as fixed, free or bounded.
- Compare candidate models with an exact forward solver on the same data and
  split, where one is available (manuscript, "Model adequacy"). A misfit above
  the noise level that is independent of the split and of the network points to
  an inadequate model: in Case 2 the van Genuchten fit stays above it
  ($\chi^2_\nu$ about 1.6-2.1), whereas LETd (Case 3) reaches about 0.9-1.0.
- Do not read a low $\chi^2_\nu$ as evidence of a better model. The `-es`
  solutions of Case 2 fit the data better but, inserted into the exact solver,
  do not satisfy the equation for the parameters they report.
- Parameters that reproduce the data equally well can differ strongly. Compare
  the recovered $D(\theta)$ and not only the parameters; they differ most
  where the data carry little information, at the dry end.

**2. Choose the parameter ranges and the initial values.**
- The initial values are fixed in the search drivers, so the ranking of
  architectures is conditional on them. Run several starting points
  (`analysis/multistart.py`). In Case 1, about 40-50 % of the starts did not
  converge, and the non-convergent runs are flagged by `obj`.
- Bounds that exclude the true value are not detected by the fit: qualified
  runs ended on or next to the bound with errors of 12-240 % in a parameter.
  Treat an estimate within a few per cent of a bound as not determined and
  relax the bound.

**3. Choose the architecture and the hyperparameters.**
- Search space of the paper (960 combinations): hidden layers {1, 2, 3}, width
  {8, 12, 16, 24, 32}, activation {tanh, sigmoid, GeLU, Swish}, elements
  {10, 20, 30, 40}, polynomial degree {3, 4, 5, 6}; quadrature order 24 and
  $\lambda_{bc}=10$ fixed. Small networks were preferred.
- Optuna TPE sampler (multivariate, grouped, 25 initial random trials),
  250 trials, 200 000 steps per trial without early stopping, 20 % of the points
  held out. The objective is $\chi^2_{\nu,\mathrm{train}} + \chi^2_{\nu,\mathrm{test}}$.
- A trial is *admissible* if the weak-form residual re-evaluated at a held-out
  quadrature order ($Q_{\mathrm{eval}}=40$) is below $10^{-5}$, *calibrated* if
  `obj` < 4, and *qualified* if both hold. The winner is the combination with
  the most qualified runs, not the single best trial.
- The physics loss at the training nodes can underestimate its value at held-out
  nodes (a factor of about 30 at $Q=24$ in Case 1), so judge enforcement on the
  held-out evaluation and on the strong-form residual
  (`core/strong_form_residual.py`).

**4. Interpret the convergence.**
- In Case 1 the data loss falls below the nominal noise level within about
  1.1-1.4 x 10^3 steps, when the trainable parameters are still at their initial
  values; the parameter error stays near 100 % until 3-6 x 10^3 steps. The
  parameters are determined only once the physics loss starts to act.
- Physics loss is oscillatory. The early-stopping indicator of the paper is its
  exponential moving average ($\beta_s = 0.99$, $\tau_{stop} = 5\times10^{-8}$,
  window $10^4$). It worked within the tested range in Case 1 but not in Case 3.
- A single run is not sufficient. Use several seeds and starting points, the
  acceptance criteria above, and a forward check with an exact solver where one
  exists. A calibrated fit does not certify the parameters: in Case 1 the data
  constrain a ridge in the $(K_s/\alpha, m)$ plane and not a point.

"""
Covariance hyperparameter optimisation via scipy.optimize.

Overview
--------
Covariance hyperparameters θ = (σ², a, …) are optimised in log-space to
enforce positivity constraints:  log θ ∈ ℝ  (unconstrained).  At each
candidate log θ, the model is refitted via EP and the chosen evidence
objective is evaluated.

Evidence objectives
-------------------
Two objectives are supported:

  'ep'  (default)
      F(θ) = Σ_i log z_i(θ)   — sum of EP normalisation constants.
      This is the EP lower bound on the log-marginal likelihood.
      Works well for non-Gaussian likelihoods (Exponential, Student-t, etc.)
      because it directly measures how well the approximate posterior explains
      the data.

  'active_set'
      F(θ) = active_set_log_evidence(θ)   — the full MATLAB ogpevid formula.
      Uses the complete active-set evidence including log|I + K_B C|,
      quadratic, and EP correction terms.  Recommended for Gaussian likelihood
      where it is the exact log-marginal likelihood up to the sparse approximation.

Gradient modes
--------------
  'analytic'  Only available with evidence='active_set'.  Computes
              ∂F/∂ log θ analytically via the trace formula in
              VRK.analytic_evidence_gradient() (see vrk.py).  Fast and
              accurate; preferred with L-BFGS-B.

  'fd'        Finite-difference gradient, always available.  Perturbs each
              log-parameter by eps=1e-3 and recomputes the evidence.  Slower
              by a factor of n_params + 1 per gradient call but does not
              require analytic gradient implementation.

  None        Auto-selects: analytic for active_set + L-BFGS-B, fd otherwise.

Optimisation strategy
---------------------
The function runs n_restarts optimisation attempts:
  - Restart 0: starts from the initial (user-supplied) log-parameters.
  - Restart k > 0: random perturbation of the initial parameters (unit Gaussian).

Each attempt is clipped to the specified bounds in log-space.  The best result
(highest evidence) is retained and the model is refitted at those parameters.
"""
import numpy as np
from scipy.optimize import minimize


def optimise_hyperparameters(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_restarts: int = 1,
    method: str = "L-BFGS-B",
    evidence: str = "ep",
    bounds: list | None = None,
    maxiter: int = 100,
    gradient: str | None = None,
) -> float:
    """
    Optimise covariance hyperparameters by maximising the log evidence.

    Parameters
    ----------
    model      : VRK instance (modified in-place)
    X          : (n, d) training locations
    y          : (n,) observations
    n_restarts : int
        Number of optimisation restarts (restart 0 uses the current params).
    method     : str
        scipy.optimize.minimize method.  'L-BFGS-B' (default, gradient-based,
        supports bounds) or 'Nelder-Mead' (gradient-free, slower but robust).
    evidence   : str
        Objective function: 'ep' (sum of log Z_i, recommended for non-Gaussian
        likelihoods) or 'active_set' (ogpevid formula, recommended for Gaussian).
    bounds     : list of (lo, hi) pairs in log-space, one per hyperparameter.
        Defaults to [(-5, 5)] per parameter, i.e. parameter in [exp(-5), exp(5)].
    maxiter    : int
        Maximum optimiser iterations per restart.
    gradient   : str or None
        'analytic' — use VRK.analytic_evidence_gradient() (active_set only).
        'fd'       — finite-difference gradient (always available).
        None       — auto-select.

    Returns
    -------
    best_evidence : float
        Best evidence value achieved across all restarts.
        The model covariance is set to the corresponding optimal parameters.
    """
    covariance = model.covariance
    init_log_params = covariance.log_params.copy()
    n_params = len(init_log_params)

    # Select the evidence function
    if evidence == "active_set":
        def _ev():
            return model.active_set_log_evidence()
    else:
        def _ev():
            return model.approximate_evidence()

    # Select gradient mode
    if gradient is None:
        use_analytic = (evidence == "active_set" and method == "L-BFGS-B")
    else:
        use_analytic = (gradient == "analytic")

    if use_analytic and evidence != "active_set":
        raise ValueError(
            "analytic gradient is only available for evidence='active_set'."
        )

    default_bounds = [(-5.0, 5.0)] * n_params
    if bounds is None:
        bounds = default_bounds

    # Objective functions passed to scipy.optimize.minimize  (minimise −F)

    def neg_evidence_and_analytic_grad(log_params):
        """Returns (−F, −∇F) using the analytic gradient from vrk.py."""
        covariance.log_params = log_params
        model.fit(X, y)
        ev = _ev()
        grad = model.analytic_evidence_gradient()
        return -ev, -grad

    def neg_evidence_and_fd_grad(log_params):
        """Returns (−F, −∇F) using finite-difference gradient."""
        covariance.log_params = log_params
        model.fit(X, y)
        ev = _ev()
        if evidence == "active_set":
            grad = model.active_set_evidence_gradient()
        else:
            grad = model.evidence_gradient()
        return -ev, -grad

    def neg_evidence(log_params):
        """Returns −F only (for gradient-free or internal-FD methods)."""
        covariance.log_params = log_params
        model.fit(X, y)
        return -_ev()

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # Evaluate the initial parameters as baseline
    covariance.log_params = init_log_params
    model.fit(X, y)
    best_ev = _ev()
    best_params = init_log_params.copy()

    for restart in range(n_restarts):
        if restart == 0:
            x0 = np.clip(init_log_params, lo, hi)
        else:
            # Random perturbation around the initial params, clipped to bounds
            x0 = init_log_params + np.random.randn(n_params)
            x0 = np.clip(x0, lo, hi)

        try:
            if method == "L-BFGS-B":
                if use_analytic:
                    res = minimize(
                        neg_evidence_and_analytic_grad,
                        x0,
                        jac=True,
                        method="L-BFGS-B",
                        bounds=bounds,
                        options={"maxiter": maxiter, "ftol": 1e-6},
                    )
                elif evidence == "active_set":
                    res = minimize(
                        neg_evidence,
                        x0,
                        method="L-BFGS-B",
                        jac="2-point",
                        bounds=bounds,
                        options={"maxiter": maxiter, "ftol": 1e-6, "eps": 1e-3},
                    )
                else:
                    res = minimize(
                        neg_evidence_and_fd_grad,
                        x0,
                        jac=True,
                        method="L-BFGS-B",
                        bounds=bounds,
                        options={"maxiter": maxiter, "ftol": 1e-6},
                    )
            else:
                res = minimize(
                    neg_evidence,
                    x0,
                    method=method,
                    options={"maxiter": 300, "xatol": 1e-2, "fatol": 1e-3},
                )
            ev = -res.fun
            if ev > best_ev:
                best_ev = ev
                best_params = res.x.copy()
        except Exception:
            pass

    # Restore the model at the best parameters found
    covariance.log_params = best_params
    model.fit(X, y)
    return best_ev

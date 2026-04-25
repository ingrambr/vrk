"""
Alternative EP update methods for the one-sided exponential likelihood.

All classes implement the same Likelihood interface as ExponentialNoiseLikelihood
and can be dropped in as direct replacements.  They are provided for comparison
with the exact analytical formula.

Methods
-------
ExponentialNoiseGLLikelihood
    Gauss-Laguerre quadrature.  Uses the substitution f = y - t/λ to map the
    truncated integral to ∫₀^∞ e^{-t} g(t) dt, which is the natural Laguerre
    weight.  All nodes satisfy f_k ≤ y by construction — truncation is not an
    issue.

ExponentialNoiseLegendrelikelihood
    Gauss-Legendre quadrature on the explicit finite interval [a, y].  The upper
    bound y encodes the truncation directly; the lower bound a is chosen to cover
    effectively all Gaussian cavity mass.

ExponentialNoiseModeGHLikelihood
    Mode-adapted Gauss-Hermite.  Standard GH nodes are centred on the mode
    f* = min(μ̃ + λσ̃², y) rather than on the cavity mean μ̃.  When z > 0 (cavity
    above observation), f* = y puts half the nodes on each side of the truncation
    boundary, dramatically improving coverage in the tail regime.

ExponentialNoiseLaplaceLikelihood
    Laplace approximation.  Finds the mode f* and approximates the tilted
    distribution as a Gaussian at f*.  For the exponential likelihood, log p(y|f)
    is linear in f, so the Hessian is zero and the Laplace approximation gives an
    uninformative site (zero precision).  Included to demonstrate why Laplace
    fails for linear log-likelihoods.

ExponentialNoiseTruncISLikelihood
    Importance sampling with a truncated-Gaussian proposal q(f) = N(f; μ̃, σ̃²)
    truncated to f ≤ y.  All samples are in the valid region; weights are
    w_m ∝ exp(−λ(y − f_m)).  Unlike the plain-Gaussian IS (which has the same
    node-placement problem as standard GH), this converges everywhere.
"""
import numpy as np
from scipy.special import erfc, ndtr   # ndtr(x) = Φ(x)
from scipy.stats import truncnorm
from vrk.likelihoods.base import Likelihood

_LAMBDA_TOL = 1e-10
_PREC_CAP   = 100.0    # same precision cap as the analytical implementation


# ---------------------------------------------------------------------------
# Shared moment-matching helper
# ---------------------------------------------------------------------------

def _moment_match(
    mean_t: float,
    var_t:  float,
    cavity_mean: float,
    cavity_var:  float,
    log_z: float,
) -> tuple[float, float, float]:
    """
    Convert tilted-distribution moments (mean_t, var_t) to EP site natural
    parameters (q, r) following the StudentTLikelihood convention.

    Returns (0, −ε, log_z) whenever the moment-matching is degenerate (e.g.
    var_t ≥ cavity_var, meaning the likelihood provides no information).
    """
    denom = cavity_var - var_t
    if abs(denom) < _LAMBDA_TOL:
        return 0.0, -_LAMBDA_TOL, log_z

    var_site = cavity_var * var_t / denom
    if var_site <= 0.0:
        return 0.0, -_LAMBDA_TOL, log_z

    mean_site = (
        mean_t * (cavity_var + var_site) - cavity_mean * var_site
    ) / cavity_var

    r = -1.0 / (var_site + cavity_var)
    q = r * (cavity_mean - mean_site)

    if r >= 0.0:
        return 0.0, -_LAMBDA_TOL, log_z

    # Precision cap (same as analytical implementation)
    if -r / (1.0 + r * cavity_var) > _PREC_CAP:
        r = -1.0 / (cavity_var + 1.0 / _PREC_CAP)
    if -r < 1.0 / _PREC_CAP ** 2:
        r = -1.0 / _PREC_CAP ** 2

    return q, r, log_z


# ---------------------------------------------------------------------------
# 1. Gauss-Laguerre
# ---------------------------------------------------------------------------

class ExponentialNoiseGLLikelihood(Likelihood):
    """
    Gauss-Laguerre EP update for the one-sided exponential likelihood.

    Substitution: let t = λ(y − f) > 0 (valid since f ≤ y).  Then:

        Z = ∫_{−∞}^y λ e^{−λ(y−f)} N(f; μ̃, σ̃²) df
          = ∫_0^∞ e^{−t} N(y − t/λ; μ̃, σ̃²) dt
          ≈ Σ_k  w_k  N(y − t_k/λ; μ̃, σ̃²)

    The Gauss-Laguerre nodes {t_k} are strictly positive, so f_k = y − t_k/λ < y
    for every k.  The truncation constraint is satisfied by construction; there is
    no "dead node" problem regardless of z+s.
    """

    def __init__(self, rate: float = 1.0, n_gl: int = 20):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._rate = float(rate)
        self._nodes, self._weights = np.polynomial.laguerre.laggauss(int(n_gl))

    @property
    def rate(self) -> float:
        return self._rate

    @rate.setter
    def rate(self, v: float) -> None:
        if v <= 0:
            raise ValueError("rate must be positive")
        self._rate = float(v)

    def update_coefficients(
        self, obs: float, cavity_mean: float, cavity_var: float
    ) -> tuple[float, float, float]:
        cavity_var = max(cavity_var, _LAMBDA_TOL)
        rate = self._rate

        # f_k = y − t_k/λ  (all satisfy f_k < y)
        f_pts = obs - self._nodes / rate

        # log N(f_k; μ̃, σ̃²)
        log_g = (
            -0.5 * (f_pts - cavity_mean) ** 2 / cavity_var
            - 0.5 * np.log(2.0 * np.pi * cavity_var)
        )
        log_max = np.max(log_g)
        g = np.exp(log_g - log_max)

        Z = np.dot(self._weights, g)
        if Z < _LAMBDA_TOL:
            return 0.0, -_LAMBDA_TOL, -1e10
        log_z = log_max + np.log(Z)

        mean_t = np.dot(self._weights, f_pts * g) / Z
        var_t  = np.dot(self._weights, f_pts ** 2 * g) / Z - mean_t ** 2
        var_t  = max(var_t, _LAMBDA_TOL)

        return _moment_match(mean_t, var_t, cavity_mean, cavity_var, log_z)

    @property
    def has_em_step(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# 2. Gauss-Legendre on explicit interval [a, y]
# ---------------------------------------------------------------------------

class ExponentialNoiseLegendrelikelihood(Likelihood):
    """
    Gauss-Legendre EP update on the finite interval [a, y].

    The integral ∫_{a}^y λ exp(−λ(y−f)) N(f; μ̃, σ̃²) df is approximated by
    K-point Gauss-Legendre quadrature after a linear mapping from [−1, 1] to
    [a, y].

    The lower bound is  a = min(μ̃ − N_sigma·σ̃, y − 10/λ), chosen to cover
    essentially all Gaussian cavity mass and at least 10 exponential decay
    lengths below y.

    Unlike GH, all K nodes lie strictly within [a, y] ⊆ (−∞, y], so
    truncation is handled explicitly without wasted evaluations.
    """

    def __init__(self, rate: float = 1.0, n_gl: int = 20, n_sigma: float = 6.0):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._rate    = float(rate)
        self._n_sigma = float(n_sigma)
        self._nodes, self._weights = np.polynomial.legendre.leggauss(int(n_gl))

    @property
    def rate(self) -> float:
        return self._rate

    @rate.setter
    def rate(self, v: float) -> None:
        if v <= 0:
            raise ValueError("rate must be positive")
        self._rate = float(v)

    def update_coefficients(
        self, obs: float, cavity_mean: float, cavity_var: float
    ) -> tuple[float, float, float]:
        cavity_var = max(cavity_var, _LAMBDA_TOL)
        rate  = self._rate
        sq_sig = np.sqrt(cavity_var)

        # Integration bounds
        a = min(cavity_mean - self._n_sigma * sq_sig, obs - 10.0 / rate)
        b = obs
        if a >= b:
            a = b - 1e-6

        # Map GL nodes from [−1, 1] to [a, b]
        mid  = 0.5 * (a + b)
        half = 0.5 * (b - a)
        f_pts = mid + half * self._nodes   # shape (K,)

        # Full integrand: λ exp(−λ(y−f)) N(f; μ̃, σ̃²) in log-space
        log_integ = (
            np.log(rate)
            - rate * (obs - f_pts)
            - 0.5 * (f_pts - cavity_mean) ** 2 / cavity_var
            - 0.5 * np.log(2.0 * np.pi * cavity_var)
        )
        log_max  = np.max(log_integ)
        integ    = np.exp(log_integ - log_max)

        # Z ≈ half-width × Σ_k w_k integ_k
        Z = half * np.dot(self._weights, integ)
        if Z < _LAMBDA_TOL:
            return 0.0, -_LAMBDA_TOL, -1e10
        log_z = log_max + np.log(Z)

        norm   = half * Z
        mean_t = half * np.dot(self._weights, f_pts * integ) / Z
        var_t  = half * np.dot(self._weights, f_pts ** 2 * integ) / Z - mean_t ** 2
        var_t  = max(var_t, _LAMBDA_TOL)

        return _moment_match(mean_t, var_t, cavity_mean, cavity_var, log_z)

    @property
    def has_em_step(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# 3. Mode-adapted Gauss-Hermite
# ---------------------------------------------------------------------------

class ExponentialNoiseModeGHLikelihood(Likelihood):
    """
    Mode-adapted Gauss-Hermite EP update.

    Standard GH nodes are centred on the mode of the tilted log-density:

        f* = argmin_f [ λ(y−f) + (f−μ̃)²/(2σ̃²) ]  subject to f ≤ y
           = min(μ̃ + λσ̃², y)

    When z+s > 0 (cavity above observation), the unconstrained mode μ̃ + λσ̃²
    exceeds y, so f* = y.  Centring GH at y places exactly K/2 nodes on each
    side of the truncation boundary; those above y get zero likelihood weight.
    This is dramatically better than standard GH, where the fraction of live
    nodes collapses to near zero as z grows.

    When z+s < 0 (observation well above cavity), f* = μ̃ + λσ̃² < y and all
    nodes may be valid.
    """

    def __init__(self, rate: float = 1.0, n_gh: int = 20):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._rate = float(rate)
        self._nodes, self._weights = np.polynomial.hermite.hermgauss(int(n_gh))

    @property
    def rate(self) -> float:
        return self._rate

    @rate.setter
    def rate(self, v: float) -> None:
        if v <= 0:
            raise ValueError("rate must be positive")
        self._rate = float(v)

    def update_coefficients(
        self, obs: float, cavity_mean: float, cavity_var: float
    ) -> tuple[float, float, float]:
        cavity_var = max(cavity_var, _LAMBDA_TOL)
        rate   = self._rate
        sq_sig = np.sqrt(cavity_var)

        # Mode of tilted log-density (constrained to f ≤ y)
        mode = min(cavity_mean + rate * cavity_var, obs)

        # GH nodes centred on mode
        f_pts   = np.sqrt(2.0) * sq_sig * self._nodes + mode
        log_lik = np.where(
            f_pts <= obs,
            np.log(rate) - rate * (obs - f_pts),
            -np.inf,
        )

        finite = np.isfinite(log_lik)
        if not np.any(finite):
            return 0.0, -_LAMBDA_TOL, -1e10

        log_max = np.max(log_lik[finite])
        lik     = np.exp(log_lik - log_max)

        Z = np.dot(self._weights, lik) / np.sqrt(np.pi)
        if Z < _LAMBDA_TOL:
            return 0.0, -_LAMBDA_TOL, -1e10
        log_z = log_max + np.log(Z)

        norm   = np.sqrt(np.pi) * Z
        mean_t = np.dot(self._weights, f_pts * lik) / norm
        var_t  = np.dot(self._weights, f_pts ** 2 * lik) / norm - mean_t ** 2
        var_t  = max(var_t, _LAMBDA_TOL)

        return _moment_match(mean_t, var_t, cavity_mean, cavity_var, log_z)

    @property
    def has_em_step(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# 4. Laplace approximation
# ---------------------------------------------------------------------------

class ExponentialNoiseLaplaceLikelihood(Likelihood):
    """
    Laplace EP update for the one-sided exponential likelihood.

    The Laplace approximation finds the mode f* of the tilted log-density and
    approximates the tilted distribution as a Gaussian at f* with variance equal
    to the negative inverse Hessian.

    For the exponential likelihood, log p(y|f) = log λ − λ(y−f) is *linear* in f,
    so its second derivative is zero.  The Hessian of the tilted log-density is
    therefore just −1/σ̃² (from the Gaussian cavity alone), giving a tilted
    variance of σ̃² — identical to the cavity.  The moment-matching then yields
    a zero site precision: the Laplace approximation is **degenerate** for the
    exponential likelihood and provides no information.

    The log normalisation is approximated by:
        • Unconstrained mode (f* < y):  log Z ≈ log p̃(f*) + ½log(2πσ̃²)
        • Constrained mode  (f* = y):   log Z ≈ log p̃(y)  + ½log(2πσ̃²) − log 2
          (the half-space Laplace approximation)

    This class is included to demonstrate the fundamental incompatibility of the
    Laplace method with linear log-likelihoods.
    """

    def __init__(self, rate: float = 1.0):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._rate = float(rate)

    @property
    def rate(self) -> float:
        return self._rate

    @rate.setter
    def rate(self, v: float) -> None:
        if v <= 0:
            raise ValueError("rate must be positive")
        self._rate = float(v)

    def update_coefficients(
        self, obs: float, cavity_mean: float, cavity_var: float
    ) -> tuple[float, float, float]:
        cavity_var = max(cavity_var, _LAMBDA_TOL)
        rate = self._rate

        # Mode of tilted log-density
        mode = min(cavity_mean + rate * cavity_var, obs)
        constrained = (mode >= obs - 1e-12)

        # log p̃(f*) = log λ − λ(y−f*) − (f*−μ̃)²/(2σ̃²) − ½log(2πσ̃²)
        log_p_mode = (
            np.log(rate)
            - rate * (obs - mode)
            - 0.5 * (mode - cavity_mean) ** 2 / cavity_var
            - 0.5 * np.log(2.0 * np.pi * cavity_var)
        )

        # Laplace log Z approximation
        log_z_laplace = log_p_mode + 0.5 * np.log(2.0 * np.pi * cavity_var)
        if constrained:
            log_z_laplace -= np.log(2.0)   # half-space correction

        # Tilted mean = mode, tilted variance = cavity_var (zero Hessian)
        mean_t = mode
        var_t  = cavity_var   # ← same as cavity: degenerate

        # moment_match will detect denom = 0 and return uninformative site
        return _moment_match(mean_t, var_t, cavity_mean, cavity_var, log_z_laplace)

    @property
    def has_em_step(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# 5. Importance sampling with truncated-Gaussian proposal
# ---------------------------------------------------------------------------

class ExponentialNoiseTruncISLikelihood(Likelihood):
    """
    Importance-sampling EP update using a truncated-Gaussian proposal.

    Proposal: q(f) = N(f; μ̃, σ̃²) restricted to f ≤ y (TruncNorm).
    All M samples are in the valid region by construction.

    Unnormalised weights:  w_m = p(y|f_m) / q(f_m)
                               ∝ λ exp(−λ(y−f_m)) / [N(f_m; μ̃, σ̃²)/Φ(−z)]
                               = λ Φ(−z) exp(−λ(y−f_m))

    Since only the relative weights matter for the moment estimates, we use:
        log w_m = −λ(y − f_m)   (the exponential factor only)

    Log normalisation:
        Z = E_{q}[p(y|f)] · P(f ≤ y)
          ≈ [λ/M  Σ_m exp(−λ(y−f_m))] · Φ(−z)

    Convergence is uniform in z (all samples are valid), unlike plain-Gaussian IS
    which has the same dead-node problem as standard GH.

    Parameters
    ----------
    rate  : float > 0   exponential rate λ
    n_mc  : int         number of IS samples (default 500)
    seed  : int         random seed for reproducibility
    """

    def __init__(self, rate: float = 1.0, n_mc: int = 500, seed: int = 0):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self._rate = float(rate)
        self._n_mc = int(n_mc)
        self._rng  = np.random.default_rng(seed)

    @property
    def rate(self) -> float:
        return self._rate

    @rate.setter
    def rate(self, v: float) -> None:
        if v <= 0:
            raise ValueError("rate must be positive")
        self._rate = float(v)

    def update_coefficients(
        self, obs: float, cavity_mean: float, cavity_var: float
    ) -> tuple[float, float, float]:
        cavity_var = max(cavity_var, _LAMBDA_TOL)
        rate   = self._rate
        sq_sig = np.sqrt(cavity_var)

        # Upper bound in standardised units: b = (y − μ̃)/σ̃ = −z
        b_std = (obs - cavity_mean) / sq_sig
        if b_std < -30.0:
            # obs so far below cavity that P(f ≤ y) ≈ 0 — no information
            return 0.0, -_LAMBDA_TOL, -1e10

        # Sample from TruncNorm(μ̃, σ̃², upper=y)
        # scipy convention: TruncNorm(a, b) on [(μ+a·σ), (μ+b·σ)]
        samples = truncnorm.rvs(
            a=-np.inf, b=b_std,
            loc=cavity_mean, scale=sq_sig,
            size=self._n_mc,
            random_state=self._rng,
        )

        # IS log-weights: log w_m = −λ(y − f_m) (all non-positive since f_m ≤ y)
        log_w = -rate * (obs - samples)
        log_w_max = np.max(log_w)
        w = np.exp(log_w - log_w_max)
        W = np.sum(w)

        # Tilted moments (importance-weighted)
        mean_t = np.dot(w, samples) / W
        var_t  = np.dot(w, samples ** 2) / W - mean_t ** 2
        var_t  = max(var_t, _LAMBDA_TOL)

        # log Z = log P(f ≤ y) + log λ + log_w_max + log(W/M)
        log_Phi_neg_z = np.log(max(ndtr(b_std), _LAMBDA_TOL))
        log_z = log_Phi_neg_z + np.log(rate) + log_w_max + np.log(W / self._n_mc)

        return _moment_match(mean_t, var_t, cavity_mean, cavity_var, log_z)

    @property
    def has_em_step(self) -> bool:
        return False

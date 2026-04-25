"""
Variable Rank Kriging (VRK) — sparse GP regression with Expectation Propagation.

Mathematical model
------------------
Prior:  f ~ GP(0, K_0)

The posterior is approximated by a finite-dimensional Gaussian parametrised by a
vector α ∈ ℝ^m and matrix C ∈ ℝ^{m×m} over a sparse *active set* (basis vector
set) B = {b_1, …, b_m} of at most max_active training locations.

    Posterior mean:     μ(x)    = k_B(x)ᵀ α
    Posterior kernel:   K(x,x') = K_0(x,x') + k_B(x)ᵀ C k_B(x')

where k_B(x) = [K_0(x, b_1), …, K_0(x, b_m)]ᵀ is the vector of prior covariances
from query point x to each basis location, and K_B ∈ ℝ^{m×m} is the covariance
matrix of the active set.  This parametrisation is Lemma 3.1 (Eq. 33–34, 55–56)
of Csáto (2002).

Stored representation
---------------------
Rather than keeping Q = K_B^{-1} explicitly (numerically fragile under repeated
rank-1 edits), we factored K_B = L L ᵀ and maintain:

    L_      (m×m) lower-triangular Cholesky factor of K_B
    diagQ_  (m,)  diagonal of K_B^{-1}, updated incrementally in O(m)
    C_      (m×m) posterior precision-adjustment matrix
    alpha_  (m,)  posterior mean coefficient vector
    P_      (n×m) projection matrix: P[i,j] records how observation i projects
                  onto basis vector j, used to remove previous site contributions
    mean_ep_(n,)  EP site means  ã_i
    var_ep_ (n,)  EP site precisions  λ_i = −r / (1 + r σ̃²_i)  [always ≥ 0]
    log_z_  (n,)  EP site log normalisation constants

EP algorithm  (Csáto 2002 thesis, Chapter 3l)
--------------------------------------------------------------
Each sweep visits every observation i = 1…n in order:

  Step (a)  Remove site i's previous contribution from (alpha_, C_) so that
            the remaining posterior is the *cavity* distribution.

  Step (b)  Compute cavity moments:
              μ̃_i   = k_iᵀ α_                        (cavity mean)
              σ̃²_i  = K_0(x_i, x_i) + k_iᵀ C_ k_i  (cavity variance)
            where k_i = K_0(B, x_i).

  Step (c)  Likelihood:  (q, r, log z_i) = lik_i.update_coefficients(y_i, μ̃_i, σ̃²_i)
            r < 0 always; −r is the effective site precision.

  Step (d)  Store updated site parameters:
              ã_i  = μ̃_i − q / r   (site mean)
              λ_i  = −r / (1 + r σ̃²_i)   (site precision)

  Step (e)  Information gain  (Csáto 2002 thesis, Eq. 57):
              ℓ     = L^{-1} k_i          (Cholesky forward-solve, O(m²))
              eHat  = L^{-T} ℓ = K_B^{-1} k_i
              γ     = K_0(x_i, x_i) − ‖ℓ‖²   ≥ 0

            γ is the variance of x_i unexplained by the current basis.
            γ = 0 when x_i lies in the column span of K_B.

  Step (f)  Dispatch based on γ and capacity:
              γ > γ_tol, |B| < max_active  → full update: add x_i to basis
              γ > γ_tol, |B| = max_active  → V3 augmented-incremental
                                               (add x_i, score all m+1, delete worst)
              otherwise                    → sparse (projected) update

Sparse update  (Csáto 2002, Eq. 57)
------------------------------------
    η      = 1 / (1 + γ r)
    s      = C_ k_i + eHat          (combined update direction ∈ ℝ^m)
    alpha_ ← alpha_ + η q s
    C_     ← C_     + η r outer(s, s)    (r < 0 → C_ contracts)

Deletion  (Csáto & Opper 2002, Eq. 3.19)
-----------------------------------------
Removing basis vector j from K_B reduces it to a (m−1) × (m−1) sub-matrix.
The corresponding Cholesky factor is restored using a rank-1 Cholesky downdate
(Seeger 2004, cholesky_utils.py) in O(m²), avoiding a full O(m³) recompute.
The (alpha_, C_, P_, diagQ_) arrays are updated via Sherman-Morrison-type identities.

EP site convention  (matching C++ PSGP)
----------------------------------------
    var_ep_[i]  = λ_i  = −r / (1 + r σ̃²_i)   > 0   (site precision)
    mean_ep_[i] = ã_i  = μ̃_i − q/r             (site mean)

References
----------
Csáto, L. (2002). *Gaussian Processes — Iterative Sparse Approximations*.
    PhD Thesis, Aston University, Birmingham, UK.

Csáto, L. and Opper, M. (2002). Sparse On-Line Gaussian Processes.
    *Neural Computation*, 14(3):641–668.

Seeger, M. W. (2004). Low Rank Updates for the Cholesky Decomposition.
    Technical Report, University of California at Berkeley.

Ingram, B., Cornford, D. and Evans, D. (2008). Fast algorithms for automatic mapping
    with space-limited covariance functions. *Stochastic Environmental Research and
    Risk Assessment*, 22:661–670.  https://doi.org/10.1007/s00477-007-0163-9
"""
from __future__ import annotations

import numpy as np
import scipy.linalg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vrk.covariance.base import CovarianceFunction
    from vrk.likelihoods.base import Likelihood

# Minimum EP site precision below which a site is treated as uninitialised.
LAMBDA_TOL = 1e-10

# Minimum information gain γ (default) to trigger adding a basis vector.
GAMMA_TOL = 1e-3


class VRK:
    """
    Variable Rank Kriging with Expectation Propagation.

    Implements sparse GP regression with a growing active set (basis vectors) of
    up to max_active locations.  Non-Gaussian likelihoods are handled via EP:
    each likelihood supplies its EP site update coefficients (q, r, log z) and
    the core (α, C) parametrisation is identical regardless of likelihood type.
    See the module docstring for the full mathematical description.

    Parameters
    ----------
    covariance : CovarianceFunction
        Stationary covariance function K_0.  Typically a SumCovariance combining
        a structural term (e.g. Matern52Covariance) and a NuggetCovariance.
    likelihood : Likelihood | list[Likelihood]
        Observation likelihood.  A single instance is shared across all training
        points.  A list of n instances assigns a distinct likelihood to each
        observation (mixed-likelihood / per-point mode).
    max_active : int
        Maximum active-set size m.  EP step cost scales as O(m²).
        Typical values: 20–100 depending on data volume and smoothness.
    n_sweeps : int
        Number of complete passes over the training data per fit() call.
        More sweeps improve convergence; 2–5 is usually sufficient.
    scoring : str
        Criterion for evicting the least useful basis vector when the active
        set is full.  Lower score → candidate for removal.  Options:
          'full_kl'          (default) Combined mean + data-support + entropy terms;
                             best approximation to KL(q_full ‖ q_reduced).
                             (Csáto 2002, Eq. 74)
          'geometric'        Prior-variance contribution: score_j = 1/diagQ_[j].
          'mean_component'   Mean contribution: α_j² / (c_jj + q_jj).
          'entropy_reduction' Entropy reduction: log(1 + c_jj/q_jj).
          'loo_score'        Leave-one-out approximation with data-support weight.
    gamma_tol : float
        Minimum information gain γ to add a new basis vector (default 1e-3).
        Observations with γ < gamma_tol are handled by a sparse projected update.
    """

    def __init__(
        self,
        covariance: "CovarianceFunction",
        likelihood,
        max_active: int = 50,
        n_sweeps: int = 2,
        scoring: str = "full_kl",
        gamma_tol: float = 1e-3,
    ):
        self.covariance = covariance
        if isinstance(likelihood, list):
            self.likelihood = None
            self._default_likelihoods: list | None = list(likelihood)
        else:
            self.likelihood = likelihood
            self._default_likelihoods = None
        self._likelihoods: list | None = None
        self._per_point: bool = False
        self.max_active = int(max_active)
        self.n_sweeps = int(n_sweeps)
        self.scoring = scoring
        self.gamma_tol = float(gamma_tol)

        # Training data (set during fit)
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._n: int = 0

        # Active-set state  (m = current active-set size)
        # K_B = L_ @ L_.T  is the m×m covariance matrix of the active set B.
        self.L_: np.ndarray | None = None        # (m×m) lower-triangular Cholesky factor of K_B
        self.diagQ_: np.ndarray | None = None    # (m,)  diagonal of K_B^{-1}
        self.C_: np.ndarray | None = None        # (m×m) posterior precision-adjustment matrix
        self.alpha_: np.ndarray | None = None    # (m,)  posterior mean coefficient vector
        self.P_: np.ndarray | None = None        # (n×m) projection matrix (observation → basis)
        self.active_set_: np.ndarray | None = None   # (m×d) active-set locations
        self.active_idx_: np.ndarray | None = None   # (m,)  indices into training data

        # EP site parameters — updated every sweep
        self.mean_ep_: np.ndarray | None = None   # (n,)  site means  ã_i
        self.var_ep_: np.ndarray | None = None    # (n,)  site precisions  λ_i ≥ 0
        self.log_z_: np.ndarray | None = None     # (n,)  site log normalisation constants

    # ==================================================================
    # Public API
    # ==================================================================

    def fit(self, X: np.ndarray, y: np.ndarray,
            likelihoods: list | None = None) -> "VRK":
        """
        Fit the VRK model by running n_sweeps of EP over the training data.

        Parameters
        ----------
        X           : (n, d) training locations
        y           : (n,) observations
        likelihoods : None | list[Likelihood]
            If None, self.likelihood is used for every training point (shared).
            If a list of n Likelihood instances, each point i uses likelihoods[i].
            Overrides any list passed to the constructor.
        """
        X = np.atleast_2d(X)
        if X.ndim == 1:
            X = X[:, None]
        y = np.asarray(y, dtype=float)
        n = X.shape[0]

        _resolved = likelihoods if likelihoods is not None else self._default_likelihoods
        if _resolved is not None:
            if len(_resolved) != n:
                raise ValueError(
                    f"Per-point likelihoods: got {len(_resolved)}, "
                    f"need {n} (one per training point)."
                )
            self._likelihoods = list(_resolved)
            self._per_point = True
        else:
            self._likelihoods = None
            self._per_point = False

        self._X = X
        self._y = y
        self._n = n

        self._run_sweeps(X.shape[1])

        return self

    def _run_sweeps(self, d: int | None = None) -> None:
        """(Re-)initialise state and run n_sweeps of EP over the training data."""
        if d is None:
            d = self._X.shape[1] if self._X is not None else 1
        self._init_state(self._n, d)
        for _ in range(self.n_sweeps):
            self._ep_sweep()

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict posterior mean and variance at new locations.

        Using the (alpha_, C_) parametrisation (Csáto 2002 thesis, Eq. 55–56):

            μ(x*)   = k_B(x*)ᵀ α_
            σ²(x*)  = K_0(x*,x*) + k_B(x*)ᵀ C_ k_B(x*)

        where  C_star = K_0(X_test, B) ∈ ℝ^{n_test × m}  is the cross-covariance
        matrix between test points and the active set, and the variance uses the
        identity  k_B(x*)ᵀ C_ k_B(x*) = sum_j [C_star C_]_{ij}² C_star[i,j].
        Variance is clipped to zero to suppress floating-point negatives.

        Parameters
        ----------
        X_test : (n_test, d) array

        Returns
        -------
        mean : (n_test,)   posterior predictive mean
        var  : (n_test,)   posterior predictive variance (≥ 0)
        """
        X_test = np.atleast_2d(X_test)
        if X_test.ndim == 1:
            X_test = X_test[:, None]

        if self.alpha_ is None or len(self.alpha_) == 0:
            # No active set: return GP prior (zero mean, prior variance)
            c_diag = self.covariance.diag(X_test)
            return np.zeros(X_test.shape[0]), c_diag

        C_star = self.covariance(X_test, self.active_set_)   # (n_test, m)
        mean = C_star @ self.alpha_
        c_diag = self.covariance.diag(X_test)
        # Efficient quadratic form:  diag(C_star C_ C_starᵀ) = row-wise sum of (C_star C_) ⊙ C_star
        var = c_diag + np.sum((C_star @ self.C_) * C_star, axis=1)
        var = np.maximum(var, 0.0)
        return mean, var

    def approximate_evidence(self) -> float:
        """
        EP approximate log-marginal likelihood: sum of log Z_i over all training points.

        This is the sum of per-observation EP normalisation constants:

            log p(y) ≈ Σ_i log Z_i

        where each log Z_i = log_z_[i] is returned by the likelihood's
        update_coefficients().  Maximising this objective w.r.t. covariance
        hyperparameters is a tractable approximation to the true marginal likelihood.
        Works well for non-Gaussian likelihoods.
        """
        return float(np.sum(self.log_z_))

    def active_set_log_evidence(self) -> float:
        """
        Full approximate log-evidence using the MATLAB ogpevid formula.

        This is a port of the MATLAB ogpevid(net, ep) function and implements
        the log-evidence approximation:

            log Z ≈ ½ [ log|I + K_B C| + αᵀ (I + K_B C)^{-1} K_B α
                        − Σ_i ã_i² λ_i + 2 Σ_i log z_i
                        − n log(2π) + Σ_i log λ_i ]

        where  Z = I + K_B C_,  K_B = L_ L_ᵀ,  ã_i = mean_ep_[i],  λ_i = var_ep_[i].

        This formula is algebraically equivalent to the standard GP marginal likelihood
        under a Gaussian likelihood; for non-Gaussian likelihoods it provides a
        reasonable surrogate.  Recommended for covariance hyperparameter optimisation
        with the Gaussian likelihood; use approximate_evidence() for others.

        Returns
        -------
        float  — maximise to optimise covariance hyperparameters
        """
        m = self.n_active
        if m == 0 or self._n == 0:
            return 0.0

        L = self.L_
        KBC = L @ (L.T @ self.C_)          # K_B C
        KBalpha = L @ (L.T @ self.alpha_)  # K_B α

        Z = np.eye(m) + KBC + 1e-8 * np.eye(m)   # I + K_B C  (regularised)

        sign, logdet = np.linalg.slogdet(Z)
        if sign <= 0:
            return -np.inf
        try:
            quad = float(self.alpha_ @ np.linalg.solve(Z, KBalpha))
        except np.linalg.LinAlgError:
            return -np.inf

        # EP correction terms: − Σ ã²λ + 2 Σ log z − n log(2π) + Σ log λ
        ep_mask = self.var_ep_ > LAMBDA_TOL
        if np.any(ep_mask):
            ep_corr = (
                - np.sum(self.mean_ep_[ep_mask] ** 2 * self.var_ep_[ep_mask])
                + 2.0 * np.sum(self.log_z_[ep_mask])
                - self._n * np.log(2.0 * np.pi)
                + np.sum(np.log(np.maximum(self.var_ep_[ep_mask], LAMBDA_TOL)))
            )
        else:
            ep_corr = 0.0

        return 0.5 * (logdet + quad + ep_corr)

    def alt_log_evidence(self) -> float:
        """
        Alternative log-evidence using the K_B^{-1} + C form of the ogpevid formula.

        Algebraically equivalent to active_set_log_evidence() when the likelihood
        is Gaussian.  Uses:

            log Z ≈ ½ [ log|K_B^{-1} + C| + log|K_B| + αᵀ (K_B^{-1} + C)^{-1} α
                        + EP correction ]

        Returns
        -------
        float — same interpretation as active_set_log_evidence()
        """
        m = self.n_active
        if m == 0 or self._n == 0:
            return 0.0

        L = self.L_
        eps = 1e-8 * np.eye(m)

        KBinv = scipy.linalg.cho_solve((L, True), np.eye(m))
        Z2 = KBinv + self.C_

        try:
            L_Z2 = scipy.linalg.cholesky(Z2 + eps, lower=True)
            logdet_Z2 = 2.0 * np.sum(np.log(np.diag(L_Z2)))
            quad = float(self.alpha_ @ scipy.linalg.cho_solve((L_Z2, True), self.alpha_))
        except np.linalg.LinAlgError:
            sign, logdet_Z2 = np.linalg.slogdet(Z2)
            if sign <= 0:
                return -np.inf
            try:
                quad = float(self.alpha_ @ np.linalg.solve(Z2, self.alpha_))
            except np.linalg.LinAlgError:
                return -np.inf

        logdet_KB = 2.0 * np.sum(np.log(np.diag(L)))

        ep_mask = self.var_ep_ > LAMBDA_TOL
        if np.any(ep_mask):
            ep_corr = (
                - np.sum(self.mean_ep_[ep_mask] ** 2 * self.var_ep_[ep_mask])
                + 2.0 * np.sum(self.log_z_[ep_mask])
                - self._n * np.log(2.0 * np.pi)
                + np.sum(np.log(np.maximum(self.var_ep_[ep_mask], LAMBDA_TOL)))
            )
        else:
            ep_corr = 0.0

        return 0.5 * (logdet_Z2 + logdet_KB + quad + ep_corr)

    def kl_from_prior(self) -> float:
        """
        KL divergence from the GP prior to the VRK posterior: KL(q ‖ p_prior).

        For a Gaussian distribution with mean m and covariance S against a
        zero-mean GP with covariance K_0, the KL is (Csáto 2002, Eq. 66):

            2 KL(q ‖ p) = tr(C_ K_B) + αᵀ K_B α − log|I + C_ K_B|

        where K_B = L_ L_ᵀ and C_ is the posterior precision adjustment.
        The result equals zero when the posterior matches the prior (no data).

        Returns
        -------
        float ≥ 0
        """
        m = self.n_active
        if m == 0:
            return 0.0

        L = self.L_
        KBC = L @ (L.T @ self.C_)           # K_B C_
        KBalpha = L @ (L.T @ self.alpha_)   # K_B α

        Z = np.eye(m) + KBC + 1e-8 * np.eye(m)
        try:
            L_Z = scipy.linalg.cholesky(Z, lower=True)
            logdet = 2.0 * np.sum(np.log(np.diag(L_Z)))
        except np.linalg.LinAlgError:
            sign, logdet = np.linalg.slogdet(Z)
            if sign <= 0:
                return 0.0

        tr_CKB = np.trace(KBC)
        quad = float(self.alpha_ @ KBalpha)

        return 0.5 * (tr_CKB + quad - logdet)

    def elbo(self) -> float:
        """
        Evidence Lower BOund (ELBO):

            ELBO = Σ_i log Z_i − KL(q ‖ p_prior)

        The first term rewards data fit (via EP normalisation constants) and the
        second penalises the posterior for deviating from the prior.  Maximising
        the ELBO with respect to hyperparameters is a standard variational
        objective.

        Returns
        -------
        float  (maximise to fit hyperparameters)
        """
        return self.approximate_evidence() - self.kl_from_prior()

    def kl_to(self, other: "VRK") -> float:
        """
        KL divergence KL(q_self ‖ q_other) between two fitted VRK posteriors.

        Used to measure how much two models (e.g. trained with different
        hyperparameters) differ.  When both models share the same active set,
        a closed-form expression is used (Csáto 2002, Eq. 74):

            2 KL = − log|I + C_1 K_B2| + log|I + C_2 K_B2|
                   + tr[(C_1 − C_2)(I + C_2 K_B2)^{-1} K_B2]
                   + (α_2 − α_1)ᵀ (I + C_2 K_B2)^{-1} K_B2 (α_2 − α_1)

        When the active sets differ, K_B2^{-1} is replaced by the cross-covariance
        K(B_2, B_1) evaluated from the other model's covariance function.

        Returns
        -------
        float ≥ 0
        """
        m1 = self.n_active
        m2 = other.n_active

        if m1 == 0 or m2 == 0:
            return 0.0

        same_bv = (
            m1 == m2
            and self.active_idx_ is not None
            and other.active_idx_ is not None
            and np.array_equal(self.active_idx_, other.active_idx_)
        )

        if same_bv:
            KB2 = other.L_ @ other.L_.T
            C1_KB2 = self.C_ @ KB2
            S_s = np.eye(m2) + other.C_ @ KB2
            m_d = other.alpha_ - self.alpha_

            try:
                L1 = scipy.linalg.cholesky(
                    np.eye(m1) + C1_KB2 + 1e-8 * np.eye(m1), lower=True
                )
                ld1 = 2.0 * np.sum(np.log(np.diag(L1)))
            except np.linalg.LinAlgError:
                _, ld1 = np.linalg.slogdet(np.eye(m1) + C1_KB2)

            try:
                L_Ss = scipy.linalg.cholesky(S_s + 1e-8 * np.eye(m2), lower=True)
                ld2 = 2.0 * np.sum(np.log(np.diag(L_Ss)))
                KB2_Ssi = scipy.linalg.cho_solve((L_Ss, True), KB2.T).T
            except np.linalg.LinAlgError:
                _, ld2 = np.linalg.slogdet(S_s)
                KB2_Ssi = np.linalg.solve(S_s.T, KB2.T).T

            tr_term = np.trace((self.C_ - other.C_) @ KB2_Ssi)
            quad_term = float(m_d @ KB2_Ssi @ m_d)

            return 0.5 * (-ld1 + ld2 + tr_term + quad_term)

        else:
            # Different active sets: use cross-covariance K(B_2, B_1)
            KB2 = other.L_ @ other.L_.T
            Kpr = other.covariance(other.active_set_, self.active_set_)   # (m2, m1)
            S_s = np.eye(m2) + other.C_ @ KB2
            m_d = KB2 @ other.alpha_ - Kpr @ self.alpha_

            try:
                L_Ss = scipy.linalg.cholesky(S_s + 1e-8 * np.eye(m2), lower=True)
                ld2 = 2.0 * np.sum(np.log(np.diag(L_Ss)))
                Ss_inv_Kpr = scipy.linalg.cho_solve((L_Ss, True), Kpr)
                inner1 = np.eye(m1) + self.C_ @ (Kpr.T @ Ss_inv_Kpr)
                try:
                    L1 = scipy.linalg.cholesky(inner1 + 1e-8 * np.eye(m1), lower=True)
                    ld1 = 2.0 * np.sum(np.log(np.diag(L1)))
                except np.linalg.LinAlgError:
                    _, ld1 = np.linalg.slogdet(inner1)

                Ss_inv_md = scipy.linalg.cho_solve((L_Ss, True), m_d)
                quad_term = float(m_d @ Ss_inv_md)

                KB1 = self.L_ @ self.L_.T
                C2_Kpr = other.C_ @ Kpr
                Ss_inv_C2_Kpr = scipy.linalg.cho_solve((L_Ss, True), C2_Kpr)
                tr_term = np.trace(self.C_ @ KB1) - np.trace(Kpr.T @ Ss_inv_C2_Kpr @ KB1)

            except np.linalg.LinAlgError:
                return 0.0

            return 0.5 * (-ld1 + ld2 + tr_term + quad_term)

    def analytic_evidence_gradient(self) -> np.ndarray:
        """
        Analytic gradient of active_set_log_evidence() w.r.t. log-space covariance params.

        Differentiates  F = ½(log|Z| + αᵀ Z^{-1} K_B α + EP_correction)
        with  Z = I + K_B C_  through the matrix identity:

            ∂F/∂θ = ½ tr(G · M)

        where:
            G   = ∂K_B/∂θ              (provided by covariance.gradient_matrix())
            M   = C_ Z^{-1} + outer(α − C_ b, â)
            â   = Z^{-T} α             (adjoint solve)
            b   = Z^{-1} K_B α         (forward solve)
            C_ b = C_ b

        The chain rule to log-space is applied via the reparametrisation
        ∂F/∂ log θ = (∂F/∂θ) · θ = (∂F/∂θ) · exp(log θ).

        Returns
        -------
        grad : (n_params,) gradient in log-space  ∂F/∂ log θ
        """
        m = self.n_active
        if m == 0 or self._X is None:
            return np.zeros(self.covariance.n_params)

        L = self.L_
        KBC = L @ (L.T @ self.C_)
        KBalpha = L @ (L.T @ self.alpha_)

        Z = np.eye(m) + KBC + 1e-8 * np.eye(m)
        try:
            lu, piv = scipy.linalg.lu_factor(Z)
            def _solve(rhs):
                return scipy.linalg.lu_solve((lu, piv), rhs)
            def _solve_T(rhs):
                return scipy.linalg.lu_solve((lu, piv), rhs, trans=1)
        except np.linalg.LinAlgError:
            return np.zeros(self.covariance.n_params)

        b    = _solve(KBalpha)      # b = Z^{-1} K_B α
        Cb   = self.C_ @ b          # C_ b
        ahat = _solve_T(self.alpha_) # â = Z^{-T} α
        ZtC  = _solve_T(self.C_)
        CZi  = ZtC.T                # C_ Z^{-1}

        M = CZi + np.outer(self.alpha_ - Cb, ahat)

        # ½ tr(G · M) for each parameter gradient matrix G = ∂K_B/∂θ
        gradient_matrices = self.covariance.gradient_matrix(self.active_set_)
        grad_natural = np.array([0.5 * np.sum(M * G) for G in gradient_matrices])

        # Convert natural-space gradient to log-space via ∂F/∂ log θ = (∂F/∂θ) · θ
        grad_log = grad_natural * np.exp(self.covariance.log_params)
        return grad_log

    def active_set_evidence_gradient(self) -> np.ndarray:
        """
        Finite-difference gradient of active_set_log_evidence w.r.t. log-space params.

        Perturbs each log-parameter by eps = 1e-3, refits the model, and computes
        the central (forward) difference.  Slower than analytic_evidence_gradient()
        but always available regardless of covariance type.
        """
        if self._X is None or self._n == 0:
            return np.zeros(self.covariance.n_params)
        eps = 1e-3
        lp0 = self.covariance.log_params.copy()
        ev0 = self.active_set_log_evidence()
        grad = np.zeros(self.covariance.n_params)
        for j in range(self.covariance.n_params):
            lp_p = lp0.copy()
            lp_p[j] += eps
            self.covariance.log_params = lp_p
            self._run_sweeps()
            grad[j] = (self.active_set_log_evidence() - ev0) / eps
        self.covariance.log_params = lp0
        self._run_sweeps()
        return grad

    def evidence_gradient(self) -> np.ndarray:
        """
        Finite-difference gradient of approximate_evidence w.r.t. log-space params.

        Uses the EP sum-of-log-Z objective.  Suitable for non-Gaussian likelihoods.

        Returns
        -------
        grad : (n_params,)  ∂(Σ log z_i) / ∂ log θ
        """
        if self._X is None or self._n == 0:
            return np.zeros(self.covariance.n_params)
        eps = 1e-3
        lp0 = self.covariance.log_params.copy()
        ev0 = self.approximate_evidence()
        grad = np.zeros(self.covariance.n_params)
        for j in range(self.covariance.n_params):
            lp_p = lp0.copy()
            lp_p[j] += eps
            self.covariance.log_params = lp_p
            self._run_sweeps()
            grad[j] = (self.approximate_evidence() - ev0) / eps
        self.covariance.log_params = lp0
        self._run_sweeps()
        return grad

    # ==================================================================
    # EP internals
    # ==================================================================

    def _init_state(self, n: int, d: int = 1) -> None:
        """Reset all state to an empty active set and uninitialised EP sites."""
        if self._X is not None:
            d = self._X.shape[1]
        self.L_ = np.zeros((0, 0))
        self.diagQ_ = np.zeros(0)
        self.C_ = np.zeros((0, 0))
        self.alpha_ = np.zeros(0)
        self.P_ = np.zeros((n, 0))
        self.active_set_ = np.zeros((0, d))
        self.active_idx_ = np.zeros(0, dtype=int)
        self.mean_ep_ = np.zeros(n)
        self.var_ep_ = np.zeros(n)
        self.log_z_ = np.zeros(n)

    def _ep_sweep(self) -> None:
        """
        One full EP pass over all n training observations.

        For each observation i, the sweep: removes site i's previous contribution
        (Step a), computes the cavity distribution (Step b), obtains new EP
        parameters from the likelihood (Step c–d), and applies either a full or
        sparse update (Steps e–f).
        See module docstring for the complete mathematical description.
        """
        n = self._n
        for i in range(n):
            self._ep_remove_previous(i)

            x_i = self._X[i: i + 1]
            cavity_mean, cavity_var, eHat, gamma, l_fwd = self._ep_cavity(i, x_i)

            lik_i = self._likelihoods[i] if self._per_point else self.likelihood
            q, r, log_z = lik_i.update_coefficients(
                self._y[i], cavity_mean, cavity_var
            )

            # Skip degenerate updates (r ≥ 0 would increase cavity variance)
            if not np.isfinite(q) or not np.isfinite(r) or r >= 0:
                continue

            # Store EP site parameters: ã_i = μ̃_i − q/r,  λ_i = −r/(1 + r σ̃²_i)
            ratio = q / r
            self.log_z_[i] = log_z + 0.5 * (np.log(2.0 * np.pi) - np.log(abs(r)) - q * ratio)
            self.mean_ep_[i] = cavity_mean - ratio
            self.var_ep_[i] = -r / (1.0 + r * cavity_var)

            k = (self.covariance(self.active_set_, x_i).flatten()
                 if len(self.active_set_) > 0 else np.zeros(0))

            already_active = i in self.active_idx_

            if already_active:
                # x_i is already a basis vector; apply sparse update in place
                self._ep_sparse_update(i, k, eHat, q, r)
            elif len(self.active_set_) < self.max_active:
                if gamma > self.gamma_tol:
                    # High novelty and room to grow: add x_i to the active set
                    self._add_active_point_v3(i, x_i, k, eHat, gamma, q, r, l_fwd)
                else:
                    # Low novelty: project update onto existing basis
                    self._ep_sparse_update(i, k, eHat, q, r)
            else:
                if gamma > self.gamma_tol:
                    # Active set full: add x_i then evict the lowest-scoring vector
                    self._add_active_augmented_v3(i, x_i, k, eHat, gamma, q, r, l_fwd)
                else:
                    self._ep_sparse_update(i, k, eHat, q, r)

    # ------------------------------------------------------------------
    # Step (a): EP site removal
    # ------------------------------------------------------------------

    def _ep_remove_previous(self, i: int) -> None:
        """
        Remove site i's previous EP contribution from (alpha_, C_).

        Before recomputing the EP update for site i, its previous contribution
        must be subtracted to obtain the cavity distribution.  This implements
        the "parameter adjustment" step of the sparse EP algorithm
        (Csáto 2002, Step (a)).

        Let:
            p   = P_[i, :]                     (projection vector for site i)
            Kp  = K_B p = L (Lᵀ p)             (K_B applied to p, via Cholesky)
            h   = C_ Kp + p                    (combined direction vector)
            ν   = λ_i / (1 − λ_i (pᵀ K_B h))  (removal scalar, λ_i = var_ep_[i])

        Update:
            alpha_ ← alpha_ + h · ν · (αᵀ K_B p − ã_i)
            C_     ← C_     + ν · outer(h, h)
        """
        if self.var_ep_[i] < LAMBDA_TOL:
            return   # Site i has never been updated; nothing to remove
        if len(self.active_set_) == 0:
            return

        p = self.P_[i, :]
        Kp = self.L_ @ (self.L_.T @ p)      # K_B p via Cholesky
        h = self.C_ @ Kp + p
        nu = self.var_ep_[i] / (1.0 - self.var_ep_[i] * (Kp @ h))
        self.alpha_ += h * nu * (self.alpha_ @ Kp - self.mean_ep_[i])
        self.C_ += nu * np.outer(h, h)

    # ------------------------------------------------------------------
    # Step (b): Cavity computation
    # ------------------------------------------------------------------

    def _ep_cavity(
        self, i: int, x_i: np.ndarray
    ) -> tuple[float, float, np.ndarray, float, np.ndarray]:
        """
        Compute the EP cavity distribution at observation i.

        The cavity is the posterior predictive at x_i after removing site i's
        previous contribution (already done by _ep_remove_previous).

        From the (α, C_) representation (Csáto 2002 thesis, Eq. 55–56):
            cavity_mean = k_iᵀ α_
            cavity_var  = K_0(x_i, x_i) + k_iᵀ C_ k_i

        Additional quantities needed for the subsequent update:
            ℓ_fwd = L^{-1} k_i        (Cholesky forward-solve, O(m²))
            eHat  = L^{-T} ℓ_fwd      (= K_B^{-1} k_i)
            γ     = K_0(x_i,x_i) − ‖ℓ_fwd‖²   (information gain, ≥ 0)

        γ measures the variance of x_i unexplained by the current basis:
        γ = 0 when x_i is a linear combination of existing basis vectors.

        Returns
        -------
        cavity_mean : float
        cavity_var  : float  (> 0)
        eHat        : (m,)   K_B^{-1} k_i
        gamma       : float  information gain
        l_fwd       : (m,)   L^{-1} k_i (forward solve result)
        """
        sigma_loc = float(self.covariance.diag(x_i)[0])   # K_0(x_i, x_i)

        if len(self.active_set_) == 0:
            # Prior: cavity = GP prior at x_i (no basis yet)
            eHat = np.zeros(0)
            l_fwd = np.zeros(0)
            return 0.0, sigma_loc, eHat, sigma_loc, l_fwd

        k = self.covariance(self.active_set_, x_i).flatten()   # k_i ∈ ℝ^m
        cavity_mean = float(k @ self.alpha_)
        cavity_var = sigma_loc + float(k @ self.C_ @ k)
        cavity_var = max(cavity_var, LAMBDA_TOL)

        # Cholesky solves for eHat and γ
        l_fwd = scipy.linalg.solve_triangular(self.L_, k, lower=True)   # L^{-1} k_i
        eHat = scipy.linalg.solve_triangular(self.L_.T, l_fwd, lower=False)  # K_B^{-1} k_i
        gamma = sigma_loc - float(np.dot(l_fwd, l_fwd))   # K_0(x_i,x_i) − kᵀ K_B^{-1} k
        gamma = max(gamma, 0.0)

        return cavity_mean, cavity_var, eHat, gamma, l_fwd

    # ------------------------------------------------------------------
    # Sparse (projected) update
    # ------------------------------------------------------------------

    def _ep_sparse_update(
        self,
        i: int,
        k: np.ndarray,
        eHat: np.ndarray,
        q: float,
        r: float,
    ) -> None:
        """
        Rank-1 EP update projected onto the existing active set.

        Used when the information gain γ < γ_tol, meaning x_i is already
        well-represented by the current basis.  Does not change the active set.

        From Csáto (2002) thesis, Eq. 57:
            γ   = K_0(x_i,x_i) − k_iᵀ eHat        (residual novelty)
            η   = 1 / (1 + γ r)                     (normalisation scalar)
            s   = C_ k_i + eHat                     (update direction ∈ ℝ^m)
            alpha_ ← alpha_ + η q s
            C_     ← C_     + η r outer(s, s)        (r < 0: contraction)

        The projection vector P_[i, :] is stored as eHat so that _ep_remove_previous()
        can undo this update in a later sweep.
        """
        if len(self.active_set_) == 0:
            return

        sigma_loc = float(self.covariance.diag(self._X[i: i + 1])[0])
        gamma = sigma_loc - float(k @ eHat)
        eta = 1.0 / (1.0 + gamma * r)

        s = self.C_ @ k + eHat
        self.alpha_ += eta * s * q
        self.C_ += r * eta * np.outer(s, s)

        self.P_[i, :] = eHat   # Store projection for future removal

    # ------------------------------------------------------------------
    # Full update — expand active set by one
    # ------------------------------------------------------------------

    def _add_active_point_v3(
        self,
        i: int,
        x_i: np.ndarray,
        k: np.ndarray,
        eHat: np.ndarray,
        gamma: float,
        q: float,
        r: float,
        l_fwd: np.ndarray,
    ) -> None:
        """
        Add observation x_i as the (m+1)-th basis vector and perform the full EP update.

        Cholesky extension  (Seeger 2004):
        The new Cholesky factor is formed by appending a row/column:

            L_new = [ L_old     0  ]    where  δ = √max(γ, 0)
                    [ ℓ_fwdᵀ   δ  ]

        This is exact because  K_B_new = L_new L_newᵀ  and  L_old L_oldᵀ = K_B_old.

        The diagonal of K_B^{-1} is updated:
            diagQ_new[j]   = diagQ_[j] + eHat[j]² / γ   for j = 0…m−1
            diagQ_new[m]   = 1 / γ

        The posterior parameters (alpha_, C_) are extended by appending a zero
        to alpha_ and a zero row/column to C_, then updated by the full EP step:
            s_aug = [C_ k_i; 1]      (update direction in ℝ^{m+1})
            alpha_new ← alpha_new + q · s_aug
            C_new     ← C_new     + r · outer(s_aug, s_aug)
        """
        m = len(self.active_set_)

        # Augmented update direction: append 1 for the new basis dimension
        s_aug = np.append(self.C_ @ k, 1.0)

        new_alpha = np.append(self.alpha_, 0.0)
        new_C = np.zeros((m + 1, m + 1))
        new_C[:m, :m] = self.C_

        new_alpha += s_aug * q
        new_C += r * np.outer(s_aug, s_aug)

        # Cholesky extension: δ = √γ
        d_new = np.sqrt(max(gamma, 0.0) + 1e-10)
        sigma_loc = float(self.covariance.diag(x_i)[0])
        if m == 0:
            new_L = np.array([[np.sqrt(sigma_loc + 1e-10)]])
        else:
            new_L = np.zeros((m + 1, m + 1))
            new_L[:m, :m] = self.L_
            new_L[m, :m] = l_fwd   # L^{-1} k_i becomes the new off-diagonal row
            new_L[m, m] = d_new

        # Update diagonal of K_B^{-1}
        g_reg = gamma + 1e-10
        if m == 0:
            new_diagQ = np.array([1.0 / (sigma_loc + 1e-10)])
        else:
            new_diagQ = np.append(
                self.diagQ_ + eHat ** 2 / g_reg,   # existing entries shift
                1.0 / g_reg                          # new entry = 1/γ
            )

        # Extend projection matrix: new observation maps to new basis column
        new_P = np.zeros((self._n, m + 1))
        new_P[:, :m] = self.P_
        new_P[i, m] = 1.0

        new_active = np.vstack([self.active_set_, x_i]) if m > 0 else x_i.copy()
        new_idx = np.append(self.active_idx_, i)

        self.alpha_ = new_alpha
        self.C_ = new_C
        self.L_ = new_L
        self.diagQ_ = new_diagQ
        self.P_ = new_P
        self.active_set_ = new_active
        self.active_idx_ = new_idx

        # Set projection row for i: purely projects onto its own (new) basis column
        self.P_[i, :] = 0.0
        self.P_[i, m] = 1.0

    # ------------------------------------------------------------------
    # V3 augmented-incremental active-set management
    # ------------------------------------------------------------------

    def _add_active_augmented_v3(
        self,
        i: int,
        x_i: np.ndarray,
        k: np.ndarray,
        eHat: np.ndarray,
        gamma: float,
        q: float,
        r: float,
        l_fwd: np.ndarray,
    ) -> None:
        """
        V3 augmented-incremental update when the active set is at capacity.

        The algorithm is:
          1. Temporarily expand the active set to m+1 (full update for x_i).
          2. Score all m+1 candidates using _score_active_points().
          3. Delete the candidate with the lowest score.

        This ensures every new high-information-gain observation gets a chance to
        enter the basis, and the globally least useful vector is evicted, rather
        than greedily rejecting the new candidate because the set is full.
        """
        self._add_active_point_v3(i, x_i, k, eHat, gamma, q, r, l_fwd)

        scores = self._score_active_points()
        i_del = int(np.argmin(scores))
        self._delete_active_point(i_del)

    def _score_active_points(self) -> np.ndarray:
        """
        Score each active point; lower score → candidate for removal.

        Variables:
            a      = alpha_              posterior mean coefficients
            diagC  = diag(C_)           diagonal of precision adjustment
            diagQ  = diagQ_             diagonal of K_B^{-1}

        Scoring modes  (Csáto 2002 thesis):

        'geometric':
            score_j = 1 / diagQ_[j]
            Proportional to the marginal prior variance at basis point j.
            Points with large prior variance (small diagQ, informative location)
            receive high scores and are retained.

        'mean_component':
            score_j = α_j² / (c_jj + q_jj)
            Weighted contribution of basis j to the posterior mean, normalised by
            total (prior + posterior) precision.

        'entropy_reduction':
            score_j = log(1 + c_jj / q_jj)
            Approximate entropy reduction in the predictive distribution due to
            including basis j.  Derived from the KL-optimal projection
            (Csáto 2002, Eq. 66).

        'loo_score':
            Adds data-support via EP site precisions: w_j = Σ_i λ_i P_[i,j]².
            score_j = (α_j² w_j) / diagQ_[j]

        'full_kl'  (default):
            Combines all three signals — mean contribution, data support, and
            entropy reduction — approximating KL(q_full ‖ q_reduced).
            (Csáto 2002, Eq. 74)
        """
        m = self.n_active
        a = self.alpha_
        diagC = np.diag(self.C_)
        diagQ = self.diagQ_

        if self.scoring == "geometric":
            return 1.0 / (diagQ + LAMBDA_TOL)

        elif self.scoring == "mean_component":
            return (a ** 2) / (diagC + diagQ + LAMBDA_TOL)

        elif self.scoring == "entropy_reduction":
            ratio = diagC / (diagQ + LAMBDA_TOL)
            ratio_clipped = np.maximum(ratio, -1.0 + 1e-10)
            return np.log1p(ratio_clipped)

        elif self.scoring == "loo_score":
            # Data-support weight: w_j = Σ_i λ_i P_[i,j]²
            diagS = np.array([
                np.sum(self.var_ep_ * self.P_[:, j] ** 2)
                for j in range(m)
            ])
            return (a ** 2 * diagS) / (diagQ + LAMBDA_TOL)

        else:  # full_kl (default)
            diagS = np.array([
                np.sum(self.var_ep_ * self.P_[:, j] ** 2)
                for j in range(m)
            ])
            ratio = diagC / (diagQ + LAMBDA_TOL)
            ratio_clipped = np.maximum(ratio, -1.0 + 1e-10)
            return (
                (a ** 2) / (diagC + diagQ + LAMBDA_TOL)    # mean contribution
                + diagS / (diagQ + LAMBDA_TOL)              # data support
                + np.log1p(ratio_clipped)                   # entropy reduction
            )

    # ------------------------------------------------------------------
    # Deletion step  (Csáto & Opper 2002, Eq. 3.19)
    # ------------------------------------------------------------------

    def _delete_active_point(self, i_del: int) -> None:
        """
        Remove basis vector i_del from the active set.

        Implements the deletion formula of Csáto & Opper (2002), Eq. 3.19, and
        Csáto (2002) thesis (Step (g)).

        Let:
            α_i   = alpha_[i_del]           scalar mean coefficient for this BV
            c_i   = C_[i_del, i_del]        diagonal entry of precision adjustment
            C_i   = C_[i_del, :]            row of C_ for this BV
            P_i   = P_[:, i_del]            projection column for this BV
            Q_col = K_B^{-1} e_{i_del}     column i_del of K_B^{-1}  (Cholesky solve)
            q_i   = Q_col[i_del]            diagonal entry of K_B^{-1}
            denom = c_i + q_i

        Update rules:
            alpha_ ← alpha_ − (α_i / denom) · (Q_col + C_i)
            C_     ← C_ + outer(Q_col, Q_col)/q_i − outer(Q_col+C_i, Q_col+C_i)/denom
            P_     ← P_ − outer(P_i, Q_col) / q_i

        After updating the n×m matrices, row/column i_del is deleted and the
        Cholesky factor L_ is updated via cholesky_delete() from cholesky_utils.py,
        which applies a rank-1 Cholesky downdate in O(m²)  (Seeger 2004).
        """
        from vrk.core.cholesky_utils import cholesky_delete

        m = len(self.active_set_)
        alpha_i = self.alpha_[i_del]
        c_i = self.C_[i_del, i_del]
        C_i = self.C_[i_del, :].copy()
        P_i = self.P_[:, i_del].copy()

        # Retrieve column i_del of K_B^{-1} via Cholesky solve: K_B^{-1} e_{i_del}
        e_j = np.eye(m)[:, i_del]
        Q_col = scipy.linalg.cho_solve((self.L_, True), e_j)
        q_i = Q_col[i_del]

        denom = c_i + q_i
        if abs(denom) < LAMBDA_TOL:
            denom = LAMBDA_TOL

        # Update alpha_ and C_ (Csáto & Opper 2002, Eq. 3.19)
        self.alpha_ -= (alpha_i / denom) * (Q_col + C_i)
        QQq = np.outer(Q_col, Q_col) / (q_i + LAMBDA_TOL)
        self.C_ += QQq - np.outer(Q_col + C_i, Q_col + C_i) / denom
        self.P_ -= np.outer(P_i, Q_col) / (q_i + LAMBDA_TOL)

        diagQ_new = self.diagQ_ - Q_col ** 2 / (q_i + LAMBDA_TOL)

        keep = np.delete(np.arange(m), i_del)

        self.alpha_ = self.alpha_[keep]
        self.C_ = self.C_[np.ix_(keep, keep)]
        self.P_ = self.P_[:, keep]
        self.diagQ_ = diagQ_new[keep]
        self.active_set_ = self.active_set_[keep]
        self.active_idx_ = self.active_idx_[keep]

        # O(m²) Cholesky downdate (Seeger 2004); avoids full O(m³) recompute
        self.L_ = cholesky_delete(self.L_, i_del)

    # ==================================================================
    # Properties
    # ==================================================================

    @property
    def n_active(self) -> int:
        """Current active-set size m (number of basis vectors)."""
        return len(self.active_set_) if self.active_set_ is not None else 0

    @property
    def n_likelihoods(self) -> int:
        """1 if shared likelihood; n if per-point likelihoods were passed to fit()."""
        return len(self._likelihoods) if self._per_point else 1

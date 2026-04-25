"""
O(m²) Cholesky rank-1 update and rank-1 deletion utilities.

The Cholesky factor L (where A = L Lᵀ) is maintained incrementally as
the active set grows and shrinks.  Both operations run in O(m²) rather
than the O(m³) cost of a full recompute.

References
----------
Seeger, M. W. (2004). Low Rank Updates for the Cholesky Decomposition.
    Technical Report, University of California at Berkeley.

Csáto, L. and Opper, M. (2002). Sparse On-Line Gaussian Processes.
    *Neural Computation*, 14(3):641–668.  (Deletion formula, Eq. 3.19)

Ingram, B., Cornford, D. and Evans, D. (2008). Fast algorithms for automatic mapping
    with space-limited covariance functions. *Stochastic Environmental Research and
    Risk Assessment*, 22:661–670.  https://doi.org/10.1007/s00477-007-0163-9
"""
import numpy as np


def cholesky_rank1_update(L: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rank-1 Cholesky update: given lower-triangular L with A = L Lᵀ and vector v,
    return L_new such that  L_new L_newᵀ = A + outer(v, v).

    Algorithm  (Seeger 2004)
    ------------------------
    The update proceeds column by column using a sequence of m Givens (plane)
    rotations.  At step k, a rotation (c, s) is chosen to zero the current
    element of the update vector v[k] into the diagonal:

        r       = hypot(L[k,k], v[k])
        c, s    = L[k,k] / r,  v[k] / r
        L[k,k]  = r

    The rotation is then propagated to the remaining sub-column:

        L[k+1:, k]_new = c · L[k+1:, k] + s · v[k+1:]
        v[k+1:]_new    = −s · L[k+1:, k] + c · v[k+1:]

    This maintains the lower-triangular structure while incorporating the
    rank-1 update v vᵀ.  The total cost is O(m²) floating-point operations.

    Parameters
    ----------
    L : (m, m) lower-triangular Cholesky factor  (L Lᵀ = A)
    v : (m,)   update vector

    Returns
    -------
    L_new : (m, m) updated lower-triangular Cholesky factor  (L_new L_newᵀ = A + vvᵀ)
    """
    L = L.copy().astype(float)
    v = v.copy().astype(float)
    m = L.shape[0]

    for k in range(m):
        r = np.hypot(L[k, k], v[k])    # r = √(L[k,k]² + v[k]²)
        if r < 1e-300:
            continue
        c = L[k, k] / r
        s = v[k] / r
        L[k, k] = r
        if k + 1 < m:
            old_col = L[k + 1:, k].copy()
            L[k + 1:, k] = c * old_col + s * v[k + 1:]
            v[k + 1:] = -s * old_col + c * v[k + 1:]

    return L


def cholesky_delete(L: np.ndarray, i_del: int) -> np.ndarray:
    """
    Cholesky downdate after deleting row/column i_del from A = L Lᵀ.

    Returns L_new ∈ ℝ^{(m-1)×(m-1)} such that  L_new L_newᵀ = A_{−i,−i},
    where A_{−i,−i} is A with row and column i_del removed.  Cost is O(m²).

    Derivation
    ----------
    Partitioning L with the deleted column separated:

        A = L Lᵀ = L_red L_redᵀ + v vᵀ

    where:
        L_red = L[keep, :][:, keep]  (lower-triangular sub-block after reindexing)
        v     = L[keep, i_del]       (column i_del of L, restricted to kept rows)

    The matrix A_{−i,−i} = L Lᵀ evaluated on the kept rows/columns satisfies:

        A_{−i,−i} = L_red L_redᵀ + v vᵀ

    and its Cholesky factor is obtained by the rank-1 update:

        L_new = cholesky_rank1_update(L_red, v)

    This exploits the fact that removing a column of A is equivalent to a
    rank-1 downdate of the reduced matrix (Seeger 2004).

    Parameters
    ----------
    L     : (m, m) lower-triangular Cholesky factor of A
    i_del : int    index of the row/column to delete from A (0-based)

    Returns
    -------
    L_new : (m-1, m-1) Cholesky factor of  A_{−i_del, −i_del}
    """
    m = L.shape[0]
    if m <= 1:
        return np.zeros((0, 0))

    keep = np.delete(np.arange(m), i_del)
    L_red = np.tril(L[np.ix_(keep, keep)])   # Lower-triangular sub-block
    v = L[keep, i_del]                        # Correction column from deleted index

    return cholesky_rank1_update(L_red, v)

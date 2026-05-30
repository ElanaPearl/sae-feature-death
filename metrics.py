"""
Metrics from "On the Relationship Between Activation Outliers and Feature
Death in Sparse Autoencoders" (Simon, Adams, Zou, 2026).
arxiv.org/abs/2605.31518
"""

import numpy as np


def compute_gamma(acts):
    """
    Compute outlier severity gamma = ||mu|| / ||sigma||.

    Applies per-token LayerNorm * sqrt(d) before computing, matching the
    paper's gamma_postLN convention.

    Args:
        acts: [N, d] numpy array.

    Returns:
        float: gamma.
    """
    x = np.asarray(acts, dtype=np.float64)
    d = x.shape[1]
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    x = (x - mean) / np.sqrt(var + 1e-5) * np.sqrt(d)

    mu = x.mean(axis=0)
    sigma = x.std(axis=0, ddof=0)

    sigma_norm = np.linalg.norm(sigma)
    if sigma_norm < 1e-10:
        return float('inf')

    return float(np.linalg.norm(mu) / sigma_norm)


def effective_rank(acts):
    """
    Effective rank of the activation covariance: exp(entropy of normalized
    eigenvalues). Applies per-token LayerNorm * sqrt(d) first, consistent
    with compute_gamma.

    Args:
        acts: [N, d] numpy array.

    Returns:
        float: effective rank, in [1, d].
    """
    x = np.asarray(acts, dtype=np.float64)
    d = x.shape[1]
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    x = (x - mean) / np.sqrt(var + 1e-5) * np.sqrt(d)
    x = x - x.mean(axis=0)

    _, s, _ = np.linalg.svd(x, full_matrices=False)
    p = s ** 2
    p = p / p.sum()
    p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))


def geometric_median(acts, max_iter=200, tol=1e-6):
    """
    Geometric median via Weiszfeld's algorithm.

    Args:
        acts: [N, d] numpy array.
        max_iter: maximum iterations (default 200).
        tol: convergence tolerance (default 1e-6).

    Returns:
        numpy array of shape [d].
    """
    X = np.asarray(acts, dtype=np.float64)
    y = X.mean(axis=0).copy()
    for _ in range(max_iter):
        dists = np.maximum(np.linalg.norm(X - y, axis=1, keepdims=True), 1e-8)
        weights = 1.0 / dists
        y_new = (X * weights).sum(axis=0) / weights.sum()
        if np.linalg.norm(y_new - y) < tol:
            break
        y = y_new
    return y


def predicted_dead_by_relu(gamma, n_eval=100_000):
    """
    Theoretical fraction of features dead-by-ReLU at init.

        P(dead) = Phi(-C / gamma) * 100

    where C = Phi^{-1}(1 - 1/N).

    Args:
        gamma: scalar or array.
        n_eval: number of eval samples (default 100K).
    """
    from scipy.stats import norm
    gamma = np.asarray(gamma, dtype=np.float64)
    C = norm.ppf(1 - 1 / n_eval)
    return norm.cdf(-C / gamma) * 100


def predicted_dead_by_topk(gamma, n_eval=100_000, k=64, n_features=8192):
    """
    Theoretical fraction of features dead-by-TopK at init (includes ReLU-dead).

        P(dead) = Phi(t_k - C / gamma) * 100

    where t_k = Phi^{-1}(1 - k/n). Asymptotes to (1 - k/n) * 100.

    Args:
        gamma: scalar or array.
        n_eval: number of eval samples (default 100K).
        k: top-k sparsity (default 64).
        n_features: SAE dictionary size (default 8192).
    """
    from scipy.stats import norm
    gamma = np.asarray(gamma, dtype=np.float64)
    C = norm.ppf(1 - 1 / n_eval)
    t_k = norm.ppf(1 - k / n_features)
    return norm.cdf(t_k - C / gamma) * 100


def diagnose(acts):
    """Compute gamma and effective rank together."""
    return {
        'gamma': compute_gamma(acts),
        'effective_rank': effective_rank(acts),
    }

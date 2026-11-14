"""Plug-in estimator and bootstrap confidence intervals for Gaussian BWPT."""

import numpy as np

from .gaussian import Counterfactual, pt_counterfactual


def estimate(X0, X1, X2, **ode_kwargs) -> Counterfactual:
    """Plug-in estimator from samples of control(0), control(1), treated(0),
    each of shape (n_j, d)."""
    args = []
    for X in (X0, X1, X2):
        args += [X.mean(axis=0), np.atleast_2d(np.cov(X, rowvar=False))]
    return pt_counterfactual(*args, **ode_kwargs)


def param_names(d: int) -> list[str]:
    """Names of the scalar parameters, in the order used by ``as_vector``."""
    iu = np.triu_indices(d)
    vec = [f"[{i + 1}]" for i in range(d)]
    mat = [f"[{i + 1},{j + 1}]" for i, j in zip(*iu)]
    return [f"a{s}" for s in vec] + [f"A{s}" for s in mat] + [f"mean{s}" for s in vec] + [f"cov{s}" for s in mat]


def as_vector(cf: Counterfactual) -> np.ndarray:
    """All scalar parameters in one vector: a, upper triangle of A,
    counterfactual mean, upper triangle of counterfactual covariance."""
    iu = np.triu_indices(len(cf.a))
    return np.concatenate([cf.a, cf.A[iu], cf.mean, cf.cov[iu]])


def bootstrap_ci(X0, X1, X2, n_boot: int, level: float, rng: np.random.Generator):
    """Basic bootstrap interval for every scalar parameter.

    Each of the three samples is resampled with replacement, independently.
    With theta_hat the estimate and q*(p) the p-quantile of the bootstrap
    replicates, at level 1 - alpha:

        L = 2 theta_hat - q*(1 - alpha/2)
        U = 2 theta_hat - q*(alpha/2)

    Returns (theta_hat, L, U, bootstrap standard error), each of shape (P,).
    """
    theta_hat = as_vector(estimate(X0, X1, X2))

    draws = np.empty((n_boot, len(theta_hat)))
    for b in range(n_boot):
        resampled = [X[rng.integers(len(X), size=len(X))] for X in (X0, X1, X2)]
        draws[b] = as_vector(estimate(*resampled, rtol=1e-8, atol=1e-10))

    alpha = 1 - level
    q_lo, q_hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
    lower = 2 * theta_hat - q_hi
    upper = 2 * theta_hat - q_lo
    return theta_hat, lower, upper, draws.std(axis=0, ddof=1)

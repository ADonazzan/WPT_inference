"""Plug-in estimation and bootstrap confidence intervals for Gaussian BWPT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .gaussian import Counterfactual, pt_counterfactual

Array = NDArray[np.float64]

TARGETS = ("a", "A", "mean", "cov")


def estimate(X0: ArrayLike, X1: ArrayLike, X2: ArrayLike, **ode_kwargs) -> Counterfactual:
    """Plug-in estimator from samples of control(0), control(1), treated(0),
    each of shape (n_j, d) with n_j > d."""
    Xs = [np.asarray(X, dtype=float) for X in (X0, X1, X2)]
    args = []
    for X in Xs:
        args += [X.mean(axis=0), np.atleast_2d(np.cov(X, rowvar=False))]
    return pt_counterfactual(*args, **ode_kwargs)


@dataclass(frozen=True)
class BootstrapResult:
    estimate: Counterfactual
    level: float
    method: str
    draws: dict[str, Array]  # target -> (n_boot, *shape) bootstrap replicates
    lower: dict[str, Array]  # target -> entrywise lower confidence limits
    upper: dict[str, Array]  # target -> entrywise upper confidence limits
    se: dict[str, Array]  # target -> entrywise bootstrap standard errors

    def covers(self, truth: Counterfactual) -> dict[str, NDArray[np.bool_]]:
        """Entrywise indicator that the interval contains the true value."""
        return {
            k: (self.lower[k] <= getattr(truth, k)) & (getattr(truth, k) <= self.upper[k])
            for k in TARGETS
        }


def bootstrap_ci(
    X0: ArrayLike,
    X1: ArrayLike,
    X2: ArrayLike,
    *,
    n_boot: int = 1000,
    level: float = 0.95,
    method: Literal["percentile", "basic"] = "percentile",
    rng: np.random.Generator | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> BootstrapResult:
    """Entrywise bootstrap confidence intervals for the transported tangent
    (a, A) and the counterfactual mean and covariance.

    Each of the three samples is resampled with replacement independently.
    ``method`` is "percentile" (quantiles of the replicates) or "basic"
    (2 * estimate - reversed quantiles).
    """
    if method not in ("percentile", "basic"):
        raise ValueError(f"unknown method {method!r}")
    rng = np.random.default_rng() if rng is None else rng
    Xs = [np.asarray(X, dtype=float) for X in (X0, X1, X2)]
    est = estimate(*Xs)

    draws: dict[str, list[Array]] = {k: [] for k in TARGETS}
    for _ in range(n_boot):
        resampled = [X[rng.integers(len(X), size=len(X))] for X in Xs]
        rep = estimate(*resampled, rtol=rtol, atol=atol)
        for k in TARGETS:
            draws[k].append(getattr(rep, k))

    alpha = 1.0 - level
    out_draws, lower, upper, se = {}, {}, {}, {}
    for k in TARGETS:
        D = np.stack(draws[k])
        q_lo, q_hi = np.quantile(D, [alpha / 2, 1 - alpha / 2], axis=0)
        if method == "basic":
            theta = getattr(est, k)
            q_lo, q_hi = 2 * theta - q_hi, 2 * theta - q_lo
        out_draws[k], lower[k], upper[k], se[k] = D, q_lo, q_hi, D.std(axis=0, ddof=1)
    return BootstrapResult(est, level, method, out_draws, lower, upper, se)

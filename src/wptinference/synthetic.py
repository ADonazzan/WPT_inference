from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .gaussian import Counterfactual, pt_counterfactual

Array = NDArray[np.float64]


def random_spd(d: int, rng: np.random.Generator, *, cond: float = 4.0) -> Array:
    U, R = np.linalg.qr(rng.standard_normal((d, d)))
    U = U * np.sign(np.diag(R))
    half = 0.5 * np.log(cond)
    w = np.exp(rng.uniform(-half, half, size=d))
    return (U * w) @ U.T


@dataclass(frozen=True)
class GaussianDesign:
    """Population parameters: control at time 0 and 1, treated at time 0."""

    m0: Array
    S0: Array
    m1: Array
    S1: Array
    m2: Array
    S2: Array

    @property
    def d(self) -> int:
        return len(self.m0)

    def truth(self) -> Counterfactual:
        return pt_counterfactual(self.m0, self.S0, self.m1, self.S1, self.m2, self.S2)

    def sample(
        self, n: int | tuple[int, int, int], rng: np.random.Generator
    ) -> tuple[Array, Array, Array]:
        """Three independent samples X0 ~ control(0), X1 ~ control(1),
        X2 ~ treated(0), each of shape (n_j, d)."""
        n0, n1, n2 = (n, n, n) if isinstance(n, int) else n
        return (
            rng.multivariate_normal(self.m0, self.S0, size=n0),
            rng.multivariate_normal(self.m1, self.S1, size=n1),
            rng.multivariate_normal(self.m2, self.S2, size=n2),
        )


def random_design(
    d: int, rng: np.random.Generator, *, cond: float = 4.0, mean_scale: float = 1.0
) -> GaussianDesign:
    """Random design with standard-normal means (times ``mean_scale``) and
    independent random SPD covariances."""
    means = mean_scale * rng.standard_normal((3, d))
    covs = [random_spd(d, rng, cond=cond) for _ in range(3)]
    return GaussianDesign(means[0], covs[0], means[1], covs[1], means[2], covs[2])

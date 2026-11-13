from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

Array = NDArray[np.float64]


def _sym(X: Array) -> Array:
    return 0.5 * (X + X.T)


def spd_sqrt(S: ArrayLike) -> Array:
    """Symmetric square root of a symmetric positive definite matrix."""
    w, U = np.linalg.eigh(_sym(np.asarray(S, dtype=float)))
    if w[0] <= 0:
        raise ValueError(f"matrix is not positive definite (min eigenvalue {w[0]:.3g})")
    return (U * np.sqrt(w)) @ U.T


def brenier_matrix(S0: ArrayLike, S1: ArrayLike) -> Array:
    """Brenier matrix B with B S0 B = S1, i.e. the linear part of the OT map
    N(m0, S0) -> N(m1, S1):  B = S0^{-1/2} (S0^{1/2} S1 S0^{1/2})^{1/2} S0^{-1/2}.
    """
    S0 = np.asarray(S0, dtype=float)
    S1 = np.asarray(S1, dtype=float)
    w, U = np.linalg.eigh(_sym(S0))
    if w[0] <= 0:
        raise ValueError(f"S0 is not positive definite (min eigenvalue {w[0]:.3g})")
    r = (U * np.sqrt(w)) @ U.T
    r_inv = (U / np.sqrt(w)) @ U.T
    return _sym(r_inv @ spd_sqrt(r @ S1 @ r) @ r_inv)


def lyapunov_solve(Q: Array, C: Array) -> Array:
    """Unique X solving X Q + Q X = C for symmetric positive definite Q."""
    lam, U = np.linalg.eigh(Q)
    K = (U.T @ C @ U) / (lam[:, None] + lam[None, :])
    return U @ K @ U.T


def parallel_transport(
    A0: ArrayLike,
    S0: ArrayLike,
    Bg: ArrayLike,
    *,
    t_eval: ArrayLike | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> Array:
    """Parallel transport the linear part A0 of a tangent vector at N(., S0)
    along the geodesic with Brenier matrix Bg (Theorem 3.3).
    """
    A0 = _sym(np.asarray(A0, dtype=float))
    S0 = np.asarray(S0, dtype=float)
    Bg = np.asarray(Bg, dtype=float)
    d = A0.shape[0]
    S0_inv = np.linalg.inv(S0)

    # M_t and S_t are functions of Bg, so they share its eigenvectors.
    beta, V = np.linalg.eigh(_sym(Bg))
    if beta[0] <= 0:
        raise ValueError(f"Bg is not positive definite (min eigenvalue {beta[0]:.3g})")
    W = V.T @ S0_inv @ V  # S0^{-1} in the eigenbasis of Bg

    def rhs(t: float, y: Array) -> Array:
        A = y.reshape(d, d)
        m_inv = 1.0 / (1.0 + t * (beta - 1.0))
        Q = _sym((V * m_inv) @ W @ (V * m_inv).T)
        S = (V * ((1.0 - beta) * m_inv)) @ V.T
        G = Q @ A @ S
        return lyapunov_solve(Q, G + G.T).ravel()

    t_span = (0.0, 1.0)
    if t_eval is not None:
        t_eval = np.asarray(t_eval, dtype=float)
        t_span = (0.0, max(1.0, float(t_eval[-1])))
    sol = solve_ivp(rhs, t_span, A0.ravel(), method="DOP853", t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"parallel transport ODE failed: {sol.message}")
    if t_eval is None:
        return _sym(sol.y[:, -1].reshape(d, d))
    path = sol.y.T.reshape(-1, d, d)
    return 0.5 * (path + path.transpose(0, 2, 1))


@dataclass(frozen=True)
class Counterfactual:
    """Transported tangent v1(x) = a + A (x - m2) at the treated measure
    N(m2, S2), and the counterfactual N(mean, cov) it generates."""

    a: Array  # translation part, m1 - m0
    A: Array  # transported linear part A_1
    m2: Array  # treated mean at time 0
    mean: Array  # counterfactual mean  m2 + a
    cov: Array  # counterfactual covariance (I + A) S2 (I + A)

    @property
    def T(self) -> Array:
        """Linear part I + A of the counterfactual map. The map is the Brenier
        map from the treated measure to the counterfactual iff this is PSD."""
        return np.eye(len(self.a)) + self.A

    @property
    def is_brenier(self) -> bool:
        return bool(np.linalg.eigvalsh(self.T)[0] >= 0)

    def map(self, x: ArrayLike) -> Array:
        """Push treated time-0 points (n x d) to their counterfactual at time 1."""
        x = np.asarray(x, dtype=float)
        return self.mean + (x - self.m2) @ self.T.T


def pt_counterfactual(
    m0: ArrayLike,
    S0: ArrayLike,
    m1: ArrayLike,
    S1: ArrayLike,
    m2: ArrayLike,
    S2: ArrayLike,
    **ode_kwargs,
) -> Counterfactual:
    """Counterfactual for the treated group."""
    m0, m1, m2 = (np.asarray(m, dtype=float) for m in (m0, m1, m2))
    S0, S1, S2 = (np.asarray(S, dtype=float) for S in (S0, S1, S2))
    d = len(m0)
    A0 = brenier_matrix(S0, S1) - np.eye(d)
    Bg = brenier_matrix(S0, S2)
    A1 = parallel_transport(A0, S0, Bg, **ode_kwargs)
    a = m1 - m0
    T = np.eye(d) + A1
    return Counterfactual(a=a, A=A1, m2=m2, mean=m2 + a, cov=_sym(T @ S2 @ T))

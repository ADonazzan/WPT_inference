import numpy as np
import pytest

from wptinference import (
    bootstrap_ci,
    brenier_matrix,
    estimate,
    parallel_transport,
    pt_counterfactual,
    random_design,
    random_spd,
)


@pytest.fixture
def rng():
    return np.random.default_rng(123)


def test_brenier_matrix_pushes_covariance(rng):
    S0, S1 = random_spd(4, rng), random_spd(4, rng)
    B = brenier_matrix(S0, S1)
    assert np.allclose(B, B.T)
    assert np.linalg.eigvalsh(B)[0] > 0
    assert np.allclose(B @ S0 @ B, S1)


def test_trivial_geodesic_leaves_tangent_unchanged(rng):
    S0 = random_spd(3, rng)
    A0 = brenier_matrix(S0, random_spd(3, rng)) - np.eye(3)
    assert np.allclose(parallel_transport(A0, S0, np.eye(3)), A0)


def test_commuting_case_closed_form(rng):
    # Diagonal covariances: A_t = A_0 M_t^{-1}, so A_1 = A_0 Bg^{-1} and the
    # counterfactual std is s2 + s1 - s0 coordinatewise.
    s0, s1, s2 = (rng.uniform(0.5, 2.0, size=3) for _ in range(3))
    s1 = np.maximum(s1, s0 - s2 + 0.1)  # keep the counterfactual std positive
    m = rng.standard_normal((3, 3))
    cf = pt_counterfactual(m[0], np.diag(s0**2), m[1], np.diag(s1**2), m[2], np.diag(s2**2))
    assert np.allclose(cf.A, np.diag((s1 - s0) / s2), atol=1e-8)
    assert np.allclose(cf.cov, np.diag((s2 + s1 - s0) ** 2), atol=1e-8)
    assert np.allclose(cf.mean, m[2] + m[1] - m[0])


def test_parallel_transport_preserves_norm_and_symmetry(rng):
    d = 4
    S0, S1, S2 = (random_spd(d, rng) for _ in range(3))
    A0 = brenier_matrix(S0, S1) - np.eye(d)
    Bg = brenier_matrix(S0, S2)
    ts = np.linspace(0, 1, 11)
    path = parallel_transport(A0, S0, Bg, t_eval=ts)
    assert np.allclose(path, path.transpose(0, 2, 1))
    norms = []
    for t, A in zip(ts, path):
        M = (1 - t) * np.eye(d) + t * Bg
        norms.append(np.trace(A @ M @ S0 @ M @ A))  # ||A_t (x - m_t)||^2 in L2(mu_t)
    assert np.allclose(norms, norms[0], rtol=1e-7)
    assert not np.allclose(path[-1], A0, atol=1e-3)  # the transport is non-trivial


def test_transported_field_solves_lyapunov_equation(rng):
    # Finite-difference check of the ODE in Theorem 3.3 at an interior time.
    d, t, h = 3, 0.4, 1e-5
    S0, S1, S2 = (random_spd(d, rng) for _ in range(3))
    A0 = brenier_matrix(S0, S1) - np.eye(d)
    Bg = brenier_matrix(S0, S2)
    Am, A, Ap = parallel_transport(A0, S0, Bg, t_eval=[t - h, t, t + h])
    dA = (Ap - Am) / (2 * h)
    M_inv = np.linalg.inv((1 - t) * np.eye(d) + t * Bg)
    Q = M_inv.T @ np.linalg.inv(S0) @ M_inv
    S = (np.eye(d) - Bg) @ M_inv
    assert np.allclose(dA @ Q + Q @ dA, S.T @ A @ Q + Q @ A @ S, atol=1e-6)


def test_counterfactual_map_pushes_treated_sample(rng):
    design = random_design(3, rng)
    cf = design.truth()
    X2 = rng.multivariate_normal(design.m2, design.S2, size=200_000)
    Y = cf.map(X2)
    assert np.allclose(Y.mean(axis=0), cf.mean, atol=0.02)
    assert np.allclose(np.cov(Y, rowvar=False), cf.cov, atol=0.05)


def test_estimator_is_consistent(rng):
    design = random_design(2, rng)
    truth = design.truth()
    est = estimate(*design.sample(200_000, rng))
    assert np.allclose(est.A, truth.A, atol=0.03)
    assert np.allclose(est.cov, truth.cov, atol=0.05)


def test_bootstrap_ci_shapes_and_ordering(rng):
    design = random_design(2, rng)
    samples = design.sample(200, rng)
    for method in ("percentile", "basic"):
        res = bootstrap_ci(*samples, n_boot=50, method=method, rng=rng)
        assert res.draws["A"].shape == (50, 2, 2)
        assert res.draws["a"].shape == (50, 2)
        for k in ("a", "A", "mean", "cov"):
            assert np.all(res.lower[k] <= res.upper[k])
            assert np.all(res.se[k] > 0)


def test_coverage_study_writes_outputs(tmp_path):
    from wptinference.study import StudyConfig, load_replications, n_sim_for, run_study

    assert n_sim_for(0.95, 0.005) == 1900
    cfg = StudyConfig(sample_sizes=(100,), n_sim=6, n_boot=30, workers=1)
    out = run_study(cfg, tmp_path / "run")
    for name in ("config.json", "run.log", "summary.csv", "tables.txt", "tables.tex", "coverage.pdf"):
        assert (out / name).stat().st_size > 0
    reps = load_replications(out)
    assert reps.height == 6 * 10  # replications x scalar parameters (d = 2)
    first = reps.clone()
    run_study(cfg, out)  # resuming skips finished sample sizes
    assert load_replications(out).equals(first)

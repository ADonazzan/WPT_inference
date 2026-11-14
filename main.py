"""Monte Carlo coverage of the bootstrap confidence intervals.

    uv run main.py

Each replication draws three samples of size n from a fixed Gaussian design,
computes the plug-in estimate and the basic bootstrap interval, and checks
whether the interval contains the true value.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"  # one BLAS thread per worker process

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from wptinference import as_vector, bootstrap_ci, param_names, random_design

# ----------------------------------------------------------------- setup ----
D = 2                                   # dimension
SAMPLE_SIZES = [250, 500, 1000, 2000]   # n, the same for all three samples
N_SIM = 2000                            # Monte Carlo replications per n
N_BOOT = 1000                           # bootstrap resamples per replication
LEVEL = 0.95                            # nominal coverage
DESIGN_SEED = 0                         # population parameters
SEED = 20261113                         # replications
WORKERS = 8                             # parallel processes
OUT = Path("results/coverage_d2")
# -----------------------------------------------------------------------------


def one_replication(task):
    design, n, seed = task
    rng = np.random.default_rng(seed)
    X0, X1, X2 = design.sample(n, rng)
    return bootstrap_ci(X0, X1, X2, N_BOOT, LEVEL, rng)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(OUT / "run.log", mode="w"), logging.StreamHandler()],
    )
    log = logging.info

    design = random_design(D, np.random.default_rng(DESIGN_SEED))
    truth = as_vector(design.truth())
    names = param_names(D)
    mcse = np.sqrt(LEVEL * (1 - LEVEL) / N_SIM)
    log(f"d={D}  n={SAMPLE_SIZES}  n_sim={N_SIM}  n_boot={N_BOOT}  level={LEVEL}  workers={WORKERS}")
    log(f"Monte Carlo SE of the coverage estimates: {mcse:.4f}")

    replications, summary = [], []
    seeds = np.random.SeedSequence(SEED).spawn(len(SAMPLE_SIZES))
    for n, seed in zip(SAMPLE_SIZES, seeds):
        tasks = [(design, n, s) for s in seed.spawn(N_SIM)]
        results = []
        with ProcessPoolExecutor(WORKERS) as pool:
            for result in pool.map(one_replication, tasks, chunksize=10):
                results.append(result)
                if len(results) % max(1, N_SIM // 10) == 0:
                    log(f"n={n}: {len(results)}/{N_SIM} replications")

        # Each array has shape (N_SIM, number of parameters).
        est, lower, upper, boot_se = (np.array(x) for x in zip(*results))

        replications.append(pl.DataFrame({
            "n": n,
            "rep": np.repeat(np.arange(N_SIM), len(names)),
            "param": names * N_SIM,
            "truth": np.tile(truth, N_SIM),
            "estimate": est.ravel(),
            "lower": lower.ravel(),
            "upper": upper.ravel(),
            "boot_se": boot_se.ravel(),
        }))
        summary.append(pl.DataFrame({
            "n": n,
            "param": names,
            "truth": truth,
            "bias": est.mean(axis=0) - truth,
            "rmse": np.sqrt(((est - truth) ** 2).mean(axis=0)),
            "emp_se": est.std(axis=0, ddof=1),       # SD of the estimates
            "boot_se": boot_se.mean(axis=0),         # mean bootstrap SE
            "coverage": ((lower <= truth) & (truth <= upper)).mean(axis=0),
            "ci_length": (upper - lower).mean(axis=0),
        }))

    pl.concat(replications).write_parquet(OUT / "replications.parquet")
    summary = pl.concat(summary)
    summary.write_csv(OUT / "summary.csv")

    log(f"Results ({N_SIM} replications, {N_BOOT} resamples, nominal {LEVEL}, MC SE {mcse:.4f}):")
    log(f"{'n':>5}  {'param':<9}{'truth':>8}{'bias':>9}{'rmse':>8}{'emp_se':>8}{'boot_se':>9}{'cover':>7}{'length':>8}")
    for r in summary.iter_rows(named=True):
        log(
            f"{r['n']:>5}  {r['param']:<9}{r['truth']:>8.3f}{r['bias']:>+9.4f}{r['rmse']:>8.4f}"
            f"{r['emp_se']:>8.4f}{r['boot_se']:>9.4f}{r['coverage']:>7.3f}{r['ci_length']:>8.3f}"
        )
    log(f"Wrote {OUT}/replications.parquet, summary.csv, run.log")


if __name__ == "__main__":
    main()

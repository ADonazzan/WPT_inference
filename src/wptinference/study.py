"""Monte Carlo coverage study for the bootstrap confidence intervals.

One *replication* draws three fresh samples from a fixed Gaussian design,
computes the plug-in estimate and the bootstrap intervals, and records, for
every scalar parameter, the estimate, the bootstrap standard error, the
interval limits at each nominal level, and the position of the true value in
the bootstrap distribution.

Everything is written to a run directory:

    config.json                 study configuration, design and true values
    run.log                     timestamped progress log
    replications_n{n}.parquet   one row per (replication, parameter)
    summary.csv                 performance measures, long format
    tables.txt / tables.tex     paper-style tables
    coverage.pdf / .png         nominal vs. empirical coverage

Replications are seeded individually from ``seed``, so results do not depend
on the number of worker processes, and a finished sample size is skipped when
the run is resumed.
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from wptinference.bootstrap import TARGETS, bootstrap_ci
from wptinference.gaussian import Counterfactual
from wptinference.synthetic import GaussianDesign, random_design

log = logging.getLogger("wptinference.study")

METHODS = ("percentile", "basic")


@dataclass(frozen=True)
class StudyConfig:
    d: int = 2
    sample_sizes: tuple[int, ...] = (250, 500, 1000, 2000)  # per sample (n0 = n1 = n2)
    n_sim: int = 2000  # Monte Carlo replications per sample size
    n_boot: int = 1000  # bootstrap resamples per replication
    levels: tuple[float, ...] = (0.90, 0.95, 0.99)
    seed: int = 20261113  # replication seeds
    design_seed: int = 0  # random design (population parameters)
    cond: float = 4.0  # max condition number of the design covariances
    workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 2) - 1))

    def mcse_coverage(self, level: float) -> float:
        """Monte Carlo SE of an estimated coverage whose true value is ``level``."""
        return float(np.sqrt(level * (1 - level) / self.n_sim))


def n_sim_for(level: float = 0.95, mcse: float = 0.005) -> int:
    """Replications needed for a coverage estimate with the given Monte Carlo
    SE (Morris, White & Crowther 2019, eq. 1): level (1 - level) / mcse^2."""
    return int(np.ceil(round(level * (1 - level) / mcse**2, 6)))


# --------------------------------------------------------------------------
# Scalar parameters: a_i, A_ij (i <= j), mean_i, cov_ij (i <= j), 1-indexed.
# --------------------------------------------------------------------------


def param_names(d: int) -> list[str]:
    iu = np.triu_indices(d)
    names = []
    for k in TARGETS:
        if k in ("a", "mean"):
            names += [f"{k}[{i + 1}]" for i in range(d)]
        else:
            names += [f"{k}[{i + 1},{j + 1}]" for i, j in zip(*iu)]
    return names


def _flatten(get, d: int) -> np.ndarray:
    """Stack the unique scalar entries of every target along the last axis.
    ``get(k)`` returns an array whose trailing axes are (d,) or (d, d)."""
    iu = np.triu_indices(d)
    parts = []
    for k in TARGETS:
        x = np.asarray(get(k))
        parts.append(x if k in ("a", "mean") else x[..., iu[0], iu[1]])
    return np.concatenate(parts, axis=-1)


def flatten(cf: Counterfactual) -> np.ndarray:
    return _flatten(lambda k: getattr(cf, k), len(cf.a))


# --------------------------------------------------------------------------
# One replication
# --------------------------------------------------------------------------


def _replicate(task: tuple) -> dict[str, np.ndarray]:
    design, truth, n, n_boot, levels, seed_seq = task
    rng = np.random.default_rng(seed_seq)
    d = design.d
    res = bootstrap_ci(*design.sample(n, rng), n_boot=n_boot, rng=rng)
    est = flatten(res.estimate)
    draws = _flatten(lambda k: res.draws[k], d)  # (n_boot, P)

    out = {"estimate": est, "boot_se": draws.std(axis=0, ddof=1)}
    for level in levels:
        alpha = 1 - level
        q_lo, q_hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
        tag = f"{round(level * 100):d}"
        out[f"lo_percentile_{tag}"], out[f"hi_percentile_{tag}"] = q_lo, q_hi
        out[f"lo_basic_{tag}"], out[f"hi_basic_{tag}"] = 2 * est - q_hi, 2 * est - q_lo

    # Mid-rank of the truth in the bootstrap distribution. The level-L interval
    # covers iff (1-L)/2 <= rank <= (1+L)/2, so these give coverage at any level.
    def midrank(x: np.ndarray) -> np.ndarray:
        return ((draws < x).sum(axis=0) + 0.5 * (draws == x).sum(axis=0)) / len(draws)

    out["rank_percentile"] = midrank(truth)
    out["rank_basic"] = midrank(2 * est - truth)
    return out


def _run_sample_size(
    cfg: StudyConfig, design: GaussianDesign, truth: np.ndarray, n: int, seeds: list
) -> pl.DataFrame:
    names = param_names(cfg.d)
    tasks = [(design, truth, n, cfg.n_boot, cfg.levels, s) for s in seeds]
    results: list[dict[str, np.ndarray]] = []
    t0 = time.perf_counter()
    next_report = t0 + 30.0

    def consume(iterator) -> None:
        nonlocal next_report
        for r in iterator:
            results.append(r)
            now = time.perf_counter()
            if now >= next_report and len(results) < cfg.n_sim:
                rate = len(results) / (now - t0)
                log.info(
                    "n=%-5d %5d/%d replications  (%.1f/s, about %s left)",
                    n, len(results), cfg.n_sim, rate, _hms((cfg.n_sim - len(results)) / rate),
                )
                next_report = now + 30.0

    if cfg.workers > 1:
        chunk = max(1, min(25, cfg.n_sim // (4 * cfg.workers)))
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            consume(pool.map(_replicate, tasks, chunksize=chunk))
    else:
        consume(map(_replicate, tasks))

    cols: dict[str, np.ndarray | list] = {
        "n": np.repeat(n, cfg.n_sim * len(names)),
        "rep": np.repeat(np.arange(cfg.n_sim), len(names)),
        "param": names * cfg.n_sim,
        "truth": np.tile(truth, cfg.n_sim),
    }
    for key in results[0]:
        cols[key] = np.concatenate([r[key] for r in results])
    log.info("n=%-5d finished %d replications in %s", n, cfg.n_sim, _hms(time.perf_counter() - t0))
    return pl.DataFrame(cols)


# --------------------------------------------------------------------------
# Run + summarise
# --------------------------------------------------------------------------


def run_study(cfg: StudyConfig, out_dir: Path | str) -> Path:
    """Run (or resume) the study and write all outputs to ``out_dir``."""
    from wptinference.report import write_report  # local import: matplotlib is slow

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(out_dir / "run.log")
    # Tiny matrices: BLAS threads only fight the worker processes.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")

    design = random_design(cfg.d, np.random.default_rng(cfg.design_seed), cond=cfg.cond)
    truth_cf = design.truth()
    truth = flatten(truth_cf)

    config_path = out_dir / "config.json"
    payload = {
        "config": asdict(cfg),
        "design": {k: np.asarray(v).tolist() for k, v in asdict(design).items()},
        "truth": dict(zip(param_names(cfg.d), truth.tolist())),
        "counterfactual_map_is_brenier": truth_cf.is_brenier,
    }
    if config_path.exists():
        old = json.loads(config_path.read_text())["config"]
        new = json.loads(json.dumps(payload["config"]))
        differing = [k for k in new if k not in ("workers", "sample_sizes") and old.get(k) != new[k]]
        if differing:
            raise SystemExit(
                f"{out_dir} holds a run with different settings ({', '.join(differing)}); "
                "use a new output directory."
            )
    config_path.write_text(json.dumps(payload, indent=2))

    log.info("Coverage study -> %s", out_dir)
    log.info(
        "d=%d  n in %s  n_sim=%d  n_boot=%d  levels=%s  workers=%d",
        cfg.d, list(cfg.sample_sizes), cfg.n_sim, cfg.n_boot, list(cfg.levels), cfg.workers,
    )
    log.info(
        "Monte Carlo SE of a coverage estimate: %s",
        ", ".join(f"{cfg.mcse_coverage(lv) * 100:.2f} pp at {lv:.0%}" for lv in cfg.levels),
    )
    if not truth_cf.is_brenier:
        log.warning("I + A1 is not PSD in this design: the counterfactual map is not a Brenier map.")

    # Seeds are spawned per (sample size, replication), independent of workers.
    size_seeds = np.random.SeedSequence(cfg.seed).spawn(len(cfg.sample_sizes))
    for n, ss in zip(cfg.sample_sizes, size_seeds):
        path = out_dir / f"replications_n{n}.parquet"
        if path.exists():
            log.info("n=%-5d already done, skipping (%s)", n, path.name)
            continue
        df = _run_sample_size(cfg, design, truth, n, ss.spawn(cfg.n_sim))
        df.write_parquet(path)

    reps = load_replications(out_dir)
    summary = summarise(reps, cfg.levels)
    summary.write_csv(out_dir / "summary.csv")
    write_report(out_dir, cfg, reps, summary)
    log.info("Wrote summary.csv, tables.txt, tables.tex, coverage.pdf, coverage.png")
    return out_dir


def load_replications(out_dir: Path | str) -> pl.DataFrame:
    files = sorted(Path(out_dir).glob("replications_n*.parquet"))
    if not files:
        raise FileNotFoundError(f"no replications_n*.parquet in {out_dir}")
    return pl.concat([pl.read_parquet(f) for f in files]).sort("n", "rep")


def load_config(out_dir: Path | str) -> StudyConfig:
    raw = json.loads((Path(out_dir) / "config.json").read_text())["config"]
    raw["sample_sizes"] = tuple(raw["sample_sizes"])
    raw["levels"] = tuple(raw["levels"])
    return StudyConfig(**raw)


def summarise(reps: pl.DataFrame, levels: tuple[float, ...]) -> pl.DataFrame:
    """Performance measures per (n, parameter, method, level).

    bias, med_bias, rmse, emp_se (SD of the estimates), boot_se (mean bootstrap
    SE), coverage with its Monte Carlo SE, and mean interval length.
    """
    err = pl.col("estimate") - pl.col("truth")
    point = reps.group_by("n", "param", maintain_order=True).agg(
        truth=pl.col("truth").first(),
        n_sim=pl.len(),
        bias=err.mean(),
        med_bias=err.median(),
        rmse=(err**2).mean().sqrt(),
        emp_se=pl.col("estimate").std(),
        boot_se=pl.col("boot_se").mean(),
    )
    frames = []
    for method in METHODS:
        for level in levels:
            tag = f"{round(level * 100):d}"
            lo, hi = pl.col(f"lo_{method}_{tag}"), pl.col(f"hi_{method}_{tag}")
            frames.append(
                reps.group_by("n", "param", maintain_order=True)
                .agg(
                    coverage=((lo <= pl.col("truth")) & (pl.col("truth") <= hi)).mean(),
                    ci_length=(hi - lo).mean(),
                )
                .with_columns(method=pl.lit(method), level=pl.lit(level))
            )
    cover = pl.concat(frames)
    out = cover.join(point, on=["n", "param"], maintain_order="left")
    return out.with_columns(
        coverage_mcse=(pl.col("coverage") * (1 - pl.col("coverage")) / pl.col("n_sim")).sqrt(),
        bias_mcse=pl.col("emp_se") / pl.col("n_sim").sqrt(),
    ).select(
        "n", "param", "method", "level", "truth", "n_sim", "bias", "bias_mcse", "med_bias",
        "rmse", "emp_se", "boot_se", "coverage", "coverage_mcse", "ci_length",
    )


# --------------------------------------------------------------------------


def _hms(seconds: float) -> str:
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def _setup_logging(path: Path) -> None:
    log.setLevel(logging.INFO)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
    file_h = logging.FileHandler(path)
    file_h.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S"))
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    log.addHandler(file_h)
    log.addHandler(console)

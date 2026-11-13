"""Command line interface.

    wptinference demo                      one dataset: estimate and intervals
    wptinference coverage --out DIR        Monte Carlo coverage study
    wptinference report DIR                rebuild tables and figure for a run
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np


def _demo(args: argparse.Namespace) -> None:
    from wptinference.bootstrap import bootstrap_ci
    from wptinference.study import _flatten, flatten, param_names
    from wptinference.synthetic import random_design

    rng = np.random.default_rng(args.seed)
    design = random_design(args.d, rng)
    truth = flatten(design.truth())
    res = bootstrap_ci(*design.sample(args.n, rng), n_boot=args.n_boot, level=args.level, rng=rng)
    est = flatten(res.estimate)
    se = _flatten(lambda k: res.se[k], args.d)
    lo = _flatten(lambda k: res.lower[k], args.d)
    hi = _flatten(lambda k: res.upper[k], args.d)

    pct = f"{args.level:.0%}"
    rows = [
        (name, f"{t:.3f}", f"{e:.3f}", f"{s:.3f}", f"[{l:.3f}, {h:.3f}]", "yes" if l <= t <= h else "NO")
        for name, t, e, s, l, h in zip(param_names(args.d), truth, est, se, lo, hi)
    ]
    headers = ("Parameter", "Truth", "Estimate", "Boot.SE", f"{pct} CI", "Covers")
    widths = [max(len(h), *(len(r[j]) for r in rows)) for j, h in enumerate(headers)]

    def line(items) -> str:
        return "  ".join(
            s.ljust(w) if j in (0, 5) else s.rjust(w) for j, (s, w) in enumerate(zip(items, widths))
        )

    print(f"One synthetic dataset: d = {args.d}, n = {args.n} per sample, "
          f"{args.n_boot} bootstrap resamples, {res.method} intervals.")
    print(line(headers))
    print("-" * len(line(headers)))
    for r in rows:
        print(line(r))
    print(f"\nCounterfactual map is a Brenier map (I + A PSD): {'yes' if res.estimate.is_brenier else 'no'}")


def _coverage(args: argparse.Namespace) -> None:
    from wptinference.study import StudyConfig, run_study

    cfg = StudyConfig()
    if args.quick:
        cfg = replace(cfg, sample_sizes=(200, 400), n_sim=100, n_boot=200)
    overrides = {
        "d": args.d, "sample_sizes": tuple(args.n) if args.n else None, "n_sim": args.n_sim,
        "n_boot": args.n_boot, "levels": tuple(args.levels) if args.levels else None,
        "seed": args.seed, "design_seed": args.design_seed, "workers": args.workers,
    }
    cfg = replace(cfg, **{k: v for k, v in overrides.items() if v is not None})
    out = run_study(cfg, args.out)
    print()
    print((out / "tables.txt").read_text())
    print(f"Outputs in {out}/")


def _report(args: argparse.Namespace) -> None:
    from wptinference.report import write_report
    from wptinference.study import load_config, load_replications, summarise

    out = Path(args.dir)
    cfg = load_config(out)
    reps = load_replications(out)
    cfg = replace(cfg, sample_sizes=tuple(reps["n"].unique(maintain_order=True).to_list()))
    summary = summarise(reps, cfg.levels)
    summary.write_csv(out / "summary.csv")
    print(write_report(out, cfg, reps, summary))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="wptinference", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="estimate and intervals on one synthetic dataset")
    p.add_argument("--d", type=int, default=2)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--level", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_demo)

    p = sub.add_parser("coverage", help="Monte Carlo coverage study (resumable)")
    p.add_argument("--out", required=True, help="output directory for this run")
    p.add_argument("--d", type=int, help="dimension (default 2)")
    p.add_argument("--n", type=int, nargs="+", help="sample sizes (default 250 500 1000 2000)")
    p.add_argument("--n-sim", type=int, help="Monte Carlo replications per sample size (default 2000)")
    p.add_argument("--n-boot", type=int, help="bootstrap resamples per replication (default 1000)")
    p.add_argument("--levels", type=float, nargs="+", help="nominal levels (default 0.90 0.95 0.99)")
    p.add_argument("--seed", type=int, help="seed for the replications")
    p.add_argument("--design-seed", type=int, help="seed for the population parameters")
    p.add_argument("--workers", type=int, help="worker processes (default: CPUs - 1)")
    p.add_argument("--quick", action="store_true", help="small smoke run: n in {200, 400}, 100 x 200")
    p.set_defaults(func=_coverage)

    p = sub.add_parser("report", help="rebuild tables and figure from a finished run")
    p.add_argument("dir")
    p.set_defaults(func=_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

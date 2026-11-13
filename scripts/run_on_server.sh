#!/usr/bin/env bash
# scripts/run_on_server.sh [run-name] [extra `wptinference coverage` options]
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:-coverage_d2}"
if [ "$#" -gt 0 ]; then shift; fi
WORKERS="${WORKERS:-8}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export UV_LINK_MODE=copy

exec nice -n 19 uv run --frozen --no-dev wptinference coverage \
    --out "results/${NAME}" --workers "${WORKERS}" "$@"

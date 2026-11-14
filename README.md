# wptinference

Setup parameters are at the top of `main.py`.

```bash
uv sync
uv run main.py
```

## Leviathan

```bash
git pull && uv sync --frozen --no-dev
tmux new -s wpt
nice -n 19 uv run --no-dev main.py
```

Results (from your machine):

```bash
rsync -av ANDREW_ID@leviathan6.stat.cmu.edu:WPT_inference/results/ results/
```

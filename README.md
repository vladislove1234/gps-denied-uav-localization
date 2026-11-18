# gps-denied-uav-localization

GPS-denied UAV localization: estimating drone position by matching onboard
camera imagery against pre-existing satellite imagery.

Private repository. The Python package imports as `uavloc`.

## Setup

```bash
conda env create -f environment.yml
conda activate uavloc
pip install -e .
nbstripout --install          # keeps notebook diffs readable
```

## Working method

An **experiment** is a question; a **run** is one attempt at it. Every training
run appends a line to `experiments/<exp>/runs.jsonl`, which records the config,
the metrics, what changed since the previous run, and enough provenance to
reproduce it.

```python
from uavloc.track import log_run, set_seed

set_seed(42)
# ... train ...
log_run(
    exp="003-backbone-comparison",
    config={"backbone": "resnet34", "lr": 1e-4, "dataset": "v6_renamed"},
    metrics={"recall@1": 0.72, "val_loss": val_hist},
    note="baseline",
    figures=[fig],
    seed=42,
)
```

Then `python scripts/report.py 003-backbone-comparison` regenerates the run
table and `experiments/INDEX.md`.

Start at **[experiments/INDEX.md](experiments/INDEX.md)**.

## Layout

```
uavloc/          importable package — all reusable logic
experiments/     one folder per question; notebook + proposal + runs + report
notebooks/       scratch exploration
scripts/         report generation
data -> ../data  symlink; data is never committed
```

## Prior work

Frozen, read-only, for reference:

- `../archive/uav_localization_2025` — the 2025 repo (18 notebooks)
- `../archive/uav_geo_localisation_2024` — the original 2024 experiments


```bash
gh repo create gps-denied-uav-localization --private --source=. --remote=origin
git push -u origin master
```

"""Experiment run tracking.

One call at the end of a training cell turns an interactive notebook run into a
durable, comparable record:

    from uavloc.track import log_run, set_seed

    set_seed(42)
    ...
    log_run(
        exp="003-backbone-comparison",
        config={"backbone": "resnet34", "lr": 1e-4, "margin": 0.3},
        metrics={"recall@1": 0.72, "val_loss": val_hist},
        note="baseline",
        figures=[fig],
    )

Every model training also logs to TensorBoard.
All runs, experiments and PoCs alike, go under one root so they can be overlaid:

    writer = tensorboard_writer("003-backbone-comparison")    # or "pocs/<slug>"
    ...
    log_history(writer, {"step": step, "train_loss": loss, "val_loss": val})
    log_images(writer, {"val/segmentation": grid}, step)       # (H, W, 3) uint8
    ...
    log_run(..., writer=writer)        # or log_hparams(writer, config, metrics)

    conda activate uavloc && $CONDA_PREFIX/bin/tensorboard --logdir tensorboard   # from src/

Design rules, in order of importance:

1. ``runs.jsonl`` is append-only. Nothing ever rewrites a past run.
2. What changed between runs is *computed*, never typed. You cannot forget.
3. Failed runs are recorded too. Dropping them makes the log lie by omission.
4. When code changed between runs, the delta is not attributable to config
   alone, and the record says so.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

if TYPE_CHECKING:
    import pandas as pd
    from torch.utils.tensorboard import SummaryWriter

__all__ = ["log_run", "load_runs", "set_seed", "experiment_dir", "tensorboard_writer", "log_history", "log_images",
           "log_hparams", "load_scalars", "REPO_ROOT", "TENSORBOARD_ROOT", "LOG_EVERY_STEPS"]

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO_ROOT / "experiments"
TENSORBOARD_ROOT = REPO_ROOT / "tensorboard"      # every training's logs, one --logdir; gitignored
SPLIT_PREFIXES = ("train", "val", "holdout")      # "val_ce" -> TensorBoard tag "val/ce"
LOG_EVERY_STEPS = 100                             # default log point for every training: curves, val, grids
SCALAR_COLUMNS = ("run", "tag", "step", "value")
UNLIMITED_EVENTS = 0                              # EventAccumulator size_guidance: keep every event
HPARAM_METRIC_PREFIX = "hparam/"                  # keeps final metrics apart from the training curves


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #

def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def _git_state() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        # A dirty tree means the recorded commit does not fully describe the
        # code that ran. Surfaced so a suspicious delta can be explained.
        "dirty": bool(status) if status is not None else None,
    }


def resolve_device(name: str = "auto") -> str:
    """``"auto"`` -> ``cuda``, else ``mps``, else ``cpu``; any other name is returned as given."""
    if name != "auto":
        return name
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _device(explicit: str | None = None) -> str:
    """The device the run actually used.

    Detection alone reports what torch *could* use, not what the run *did* —
    a notebook that hardcodes ``device="cpu"`` on a machine where MPS is
    available would otherwise be recorded as an MPS run. Always pass the
    device you actually trained on; detection is only a fallback.
    """
    if explicit:
        return str(explicit)
    try:
        import torch
    except ImportError:
        return "unknown (torch not importable)"
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)} (detected)"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps (detected)"
    return "cpu (detected)"


def _notebook_path(exp: str, explicit: str | Path | None = None) -> Path | None:
    """Resolve the notebook that produced this run.

    Convention first: ``experiments/<exp>/train.ipynb``. Papermill does NOT
    export the input path into the kernel (verified — it injects parameters
    only), and Jupyter's own hints are unreliable, so the convention is what
    actually works. Pass ``notebook=`` explicitly for anything else.
    """
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = experiment_dir(exp, create=False) / p
        return p if p.exists() else None

    default = experiment_dir(exp, create=False) / "train.ipynb"
    if default.exists():
        return default

    hint = os.environ.get("JPY_SESSION_NAME")
    if hint and hint.endswith(".ipynb") and Path(hint).exists():
        return Path(hint)
    return None


def _code_hash(nb: Path | None) -> str | None:
    """Hash of a notebook's code cells, ignoring outputs and execution counts.

    This is what catches the silent failure mode: you edit the model class
    between runs, the config diff shows nothing, and the metric moves anyway.
    """
    if nb is None:
        return None
    try:
        doc = json.loads(nb.read_text())
    except Exception:
        return None
    src = "\n".join(
        "".join(c.get("source", ""))
        for c in doc.get("cells", [])
        if c.get("cell_type") == "code"
    )
    return hashlib.sha256(src.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# config diffing
# --------------------------------------------------------------------------- #

def _flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, Mapping):
            flat.update(_flatten(v, f"{key}."))
        else:
            flat[key] = v
    return flat


def _diff_config(prev: Mapping[str, Any] | None,
                 curr: Mapping[str, Any]) -> dict[str, Any]:
    """Compute what changed. Returns {} for the first run of an experiment."""
    if prev is None:
        return {}
    # prev was read back from JSON; round-trip curr the same way so a tuple equals its list
    curr = json.loads(json.dumps(curr, default=str))
    a, b = _flatten(prev), _flatten(curr)
    changed: dict[str, Any] = {}
    for k in sorted(set(a) | set(b)):
        if a.get(k, "<unset>") != b.get(k, "<unset>"):
            changed[k] = {"from": a.get(k, "<unset>"), "to": b.get(k, "<unset>")}
    return changed


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Seed every RNG in play. Call before building data loaders or models."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


# --------------------------------------------------------------------------- #
# storage
# --------------------------------------------------------------------------- #

def experiment_dir(exp: str, create: bool = True) -> Path:
    d = EXPERIMENTS / exp
    if create:
        (d / "figures").mkdir(parents=True, exist_ok=True)
        (d / "artifacts").mkdir(parents=True, exist_ok=True)
    return d


def load_runs(exp: str) -> list[dict[str, Any]]:
    """Every run recorded for an experiment, oldest first."""
    f = experiment_dir(exp, create=False) / "runs.jsonl"
    if not f.exists():
        return []
    runs = []
    for line in f.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # never let one corrupt line hide the rest
    return runs


def _next_run_id(exp: str, runs: list[dict[str, Any]]) -> str:
    stem = exp.split("-", 1)[0]          # "003-backbone-comparison" -> "003"
    return f"{stem}.{len(runs) + 1}"


# --------------------------------------------------------------------------- #
# tensorboard
# --------------------------------------------------------------------------- #

def tensorboard_writer(name: str) -> SummaryWriter:
    """Writer for one training run, in ``tensorboard/<name>/<run>``.

    ``name`` is an experiment slug (``"019-building-gap-loss"``) or any other
    group such as ``"pocs/<slug>"``. For an experiment, ``<run>`` starts with the
    id ``log_run`` will assign next, so curves line up with ``runs.jsonl``; the
    UTC timestamp keeps dry runs from sharing a directory with the logged run.
    """
    from torch.utils.tensorboard import SummaryWriter

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    is_experiment = (EXPERIMENTS / name / "proposal.md").exists() or (EXPERIMENTS / name / "runs.jsonl").exists()
    run = f"{_next_run_id(name, load_runs(name))}_{stamp}" if is_experiment else stamp
    return SummaryWriter(log_dir=str(TENSORBOARD_ROOT / name / run))


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _tag(key: str) -> str:
    split, _, rest = key.partition("_")
    return f"{split}/{rest}" if split in SPLIT_PREFIXES and rest else key


def log_history(writer: SummaryWriter | None, entry: Mapping[str, Any], step_key: str = "step") -> None:
    """Write every scalar of one history entry at ``entry[step_key]``; ``val_ce`` becomes tag ``val/ce``."""
    if writer is None:
        return
    step = int(entry[step_key])
    for k, v in entry.items():
        if k != step_key and _is_scalar(v):
            writer.add_scalar(_tag(k), v, step)


def log_images(writer: SummaryWriter | None, images: Mapping[str, Any], step: int) -> None:
    """``(H, W, 3)`` uint8 images, one tag each; TensorBoard's step slider scrubs through them."""
    if writer is None:
        return
    for tag, image in images.items():
        writer.add_image(tag, image, step, dataformats="HWC")


def log_hparams(writer: SummaryWriter | None, config: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    """Config and scalar metrics into the HParams tab, in the run's own directory, then close the writer.

    ``log_run(writer=)`` calls this; call it directly for trainings without a ``runs.jsonl`` (PoCs).
    """
    if writer is None:
        return
    hparams = {k: v if isinstance(v, (bool, int, float, str)) else str(v)
               for k, v in _flatten(json.loads(json.dumps(config, default=str))).items()}
    writer.add_hparams(hparams, {f"{HPARAM_METRIC_PREFIX}{k}": v for k, v in metrics.items() if _is_scalar(v)},
                       run_name=".")
    writer.close()


def load_scalars(run_dirs: Iterable[str | Path], tags: Iterable[str], cache: Path | None = None,
                 refresh: bool = False) -> pd.DataFrame:
    """Scalars of the given runs as a long table with columns ``run, tag, step, value``.

    A relative run dir is taken under ``TENSORBOARD_ROOT``; ``run`` is that relative path. A tag a run never
    logged gives no rows. With ``cache`` (a CSV path) the table is read from it when it exists, else written
    to it, so a figure can be redrawn without the event files; ``refresh`` re-reads them.
    """
    import pandas as pd
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    if cache is not None and cache.exists() and not refresh:
        return pd.read_csv(cache)
    tags = list(tags)
    rows = []
    for run_dir in run_dirs:
        path = TENSORBOARD_ROOT / run_dir
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": UNLIMITED_EVENTS})
        accumulator.Reload()
        logged = set(accumulator.Tags()["scalars"])
        rows += [(str(run_dir), tag, event.step, event.value)
                 for tag in tags if tag in logged for event in accumulator.Scalars(tag)]
    table = pd.DataFrame(rows, columns=list(SCALAR_COLUMNS))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(cache, index=False)
    return table


def _relative(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _save_figures(figures: Iterable[Any], exp: str, run_id: str) -> list[str]:
    out = experiment_dir(exp) / "figures" / f"run_{run_id.replace('.', '_')}"
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, fig in enumerate(figures, 1):
        path = out / f"fig_{i:02d}.png"
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            saved.append(str(path.relative_to(experiment_dir(exp))))
        except Exception as e:                       # never lose a run to a plot
            saved.append(f"<failed: {type(e).__name__}: {e}>")
    return saved


# --------------------------------------------------------------------------- #
# the entry point
# --------------------------------------------------------------------------- #

def log_run(
    exp: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    note: str = "",
    figures: Iterable[Any] | None = None,
    status: str = "ok",
    repairs: list[Mapping[str, Any]] | None = None,
    seed: int | None = None,
    checkpoint: str | None = None,
    device: str | None = None,
    notebook: str | Path | None = None,
    writer: SummaryWriter | None = None,
) -> dict[str, Any]:
    """Append one run to ``experiments/<exp>/runs.jsonl``.

    Args:
        exp: Experiment slug, e.g. ``"003-backbone-comparison"``.
        config: Everything needed to reproduce. Diffed against the previous run.
        metrics: Results. Scalars are compared across runs; lists (per-epoch
            histories) are stored but not diffed.
        note: Why you made this change. One line.
        figures: Matplotlib figures, written to ``figures/run_<id>/``.
        status: ``"ok"`` or ``"failed"``. Failed runs are recorded, not dropped.
        repairs: Fixes applied to get the run to complete, if any.
        seed: Recorded for reproducibility.
        checkpoint: Path to the saved model, if any.
        device: The device actually trained on, e.g. ``"mps"`` or ``"cpu"``.
            Pass it — detection cannot tell what the notebook chose, and
            results are not comparable across devices (different RNG streams).
        notebook: Source notebook. Defaults to ``experiments/<exp>/train.ipynb``.
        writer: The run's ``tensorboard_writer``. Gets config and scalar
            metrics as hparams, is closed, and its directory is recorded.

    Returns:
        The run record, including the computed ``changed`` diff.
    """
    runs = load_runs(exp)
    prev = runs[-1] if runs else None
    run_id = _next_run_id(exp, runs)
    nb = _notebook_path(exp, notebook)
    code_hash = _code_hash(nb)
    dev = _device(device)

    prev_hash = prev.get("code_hash") if prev else None
    code_changed = (
        None if (code_hash is None or prev_hash is None)
        else code_hash != prev_hash
    )

    record: dict[str, Any] = {
        "run": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "note": note,
        "config": dict(config),
        "changed": _diff_config(prev.get("config") if prev else None, config),
        "metrics": dict(metrics),
        "seed": seed,
        "checkpoint": checkpoint,
        "figures": _save_figures(figures, exp, run_id) if figures else [],
        "repairs": list(repairs) if repairs else [],
        "code_hash": code_hash,
        "code_changed": code_changed,
        "tensorboard": _relative(Path(writer.log_dir)) if writer else None,
        "notebook": str(nb.relative_to(REPO_ROOT)) if nb and nb.is_relative_to(REPO_ROOT) else (str(nb) if nb else None),
        "git": _git_state(),
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "device": dev,
        },
    }

    d = experiment_dir(exp)
    with (d / "runs.jsonl").open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    log_hparams(writer, record["config"], record["metrics"])

    _print_summary(record, prev)
    return record


def _print_summary(record: dict[str, Any], prev: dict[str, Any] | None) -> None:
    """Immediate feedback in the notebook, so the cell still feels interactive."""
    print(f"\n[{record['run']}] {record['status']}  {record['timestamp']}")
    if record["note"]:
        print(f"  note: {record['note']}")

    if record["changed"]:
        print("  changed:")
        for k, v in record["changed"].items():
            print(f"    {k}: {v['from']} -> {v['to']}")
    elif prev is not None:
        print("  changed: nothing in config")
    else:
        print("  baseline run")

    prev_metrics = prev.get("metrics", {}) if prev else {}
    for k, v in record["metrics"].items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            pv = prev_metrics.get(k)
            if isinstance(pv, (int, float)) and not isinstance(pv, bool):
                print(f"  {k}: {v:.4g}  (Δ {v - pv:+.4g})")
            else:
                print(f"  {k}: {v:.4g}")

    if record["code_changed"]:
        print("  ⚠ code changed since the previous run — delta is not "
              "attributable to config alone")
    if record["code_hash"] is None:
        print("  ⚠ notebook not found — code changes will go undetected. "
              "Pass notebook= or name it experiments/<exp>/train.ipynb")

    prev_dev = (prev.get("env", {}) or {}).get("device") if prev else None
    if prev_dev and prev_dev != record["env"]["device"]:
        print(f"  ⚠ device changed ({prev_dev} -> {record['env']['device']}) — "
              "RNG streams differ across devices, so this delta is not "
              "comparable to the previous run")
    if "(detected)" in record["env"]["device"]:
        print(f"  ⚠ device was detected, not declared ({record['env']['device']}) "
              "— pass device= to record what you actually trained on")
    if len(record["changed"]) > 1:
        print(f"  ⚠ {len(record['changed'])} config variables changed at once — "
              "delta is confounded")
    if record["git"]["dirty"]:
        print("  ⚠ working tree is dirty — commit before trusting this record")

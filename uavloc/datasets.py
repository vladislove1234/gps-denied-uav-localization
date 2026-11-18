"""Dataset registration and provenance.

An experiment asks a question; a dataset is *equipment* — it either exists and
is sound, or it does not. There is no bar to miss, so datasets do not belong in
``runs.jsonl``. What they do need is an answer to "is this still the same data
it was last week?", because ``data/`` is gitignored and git cannot answer it.

    from uavloc.datasets import dataset_ref

    log_run(exp="002-semantic-vs-rgb",
            config={"dataset": dataset_ref("seg-masks"), ...})

``dataset_ref`` returns ``"seg-masks@a3f9c2"``. Rebuild the data with a
different threshold and that string changes, so ``track.py``'s config diff
reports it like any other changed variable — instead of an unexplained jump in
the metric. This is the same guard ``code_hash`` provides for notebooks.

Layout:

    datasets/<name>/card.md        datasheet, hand-written, tracked
    datasets/<name>/build.ipynb    the build, tracked
    datasets/<name>/manifest.json  generated, tracked
    data/derived/<name>/           the actual bytes, gitignored
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

# _git_state is deliberately shared rather than duplicated: a dataset's build
# commit and a run's commit must be recorded the same way to be comparable.
from uavloc.track import REPO_ROOT, _git_state

__all__ = [
    "DATASETS", "DERIVED",
    "dataset_dir", "data_root", "write_manifest", "load_manifest",
    "dataset_ref", "verify_dataset", "list_datasets",
]

DATASETS = REPO_ROOT / "datasets"          # recipes + manifests, tracked
DERIVED = REPO_ROOT / "data" / "derived"   # the bytes, gitignored


def dataset_dir(name: str, create: bool = False) -> Path:
    d = DATASETS / name
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def data_root(name: str, create: bool = False) -> Path:
    """Where a dataset's actual files live. Never committed."""
    d = DERIVED / name
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# hashing
# --------------------------------------------------------------------------- #

def _iter_files(root: Path) -> Iterator[Path]:
    """Every real file under root, sorted, skipping dotfiles.

    The dotfile skip is not cosmetic: macOS scatters `.DS_Store` through these
    trees, and letting it into the hash would make the dataset appear to change
    every time a folder is opened in Finder.
    """
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            yield p


def _hashes(root: Path) -> dict[str, Any]:
    """Two hashes with different cost/strength trade-offs.

    ``content`` covers every byte — authoritative, but re-reads the whole
    dataset. ``quick`` covers relative paths and sizes only, from stat() calls,
    so it is effectively free and still catches the realistic accidents: files
    added, deleted, renamed, truncated, or regenerated at a different size.
    """
    full = hashlib.sha256()
    quick = hashlib.sha256()
    n_files = 0
    total = 0

    for p in _iter_files(root):
        rel = str(p.relative_to(root)).encode()
        size = p.stat().st_size
        quick.update(rel + b"\0" + str(size).encode() + b"\0")
        full.update(rel + b"\0")
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                full.update(chunk)
        n_files += 1
        total += size

    return {
        "content": full.hexdigest(),
        "quick": quick.hexdigest(),
        "n_files": n_files,
        "bytes": total,
    }


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #

def write_manifest(
    name: str,
    *,
    version: str,
    description: str,
    source: str,
    config: Mapping[str, Any],
    samples: int | None = None,
    checks: Mapping[str, Any] | None = None,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Record what was built, from what, and how good it turned out.

    Args:
        name: Dataset slug, e.g. ``"seg-masks"``. One folder per *definition* —
            never ``_v2_small_easy``. Version lives here, not in the path.
        version: Bumped by hand when the build recipe changes meaningfully.
        description: One line. What is in here.
        source: What this derives from — a path, another dataset ref, a log file.
        config: Build parameters. Everything needed to reproduce these bytes.
        samples: Logical sample count, if it differs from the file count
            (4137 pairs is 8274 files).
        checks: Quality evidence — empty-mask rate, spot-check agreement, class
            histogram. A dataset has no metric to optimise, but when an
            experiment comes back null you must be able to rule the data out as
            the cause. That is only possible if this was written down.
        root: Where the files are. Defaults to ``data/derived/<name>``.

    Returns:
        The manifest, also written to ``datasets/<name>/manifest.json``.
    """
    r = Path(root) if root else data_root(name)
    if not r.exists():
        raise FileNotFoundError(
            f"{r} does not exist — build the data before writing a manifest"
        )

    h = _hashes(r)
    if h["n_files"] == 0:
        # A manifest describing nothing is worse than no manifest: it makes an
        # empty dataset look registered and verified.
        raise ValueError(f"{r} contains no files — nothing to register")

    manifest = {
        "name": name,
        "version": version,
        "description": description,
        "source": source,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(r.relative_to(REPO_ROOT)) if r.is_relative_to(REPO_ROOT) else str(r),
        "n_files": h["n_files"],
        "samples": samples,
        "bytes": h["bytes"],
        "hash": h["content"],
        "quick_hash": h["quick"],
        "config": dict(config),
        "checks": dict(checks) if checks else {},
        "git": _git_state(),
    }

    d = dataset_dir(name, create=True)
    (d / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n"
    )

    print(f"[{name}@{manifest['hash'][:6]}] v{version}")
    print(f"  {h['n_files']} files, {h['bytes'] / 1e6:.1f} MB"
          + (f", {samples} samples" if samples else ""))
    if not checks:
        print("  ⚠ no quality checks recorded — a null experiment result will "
              "not be separable from bad data")
    else:
        for k, v in checks.items():
            print(f"  {k}: {v}")
    return manifest


def load_manifest(name: str) -> dict[str, Any]:
    f = dataset_dir(name) / "manifest.json"
    if not f.exists():
        raise FileNotFoundError(
            f"no manifest for '{name}' — run datasets/{name}/build.ipynb first"
        )
    return json.loads(f.read_text())


def verify_dataset(name: str, deep: bool = False) -> dict[str, Any]:
    """Check the data on disk still matches its manifest.

    ``deep=True`` re-reads every byte. The default stat-only check is fast
    enough to run at the top of every training notebook.
    """
    m = load_manifest(name)
    r = REPO_ROOT / m["root"]
    if not r.exists():
        return {"ok": False, "reason": f"{m['root']} is missing"}

    h = _hashes(r) if deep else None
    if deep:
        if h["content"] != m["hash"]:
            return {"ok": False, "reason": "file contents differ from manifest"}
        return {"ok": True, "reason": "content hash matches"}

    # stat-only pass, mirroring the quick hash
    quick = hashlib.sha256()
    n = 0
    for p in _iter_files(r):
        quick.update(str(p.relative_to(r)).encode() + b"\0"
                     + str(p.stat().st_size).encode() + b"\0")
        n += 1
    if quick.hexdigest() != m["quick_hash"]:
        if n != m["n_files"]:
            why = f"{n} files on disk, {m['n_files']} recorded"
        else:
            # Same count, different stat fingerprint: renamed or resized.
            # Saying "sizes differ" alone would send you looking for the
            # wrong thing.
            why = f"same file count ({n}) but names or sizes changed"
        return {"ok": False, "reason": f"does not match manifest — {why}"}
    return {"ok": True, "reason": "file list and sizes match"}


def dataset_ref(name: str, verify: bool = True) -> str:
    """The string to put in a run's ``config["dataset"]``.

    Raises if the data on disk no longer matches its manifest. That is
    deliberate: training on silently-changed data produces a result nobody can
    explain later, and this is the one moment it is cheap to catch.
    """
    m = load_manifest(name)
    if verify:
        v = verify_dataset(name)
        if not v["ok"]:
            raise RuntimeError(
                f"dataset '{name}' does not match its manifest: {v['reason']}. "
                f"Re-run datasets/{name}/build.ipynb, or pass verify=False if "
                f"you know why and are recording it in the run note."
            )
    return f"{name}@{m['hash'][:6]}"


def list_datasets() -> list[dict[str, Any]]:
    """Every registered dataset, for the index."""
    if not DATASETS.exists():
        return []
    out = []
    for d in sorted(DATASETS.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        try:
            out.append(load_manifest(d.name))
        except (FileNotFoundError, json.JSONDecodeError):
            out.append({"name": d.name, "version": None, "unbuilt": True})
    return out

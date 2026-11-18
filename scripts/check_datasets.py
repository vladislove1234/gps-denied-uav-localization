#!/usr/bin/env python
"""Integrity checks for the dataset manifest machinery.

The narrow exception to "verify on real data": this code has no visible output
to judge. Hash stability, `.DS_Store` immunity and rename detection cannot be
eyeballed on a figure, so the check is a script — and it is kept as a file
rather than run once and thrown away.

    python scripts/check_datasets.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from uavloc.datasets import (  # noqa: E402
    DATASETS, data_root, dataset_ref, verify_dataset, write_manifest,
)

NAME = "_check"
results: list[tuple[bool, str]] = []


def check(ok: bool, label: str) -> None:
    results.append((ok, label))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")


def main() -> int:
    root = data_root(NAME, create=True)
    (root / "sub").mkdir(exist_ok=True)
    (root / "a.txt").write_text("sample 0\n")      # 9 bytes
    (root / "b.txt").write_text("sample 1\n")
    (root / "sub" / "c.txt").write_text("nested\n")

    print("dataset manifest checks")
    write_manifest(NAME, version="1", description="integrity check",
                   source="synthetic", config={"k": 1}, samples=2,
                   checks={"synthetic": True})
    ref = dataset_ref(NAME)
    check(ref.startswith(f"{NAME}@"), f"ref has name@hash form ({ref})")

    # macOS scatters .DS_Store through data trees; if it entered the hash the
    # dataset would appear to change every time Finder opened a folder.
    (root / ".DS_Store").write_bytes(b"junk")
    check(verify_dataset(NAME)["ok"] and dataset_ref(NAME) == ref,
          ".DS_Store does not perturb the hash")

    # Same byte length, different content: the stat-only check cannot see this
    # by construction. Documented limitation, asserted so it stays documented.
    (root / "a.txt").write_text("sampleZ0\n")
    check(verify_dataset(NAME)["ok"], "stat check misses same-size edit (known)")
    check(not verify_dataset(NAME, deep=True)["ok"], "deep check catches it")
    (root / "a.txt").write_text("sample 0\n")

    (root / "a.txt").rename(root / "renamed.txt")
    check(not verify_dataset(NAME)["ok"], "rename caught by stat check")
    (root / "renamed.txt").rename(root / "a.txt")

    (root / "extra.txt").write_text("new\n")
    check(not verify_dataset(NAME)["ok"], "added file caught by stat check")
    try:
        dataset_ref(NAME)
        check(False, "dataset_ref raises on mismatch")
    except RuntimeError:
        check(True, "dataset_ref raises on mismatch")
    check(dataset_ref(NAME, verify=False) == ref, "verify=False escape hatch")

    m2 = write_manifest(NAME, version="2", description="integrity check",
                        source="synthetic", config={"k": 2})
    check(f"{NAME}@{m2['hash'][:6]}" != ref, "rebuild moves the ref")

    empty = data_root(f"{NAME}-empty", create=True)
    try:
        write_manifest(f"{NAME}-empty", version="1", description="x",
                       source="x", config={})
        check(False, "empty directory refused")
    except ValueError:
        check(True, "empty directory refused")

    for d in (root, empty, DATASETS / NAME, DATASETS / f"{NAME}-empty"):
        shutil.rmtree(d, ignore_errors=True)

    failed = [lbl for ok, lbl in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    for lbl in failed:
        print(f"  FAILED: {lbl}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

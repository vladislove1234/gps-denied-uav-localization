"""Units of reusable work, and the handoff around them.

A build is one piece of equipment being made — a log reader, a dataset, a
matcher. The folder holds the *record and the evidence*, never the code itself:
reusable code lives in ``uavloc/`` where it belongs.

    builds/<slug>/notes.md      what it is, what I need from you, does it work
    builds/<slug>/verify.ipynb  runs it on real data, emits the figures
    builds/<slug>/figures/      what you actually look at

Verification here means *run it on real data and produce something a human can
judge*, not assert against synthetic input. For code that touches flight logs
that is the stronger check: a unit test would pass with a wrong GPS epoch
offset forever, while a plotted track landing in the wrong field is obvious in
one second.

``notes.md`` is the single source of truth. Everything below parses it rather
than keeping a parallel status file, so the two can never disagree.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from uavloc.track import REPO_ROOT

__all__ = ["BUILDS", "build_dir", "parse_notes", "list_builds", "open_blocks"]

BUILDS = REPO_ROOT / "builds"

# "**Status:** ready for your review · **Unblocks:** `002`, `datasets/x`"
_STATUS = re.compile(r"\*\*Status:\*\*\s*(?P<status>[^·\n]+?)\s*(?:·|$)", re.M)
_UNBLOCKS = re.compile(r"\*\*Unblocks:\*\*\s*(?P<unblocks>[^\n]+)", re.M)
# "- [x] A flight log → `data/flights/logs/x.bin`"
_NEED = re.compile(
    r"^-\s*\[(?P<done>[ xX])\]\s*(?P<what>.+?)"
    r"(?:\s*(?:→|->)\s*`(?P<path>[^`]+)`)?\s*$", re.M
)

STATUSES = ("planning", "blocked", "ready for your review", "accepted", "parked")


def build_dir(name: str, create: bool = False) -> Path:
    d = BUILDS / name
    if create:
        (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


def _section(text: str, heading: str) -> str:
    """The body of one '## heading' section, up to the next heading."""
    m = re.search(rf"^##\s*{re.escape(heading)}\s*$(.*?)(?=^##\s|\Z)",
                  text, re.M | re.S)
    return m.group(1) if m else ""


def parse_notes(name: str) -> dict[str, Any]:
    """Read a build's state out of its notes.md.

    Returns status, what it unblocks, and the inputs needed from the user —
    each with whether it is ticked off and whether its path exists on disk.
    """
    f = build_dir(name) / "notes.md"
    if not f.exists():
        return {"name": name, "status": "missing notes.md", "unblocks": [],
                "needs": [], "ok": False}

    text = f.read_text()
    sm = _STATUS.search(text)
    um = _UNBLOCKS.search(text)

    needs = []
    for m in _NEED.finditer(_section(text, "What I needed from you")):
        path = m.group("path")
        needs.append({
            "what": m.group("what").strip(),
            "done": m.group("done").lower() == "x",
            "path": path,
            # A path that exists but is still unticked is the interesting case:
            # the user delivered and nobody noticed.
            "exists": (REPO_ROOT / path).exists() if path else None,
        })

    return {
        "name": name,
        "status": (sm.group("status").strip() if sm else "unknown"),
        "unblocks": [u.strip(" `") for u in um.group("unblocks").split(",")]
                    if um else [],
        "needs": needs,
        "ok": True,
    }


def list_builds() -> list[dict[str, Any]]:
    if not BUILDS.exists():
        return []
    return [parse_notes(d.name) for d in sorted(BUILDS.iterdir())
            if d.is_dir() and not d.name.startswith("_")]


def open_blocks() -> dict[str, list[dict[str, Any]]]:
    """What the session start needs to know.

    ``arrived`` is the point of all this: an input that was waiting on the user
    and is now present on disk. That is a signal to start work, not to ask what
    to do next.
    """
    arrived, waiting = [], []
    for b in list_builds():
        for n in b["needs"]:
            if n["done"] or not n["path"]:
                continue
            row = {"build": b["name"], **n}
            (arrived if n["exists"] else waiting).append(row)
    return {"arrived": arrived, "waiting": waiting}


def _cli() -> None:
    blocks = open_blocks()
    for b in list_builds():
        print(f"{b['name']:<28} {b['status']}")
    if blocks["arrived"]:
        print("\nArrived — ready to start:")
        for r in blocks["arrived"]:
            print(f"  {r['build']}: {r['what']} -> {r['path']}")
    if blocks["waiting"]:
        print("\nStill waiting on you:")
        for r in blocks["waiting"]:
            print(f"  {r['build']}: {r['what']} -> {r['path']}")


if __name__ == "__main__":
    _cli()

"""Run reference adapters over locations.

    python -m tools.annotator.refs run --source all --locations all
    python -m tools.annotator.refs run --source osm ms-footprints --locations birky-49p9105-23p9192
    python -m tools.annotator.refs list

Resumable: a (source, location) that already has ``meta.json`` is skipped
unless ``--force``. A failure is logged and the batch goes on.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

from .. import config
from . import REGISTRY, Location, base


def run(sources: list[str], slugs: list[str], force: bool) -> int:
    failed = 0
    for src in sources:
        adapter = REGISTRY[src]()
        t_all = time.time()
        done = skipped = 0
        for i, slug in enumerate(slugs, 1):
            if not force and base.exists(src, slug):
                skipped += 1
                continue
            t0 = time.time()
            try:
                loc = Location(slug)
                meta = base.write(adapter, loc, adapter.run(loc), time.time() - t0)
                stats = " ".join(f"{k}:{v['features'] or v['px']}" for k, v in meta["stats"].items())
                print(f"[{src}] {i:>2}/{len(slugs)} {slug:32s} {time.time() - t0:6.1f}s  {stats}",
                      flush=True)
                done += 1
            except Exception as exc:          # noqa: BLE001 -- one bad location must not kill the batch
                failed += 1
                print(f"[{src}] {i:>2}/{len(slugs)} {slug:32s} FAILED {type(exc).__name__}: {exc}",
                      flush=True)
                traceback.print_exc(limit=3, file=sys.stdout)
        adapter.close()
        print(f"[{src}] done {done}, skipped {skipped}, {time.time() - t_all:.0f}s", flush=True)
    return failed


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tools.annotator.refs")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--source", nargs="+", default=["all"])
    r.add_argument("--locations", nargs="+", default=["all"])
    r.add_argument("--force", action="store_true")
    sub.add_parser("list")
    a = p.parse_args(argv)

    if a.cmd == "list":
        slugs = config.locations("sat")
        for src in REGISTRY:
            n = sum(base.exists(src, s) for s in slugs)
            print(f"{src:15s} {n}/{len(slugs)}  provides {', '.join(REGISTRY[src].provides)}")
        return

    sources = list(REGISTRY) if a.source == ["all"] else a.source
    unknown = [s for s in sources if s not in REGISTRY]
    if unknown:
        p.error(f"unknown source(s) {unknown}; known: {list(REGISTRY)}")
    # references are for map locations only; drone frames are drawn by hand
    slugs = config.locations("sat") if a.locations == ["all"] else a.locations
    sys.exit(1 if run(sources, slugs, a.force) else 0)


if __name__ == "__main__":
    main()

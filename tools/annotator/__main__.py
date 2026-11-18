"""Annotator entry point. Run from ``src/`` in the ``uavloc`` env.

    python -m tools.annotator serve [--port 8765]        # then open http://localhost:8765
    python -m tools.annotator export --new|--stale|--all
    python -m tools.annotator status                      # per-location progress
    python -m tools.annotator.refs run --source all --locations all
"""

from __future__ import annotations

import argparse
import json


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m tools.annotator")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    e = sub.add_parser("export")
    g = e.add_mutually_exclusive_group()
    g.add_argument("--new", dest="mode", action="store_const", const="new")
    g.add_argument("--stale", dest="mode", action="store_const", const="stale")
    g.add_argument("--all", dest="mode", action="store_const", const="all")
    sub.add_parser("status")
    a = p.parse_args(argv)

    if a.cmd == "serve":
        import uvicorn
        print(f"annotator on http://{a.host}:{a.port}")
        uvicorn.run("tools.annotator.server.app:app", host=a.host, port=a.port, log_level="warning")
    elif a.cmd == "export":
        from .server.export import export
        print(json.dumps(export(a.mode or "new"), indent=1))
    elif a.cmd == "status":
        from collections import Counter
        from .server.workspace import list_locations
        locs = list_locations()
        print(dict(Counter(l["status"] for l in locs)))
        for l in locs:
            if l["status"] != "todo":
                print(f"  {l['status']:12s} {l['slug']}")


if __name__ == "__main__":
    main()

"""Writes the map commands a real Birky plan produces, so the map page can be checked in a plain browser.

Run from `src/`: `PYTHONPATH=. python builds/route-planner-app/harness/make_payloads.py`
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.route_planner.config import AppConfig
from tools.route_planner.planning_job import PlannerEngine
from tools.route_planner.tile_sources import tile_source
from tools.route_planner.view_model import routes_payload
from uavloc.geo import LatLon
from uavloc.planning.config import MissionInputs
from uavloc.track import REPO_ROOT

OUT = Path(__file__).resolve().parent / "payloads.json"
START, END = LatLon(49.905, 23.885), LatLon(49.915, 23.925)
OVERLAY_OPACITY = {"classes": 0.55, "blind": 0.5}

config = AppConfig.defaults(REPO_ROOT)
outcome = PlannerEngine(config).plan(MissionInputs(START, END))
overlays = {"classes": outcome.class_overlay, "blind": outcome.blind_overlay}
OUT.write_text(json.dumps({
    "tiles": tile_source(config.tile_source_key).leaflet_options(),
    "start": START, "end": END,
    "routes": routes_payload(outcome.result.routes),
    "overlays": [{"name": name, "dataUrl": o.data_url, "bounds": o.leaflet_bounds,
                  "opacity": OVERLAY_OPACITY[name], "visible": True} for name, o in overlays.items()],
}))
print(OUT, OUT.stat().st_size)

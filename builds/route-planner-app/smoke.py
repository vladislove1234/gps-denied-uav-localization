"""Drives the real app on the Birky case and saves screenshots. Run from `src/`:

    ~/miniconda3/envs/uavloc/bin/python builds/route-planner-app/smoke.py [--sitl]

The window appears on screen: QtWebEngine does not render offscreen.
With --no-map the map page is not waited for (QtWebEngine cannot start its renderer inside the agent sandbox):
the Python side is driven for real, the map commands go nowhere, and only the side panel is screenshotted.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QMessageBox

from tools.route_planner.__main__ import create_application
from tools.route_planner.config import AppConfig
from tools.route_planner.main_window import MainWindow
from tools.route_planner.map_bridge import MARKER_END, MARKER_START
from uavloc.geo import LatLon
from uavloc.track import REPO_ROOT

FIGURES = Path(__file__).resolve().parent / "figures"
START = LatLon(49.905, 23.885)
END = LatLon(49.915, 23.925)
TILE_WAIT_S = 12.0
PLAN_TIMEOUT_S = 900.0
TASK_TIMEOUT_S = 120.0
POLL_MS = 100
BIRKY_ZOOM = 14
SITL_PORT_WAIT_S = 900.0   # other sessions run SITL on the same port
PANEL_ONLY = False   # set by --no-map


def pump(app, seconds: float) -> None:
    loop = QEventLoop()
    QTimer.singleShot(int(seconds * 1000), loop.quit)
    loop.exec()


def wait_until(app, condition, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise TimeoutError(what)
        pump(app, POLL_MS / 1000)


def shot(window: MainWindow, name: str) -> Path:
    path = FIGURES / name
    (window.panel if PANEL_ONLY else window).grab().save(str(path))
    print("saved", path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sitl", action="store_true")
    parser.add_argument("--no-map", action="store_true")
    args = parser.parse_args()
    global PANEL_ONLY
    PANEL_ONLY = args.no_map
    logging.basicConfig(level=logging.INFO)
    FIGURES.mkdir(parents=True, exist_ok=True)
    errors: list[tuple[str, str]] = []
    app = create_application(sys.argv[:1])
    out_dir = Path(tempfile.mkdtemp(prefix="route-planner-export-"))
    config = AppConfig.defaults(REPO_ROOT)
    window = MainWindow(config)
    window.show()
    window.show_error = lambda title, text: errors.append((title, text))
    QMessageBox.warning = staticmethod(lambda *a, **k: errors.append(("warning", str(a[-1]))))
    window.panel.export_dir.setText(str(out_dir))
    if args.no_map:
        window._on_map_ready()
    wait_until(app, lambda: window.map_is_ready, 30, "map ready")
    window.bridge.send("setView", {"lat": 49.91, "lon": 23.905, "zoom": BIRKY_ZOOM})
    pump(app, TILE_WAIT_S)
    shot(window, "01-empty-map.png")

    window._on_map_clicked(START.lat, START.lon)
    window._on_map_clicked(END.lat, END.lon)
    assert (window.start, window.end) == (START, END), (window.start, window.end)
    pump(app, 2)
    shot(window, "02-points.png")

    window.panel.plan_button.click()
    wait_until(app, lambda: window.outcome is not None or errors, PLAN_TIMEOUT_S, "planning")
    assert not errors, errors
    pump(app, TILE_WAIT_S)
    shot(window, "03-routes.png")
    print("rows", window.panel.table.rowCount(), "elevation error", window.outcome.elevation_error)

    window.panel.layer_boxes["classes"].setChecked(True)
    pump(app, 3)
    shot(window, "04-class-map.png")
    window.panel.layer_boxes["classes"].setChecked(False)
    window.panel.layer_boxes["blind"].setChecked(True)
    pump(app, 3)
    shot(window, "05-blind-map.png")
    window.panel.layer_boxes["blind"].setChecked(False)

    last = window.panel.table.rowCount() - 1
    window.panel.table.selectRow(last)
    pump(app, 1)
    assert window.selected_route == last
    shot(window, f"06-route-{last}-selected.png")

    window.panel.export_button.click()
    wait_until(app, lambda: window.last_exported is not None or errors, TASK_TIMEOUT_S, "export")
    assert not errors, errors
    exported = window.last_exported
    print(json.dumps([str(p) for p in exported.paths()], indent=1))
    assert all(p.exists() for p in exported.paths())
    shot(window, "07-exported.png")

    if args.sitl:
        run_sitl_upload(app, window, errors)
    window.close()


def run_sitl_upload(app, window: MainWindow, errors: list) -> None:
    from uavloc.ardupilot.sitl import SitlConfig, default_ardupilot_dir, start_sitl
    from uavloc.ardupilot.sitl import port_is_open
    wait_until(app, lambda: not port_is_open(), SITL_PORT_WAIT_S, "SITL port busy: another SITL is running")
    sitl = SitlConfig(default_ardupilot_dir(), START, 200.0, 0.0, 1.0)
    with start_sitl(sitl) as url:
        window.panel.connection.setText(url)
        window.panel.upload_button.click()
        wait_until(app, lambda: window.last_upload is not None or errors, TASK_TIMEOUT_S, "upload")
        assert not errors, errors
        print(window.last_upload.summary())
        shot(window, "08-uploaded.png")


if __name__ == "__main__":
    main()

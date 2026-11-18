"""The main window: Leaflet map on the left, the control panel on the right. Wiring only; logic is in plain modules."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from PySide6.QtCore import QFile, QIODevice, Qt, QTimer, QUrl
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineScript, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QMainWindow, QMessageBox, QSplitter

from tools.route_planner.config import AppConfig
from tools.route_planner.export import ExportedFiles, ExportRequest, UploadReport, export_route, \
    upload_route, write_vehicle_params
from tools.route_planner.map_bridge import MARKER_END, MARKER_START, MapBridge
from tools.route_planner.overlays import MapOverlay
from tools.route_planner.page import inline_page
from tools.route_planner.panel import LAYER_BLIND, LAYER_CLASSES, LAYER_SATELLITE, ControlPanel
from tools.route_planner.planning_job import PlanOutcome, PlannerEngine
from tools.route_planner.tile_sources import TILE_SOURCES, MapTileSource, tile_source
from tools.route_planner.view_model import route_rows, routes_payload
from tools.route_planner.workers import TaskThread
from uavloc.geo import LatLon

LOGGER = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent / "web"
WINDOW_TITLE = "UAV route planner"
WINDOW_SIZE_PX = (1500, 900)
PANEL_WIDTH_PX = 520
OVERLAY_OPACITY = {LAYER_CLASSES: 0.55, LAYER_BLIND: 0.5}
PAGE_PROBE_DELAY_MS = 8000
PAGE_BASE_URL = "https://route-planner.local/"   # an http(s) origin, so remote tiles load like on any web page
QWEBCHANNEL_JS = ":/qtwebchannel/qwebchannel.js"
PAGE_PROBE_JS = """JSON.stringify({leaflet: typeof L, webchannel: typeof QWebChannel,
  transport: typeof qt !== "undefined" && !!qt.webChannelTransport,
  bridge: typeof bridge !== "undefined" && bridge !== null, map: typeof map !== "undefined" && map !== null,
  scripts: Array.from(document.scripts).map((s) => s.src.split("/").pop() || "inline"),
  size: [document.body.clientWidth, document.body.clientHeight],
  tiles: document.querySelectorAll(".leaflet-tile").length,
  tiles_loaded: document.querySelectorAll(".leaflet-tile-loaded").length})"""


def qwebchannel_script() -> QWebEngineScript:
    """Qt's qwebchannel.js, injected before the page's own scripts.

    Loading it with <script src="qrc:..."> from a file:// page failed silently in the packaged app (blank map).
    """
    source = QFile(QWEBCHANNEL_JS)
    if not source.open(QIODevice.ReadOnly):
        raise RuntimeError(f"cannot read {QWEBCHANNEL_JS} from the Qt resources")
    script = QWebEngineScript()
    script.setName("qwebchannel")
    script.setSourceCode(bytes(source.readAll()).decode("utf-8"))
    script.setInjectionPoint(QWebEngineScript.DocumentCreation)
    script.setWorldId(QWebEngineScript.MainWorld)
    return script


class MapPage(QWebEnginePage):
    """Sends the page's console output to the Python log, where a JavaScript error would otherwise vanish."""

    def javaScriptConsoleMessage(self, level, message, line_number, source_id) -> None:
        LOGGER.log(logging.WARNING if level != QWebEnginePage.InfoMessageLevel else logging.INFO,
                   "map js %s:%d %s", source_id, line_number, message)


class MainWindow(QMainWindow):
    def __init__(self, app_config: AppConfig, engine: PlannerEngine | None = None) -> None:
        super().__init__()
        self.app_config = app_config
        self.engine = engine or PlannerEngine(app_config)
        self.start: LatLon | None = None
        self.end: LatLon | None = None
        self.outcome: PlanOutcome | None = None
        self.selected_route: int | None = None
        self.map_is_ready = False
        self.tiles_failed = 0
        self.last_exported: ExportedFiles | None = None
        self.last_upload: UploadReport | None = None
        self._task: TaskThread | None = None
        self.show_error: Callable[[str, str], None] = self._message_box_error

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(*WINDOW_SIZE_PX)
        self.bridge = MapBridge()
        self.view = self._build_map_view()
        self.panel = ControlPanel(app_config.output_dir, app_config.connection,
                                  {key: option.label for key, option in TILE_SOURCES.items()},
                                  app_config.tile_source_key)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.view)
        splitter.addWidget(self.panel)
        splitter.setSizes([WINDOW_SIZE_PX[0] - PANEL_WIDTH_PX, PANEL_WIDTH_PX])
        splitter.setStretchFactor(0, 1)
        self.setCentralWidget(splitter)
        self._connect_signals()

    def _build_map_view(self) -> QWebEngineView:
        view = QWebEngineView()
        view.setPage(MapPage(view))
        # the page is a local file that loads remote tiles
        view.settings().setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)
        view.page().scripts().insert(qwebchannel_script())
        channel = QWebChannel(view.page())
        channel.registerObject("bridge", self.bridge)
        view.page().setWebChannel(channel)
        view.loadFinished.connect(self._on_page_loaded)
        view.page().renderProcessTerminated.connect(
            lambda status, code: LOGGER.error("map renderer terminated: %s (exit %s)", status, code))
        view.setHtml(inline_page(WEB_DIR), QUrl(PAGE_BASE_URL))
        return view

    def _on_page_loaded(self, ok: bool) -> None:
        LOGGER.info("map page loaded ok=%s from %s", ok, WEB_DIR)
        # a blank map is otherwise silent: record what the page actually has once it should be up
        QTimer.singleShot(PAGE_PROBE_DELAY_MS, lambda: self.view.page().runJavaScript(
            PAGE_PROBE_JS, 0, lambda state: LOGGER.info("map page state %s", state)))

    def _connect_signals(self) -> None:
        self.bridge.ready.connect(self._on_map_ready)
        self.bridge.map_clicked.connect(self._on_map_clicked)
        self.bridge.marker_moved.connect(self._on_marker_moved)
        self.bridge.route_clicked.connect(self.select_route)
        self.bridge.tiles_settled.connect(self._on_tiles_settled)
        self.panel.plan_requested.connect(self.plan_routes)
        self.panel.reset_requested.connect(self.reset_points)
        self.panel.layer_toggled.connect(self._on_layer_toggled)
        self.panel.tile_source_selected.connect(self.switch_tile_source)
        self.panel.route_selected.connect(self.select_route)
        self.panel.export_requested.connect(self.export_selected)
        self.panel.upload_requested.connect(self.upload_selected)
        self.panel.params_requested.connect(self.write_parameters)

    def _message_box_error(self, title: str, text: str) -> None:
        QMessageBox.critical(self, title, text)

    # map

    def _on_map_ready(self) -> None:
        LOGGER.info("map page connected over QWebChannel")
        self.map_is_ready = True
        self.switch_tile_source(self.app_config.tile_source_key)
        self.bridge.send("setView", {"lat": self.app_config.view_lat, "lon": self.app_config.view_lon,
                                     "zoom": self.app_config.view_zoom})

    def switch_tile_source(self, key: str) -> None:
        """Building a source may call Google's API, so it runs off the UI thread."""
        self.panel.set_tile_source(self.app_config.tile_source_key)   # the picker shows the active source until then
        self._run_task(lambda progress: tile_source(key), lambda source: self._use_tile_source(key, source),
                       "Switch imagery")

    def _use_tile_source(self, key: str, source: MapTileSource) -> None:
        if key != self.app_config.tile_source_key:
            self._clear_results()   # the class map, blind map and exported mosaic all come from the imagery
            self.app_config = replace(self.app_config, tile_source_key=key)
            self.engine.app_config = self.app_config
        self.panel.set_tile_source(key)
        self.panel.append_log(f"imagery: {source.label}")
        self.bridge.send("setTileSource", source.leaflet_options())

    def _on_tiles_settled(self, failed: int) -> None:
        self.tiles_failed = failed
        if failed:
            self.panel.append_log(f"{failed} map tiles failed to load (offline?)")

    def _on_map_clicked(self, lat: float, lon: float) -> None:
        if self.start is None:
            self.set_point(MARKER_START, LatLon(lat, lon))
        elif self.end is None:
            self.set_point(MARKER_END, LatLon(lat, lon))

    def _on_marker_moved(self, kind: str, lat: float, lon: float) -> None:
        self._store_point(kind, LatLon(lat, lon))
        self._clear_results()

    def _store_point(self, kind: str, point: LatLon | None) -> None:
        if kind == MARKER_START:
            self.start = point
        else:
            self.end = point
        self.panel.set_points(self.start, self.end)

    def set_point(self, kind: str, point: LatLon) -> None:
        self._store_point(kind, point)
        self.bridge.send("setPoint", {"kind": kind, "lat": point.lat, "lon": point.lon})
        self._clear_results()

    def reset_points(self) -> None:
        self.start = self.end = None
        self.panel.set_points(None, None)
        self.bridge.send("clearPoints", {})
        self._clear_results()

    def _on_layer_toggled(self, layer: str, visible: bool) -> None:
        if layer == LAYER_SATELLITE:
            self.bridge.send("setTileSourceVisible", {"visible": visible})
        elif self.outcome is not None:
            self.bridge.send("setOverlayVisible", {"name": layer, "visible": visible})

    def _clear_results(self) -> None:
        if self.outcome is None:
            return
        self.outcome = None
        self.selected_route = None
        self.panel.clear_rows()
        self.bridge.send("clearRoutes", {})
        self.bridge.send("clearOverlays", {})

    # background tasks

    def _run_task(self, function: Callable[[Callable[[str, int, int], None]], Any], on_success: Callable[[Any], None],
                  title: str) -> None:
        if self._task is not None:
            return
        task = TaskThread(function)
        task.progress.connect(self.panel.set_progress)
        task.succeeded.connect(lambda result: self._finish_task(on_success, result))
        task.failed.connect(lambda message: self._fail_task(title, message))
        self._task = task
        self.panel.set_busy(True)
        task.start()

    def _release_task(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.wait()
        self.panel.set_busy(False)

    def _finish_task(self, on_success: Callable[[Any], None], result: Any) -> None:
        self._release_task()
        on_success(result)

    def _fail_task(self, title: str, message: str) -> None:
        self._release_task()
        self.panel.set_stage("failed")
        self.panel.append_log(f"{title} failed: {message}")
        self.show_error(title, message)

    # planning

    def plan_routes(self) -> None:
        if self.start is None or self.end is None:
            self.show_error("Plan routes", "Set the start and the end point on the map first.")
            return
        inputs = self.panel.mission_inputs(self.start, self.end)
        self._clear_results()
        self.panel.append_log("planning...")
        self._run_task(lambda progress: self.engine.plan(inputs, progress), self._show_outcome, "Planning")

    def _show_outcome(self, outcome: PlanOutcome) -> None:
        self.outcome = outcome
        routes = outcome.result.routes
        self.panel.set_stage(f"done: {len(routes)} routes")
        self.panel.set_rows(route_rows(routes, outcome.elevation))
        self._send_overlay(LAYER_CLASSES, outcome.class_overlay)
        self._send_overlay(LAYER_BLIND, outcome.blind_overlay)
        self.bridge.send("showRoutes", routes_payload(routes))
        self.panel.append_log(f"{len(routes)} routes; blind limit {outcome.result.blind_limit_m:.0f} m")
        if outcome.elevation_error:
            self.panel.append_log(f"elevation check failed: {outcome.elevation_error}")
        self.select_route(0)

    def _send_overlay(self, name: str, overlay: MapOverlay) -> None:
        self.bridge.send("setOverlay", {"name": name, "dataUrl": overlay.data_url, "bounds": overlay.leaflet_bounds,
                                        "opacity": OVERLAY_OPACITY[name],
                                        "visible": self.panel.layer_boxes[name].isChecked()})

    def select_route(self, index: int) -> None:
        if self.outcome is None or not 0 <= index < len(self.outcome.result.routes):
            return
        self.selected_route = index
        self.panel.select_row(index)
        self.bridge.send("selectRoute", {"index": index})

    # export and drone

    def export_selected(self, out_dir: str) -> None:
        if self.outcome is None or self.selected_route is None:
            return
        outcome, route = self.outcome, self.outcome.result.routes[self.selected_route]
        request = ExportRequest(outcome.result, route, outcome.inputs, outcome.config,
                                self.app_config.tile_source_key, self.app_config.tile_cache,
                                self.app_config.drone_onnx)
        self.panel.set_stage("exporting")
        self._run_task(lambda progress: export_route(Path(out_dir), request), self._show_export, "Export")

    def _show_export(self, files: ExportedFiles) -> None:
        self.last_exported = files
        self.panel.set_stage("exported")
        self.panel.append_log("exported:\n" + "\n".join(f"  {path}" for path in files.paths()))

    def upload_selected(self, connection: str) -> None:
        if self.outcome is None or self.selected_route is None:
            return
        route, inputs = self.outcome.result.routes[self.selected_route], self.outcome.inputs
        self.panel.set_stage("uploading")
        self._run_task(lambda progress: upload_route(connection, route, inputs), self._show_upload, "Upload to drone")

    def _show_upload(self, report: UploadReport) -> None:
        self.last_upload = report
        self.panel.set_stage("uploaded")
        self.panel.append_log(report.summary())
        if report.home_far_from_start:
            QMessageBox.warning(self, "Vehicle home", report.summary())

    def write_parameters(self, connection: str) -> None:
        self.panel.set_stage("writing parameters")
        self._run_task(lambda progress: write_vehicle_params(connection),
                       lambda count: (self.panel.set_stage("parameters written"),
                                      self.panel.append_log(f"{count} parameters written")), "Write parameters")

"""The object the map page talks to over QWebChannel: JS calls its slots, Python sends it commands."""

from __future__ import annotations

import json

from PySide6.QtCore import QObject, Signal, Slot

MARKER_START = "start"
MARKER_END = "end"


class MapBridge(QObject):
    command = Signal(str, str)               # (map command name, JSON payload), handled in web/app.js
    ready = Signal()
    map_clicked = Signal(float, float)       # lat, lon
    marker_moved = Signal(str, float, float)  # kind, lat, lon
    route_clicked = Signal(int)
    tiles_settled = Signal(int)              # number of tiles that failed to load

    def send(self, name: str, payload: dict) -> None:
        self.command.emit(name, json.dumps(payload))

    @Slot()
    def mapReady(self) -> None:
        self.ready.emit()

    @Slot(float, float)
    def mapClicked(self, lat: float, lon: float) -> None:
        self.map_clicked.emit(lat, lon)

    @Slot(str, float, float)
    def markerMoved(self, kind: str, lat: float, lon: float) -> None:
        self.marker_moved.emit(kind, lat, lon)

    @Slot(int)
    def routeSelected(self, index: int) -> None:
        self.route_clicked.emit(index)

    @Slot(int)
    def tilesSettled(self, failed: int) -> None:
        self.tiles_settled.emit(failed)

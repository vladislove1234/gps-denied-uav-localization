"""The right-hand side panel: native Qt widgets only, all behaviour is signals out and setters in."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QProgressBar, QPlainTextEdit,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from tools.route_planner.view_model import TABLE_HEADERS, RouteRow
from uavloc.geo import LatLon
from uavloc.planning.config import (DEFAULT_ALTITUDE_M, DEFAULT_MAX_FLIGHT_TIME_S, DEFAULT_SPEED_MPS,
                                    MissionInputs)

SECONDS_PER_MINUTE = 60.0
MAX_ALTITUDE_M = 500.0
MIN_ALTITUDE_M = 5.0
MAX_SPEED_MPS = 30.0
MIN_SPEED_MPS = 0.5
MAX_FLIGHT_MIN = 240.0
MAX_HOVER_S = 3600.0
POINT_DECIMALS = 5
LOG_MAX_BLOCKS = 500
TABLE_MIN_HEIGHT_PX = 130

LAYER_SATELLITE, LAYER_CLASSES, LAYER_BLIND = "satellite", "classes", "blind"
LAYER_LABELS = {LAYER_SATELLITE: "Satellite", LAYER_CLASSES: "Class map", LAYER_BLIND: "Localizability (blind = red)"}
LAYER_DEFAULTS = {LAYER_SATELLITE: True, LAYER_CLASSES: False, LAYER_BLIND: False}


def spin_box(value: float, low: float, high: float, step: float, suffix: str) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setSingleStep(step)
    box.setSuffix(suffix)
    box.setValue(value)
    return box


def format_point(point: LatLon | None) -> str:
    return "click the map" if point is None else f"{point.lat:.{POINT_DECIMALS}f}, {point.lon:.{POINT_DECIMALS}f}"


class ControlPanel(QWidget):
    plan_requested = Signal()
    reset_requested = Signal()
    layer_toggled = Signal(str, bool)
    tile_source_selected = Signal(str)
    route_selected = Signal(int)
    export_requested = Signal(str)
    upload_requested = Signal(str)
    params_requested = Signal(str)

    def __init__(self, output_dir: Path, connection: str, tile_sources: dict[str, str], tile_source_key: str) -> None:
        """`tile_sources` maps each imagery key to the label the picker shows."""
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(self._points_group())
        layout.addWidget(self._mission_group())
        layout.addWidget(self._plan_group())
        layout.addWidget(self._layers_group(tile_sources, tile_source_key))
        layout.addWidget(self._routes_group(), stretch=1)
        layout.addWidget(self._export_group(output_dir))
        layout.addWidget(self._drone_group(connection))
        layout.addWidget(self._log_view(), stretch=1)
        self._row_details: list[str] = []
        self._results_available = False
        self._buttons = [self.plan_button, self.reset_button, self.export_button, self.upload_button,
                         self.params_button, self.tile_source_box]
        self.set_results_available(False)

    def _points_group(self) -> QGroupBox:
        group, form = QGroupBox("Points"), QFormLayout()
        self.start_label, self.end_label = QLabel(format_point(None)), QLabel(format_point(None))
        self.reset_button = QPushButton("Reset points")
        self.reset_button.clicked.connect(self.reset_requested)
        form.addRow("Start (S)", self.start_label)
        form.addRow("End (E)", self.end_label)
        form.addRow(self.reset_button)
        group.setLayout(form)
        return group

    def _mission_group(self) -> QGroupBox:
        group, form = QGroupBox("Mission"), QFormLayout()
        self.altitude = spin_box(DEFAULT_ALTITUDE_M, MIN_ALTITUDE_M, MAX_ALTITUDE_M, 5, " m")
        self.speed = spin_box(DEFAULT_SPEED_MPS, MIN_SPEED_MPS, MAX_SPEED_MPS, 0.5, " m/s")
        self.flight_time = spin_box(DEFAULT_MAX_FLIGHT_TIME_S / SECONDS_PER_MINUTE, 1, MAX_FLIGHT_MIN, 1, " min")
        self.hover = spin_box(0, 0, MAX_HOVER_S, 5, " s")
        form.addRow("Altitude (above takeoff)", self.altitude)
        form.addRow("Speed", self.speed)
        form.addRow("Max flight time", self.flight_time)
        form.addRow("Hover at end point", self.hover)
        group.setLayout(form)
        return group

    def _plan_group(self) -> QGroupBox:
        group, column = QGroupBox("Planning"), QVBoxLayout()
        self.plan_button = QPushButton("Plan routes")
        self.plan_button.clicked.connect(self.plan_requested)
        self.stage_label = QLabel("idle")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        for widget in (self.plan_button, self.stage_label, self.progress_bar):
            column.addWidget(widget)
        group.setLayout(column)
        return group

    def _layers_group(self, tile_sources: dict[str, str], tile_source_key: str) -> QGroupBox:
        group, column, row = QGroupBox("Layers"), QVBoxLayout(), QHBoxLayout()
        self.tile_source_box = QComboBox()
        for key, label in tile_sources.items():
            self.tile_source_box.addItem(label, key)
        self.set_tile_source(tile_source_key)
        self.tile_source_box.activated.connect(
            lambda index: self.tile_source_selected.emit(self.tile_source_box.itemData(index)))
        imagery = QFormLayout()
        imagery.addRow("Imagery", self.tile_source_box)
        column.addLayout(imagery)
        self.layer_boxes: dict[str, QCheckBox] = {}
        for name, label in LAYER_LABELS.items():
            box = QCheckBox(label)
            box.setChecked(LAYER_DEFAULTS[name])
            box.toggled.connect(lambda checked, layer=name: self.layer_toggled.emit(layer, checked))
            self.layer_boxes[name] = box
            row.addWidget(box)
        column.addLayout(row)
        group.setLayout(column)
        return group

    def _routes_group(self) -> QGroupBox:
        group, column = QGroupBox("Routes"), QVBoxLayout()
        self.table = QTableWidget(0, len(TABLE_HEADERS))
        self.table.setHorizontalHeaderLabels(list(TABLE_HEADERS))
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMinimumHeight(TABLE_MIN_HEIGHT_PX)
        self.table.itemSelectionChanged.connect(self._emit_selected_route)
        self.details = QLabel("")
        self.details.setWordWrap(True)
        column.addWidget(self.table)
        column.addWidget(self.details)
        group.setLayout(column)
        return group

    def _export_group(self, output_dir: Path) -> QGroupBox:
        group, row = QGroupBox("Export selected route"), QHBoxLayout()
        self.export_dir = QLineEdit(str(output_dir))
        browse = QPushButton("...")
        browse.clicked.connect(self._browse_export_dir)
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(lambda: self.export_requested.emit(self.export_dir.text()))
        for widget in (self.export_dir, browse, self.export_button):
            row.addWidget(widget)
        group.setLayout(row)
        return group

    def _drone_group(self, connection: str) -> QGroupBox:
        group, column = QGroupBox("Drone"), QVBoxLayout()
        self.connection = QLineEdit(connection)
        self.connection.setToolTip("SITL: tcp:127.0.0.1:5760 · flight controller: /dev/tty.usbmodem...,115200")
        buttons = QHBoxLayout()
        self.upload_button = QPushButton("Upload mission")
        self.upload_button.clicked.connect(lambda: self.upload_requested.emit(self.connection.text()))
        self.params_button = QPushButton("Write parameters")
        self.params_button.clicked.connect(lambda: self.params_requested.emit(self.connection.text()))
        buttons.addWidget(self.upload_button)
        buttons.addWidget(self.params_button)
        column.addWidget(self.connection)
        column.addLayout(buttons)
        group.setLayout(column)
        return group

    def _log_view(self) -> QPlainTextEdit:
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(LOG_MAX_BLOCKS)
        return self.log

    def _browse_export_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Export folder", self.export_dir.text())
        if chosen:
            self.export_dir.setText(chosen)

    def _emit_selected_route(self) -> None:
        index = self.selected_index()
        if index is not None:
            self.details.setText(self._row_details[index])
            self.route_selected.emit(index)

    def selected_index(self) -> int | None:
        rows = self.table.selectionModel().selectedRows()
        return rows[0].row() if rows else None

    def mission_inputs(self, start: LatLon, end: LatLon) -> MissionInputs:
        return MissionInputs(start, end, self.altitude.value(), self.speed.value(),
                             self.flight_time.value() * SECONDS_PER_MINUTE, self.hover.value())

    def set_points(self, start: LatLon | None, end: LatLon | None) -> None:
        self.start_label.setText(format_point(start))
        self.end_label.setText(format_point(end))

    def set_rows(self, rows: list[RouteRow]) -> None:
        self._row_details = [row.details() for row in rows]
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, text in enumerate(row.cells()):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignVCenter | (Qt.AlignLeft if c == len(TABLE_HEADERS) - 1
                                                         else Qt.AlignRight))
                if c == 0:
                    item.setBackground(QColor(row.colour))
                elif row.flags and c == len(TABLE_HEADERS) - 1:
                    item.setForeground(QColor("#c0392b"))
                self.table.setItem(r, c, item)
            if row.elevation_warnings:
                self.table.item(r, 1).setToolTip(row.details())
        self.table.blockSignals(False)
        self.set_results_available(bool(rows))

    def clear_rows(self) -> None:
        self._row_details = []
        self.table.setRowCount(0)
        self.details.setText("")
        self.set_results_available(False)

    def select_row(self, index: int) -> None:
        """Select without echoing `route_selected`, for selections that came from the map."""
        self.table.blockSignals(True)
        self.table.selectRow(index)
        self.table.blockSignals(False)
        self.details.setText(self._row_details[index])

    def set_results_available(self, available: bool) -> None:
        self._results_available = available
        for button in (self.export_button, self.upload_button):
            button.setEnabled(available)

    def set_tile_source(self, key: str) -> None:
        """Shows `key` as chosen without emitting `tile_source_selected`; puts the picker back after a failed switch."""
        self.tile_source_box.setCurrentIndex(self.tile_source_box.findData(key))

    def set_busy(self, busy: bool) -> None:
        for button in self._buttons:
            button.setEnabled(not busy)
        if not busy:
            self.set_results_available(self._results_available)

    def set_progress(self, stage: str, done: int, total: int) -> None:
        self.stage_label.setText(f"{stage}: {done}/{total}")
        self.progress_bar.setRange(0, max(total, 1))
        self.progress_bar.setValue(done)

    def set_stage(self, text: str) -> None:
        self.stage_label.setText(text)

    def append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

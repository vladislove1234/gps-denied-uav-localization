"""Video of a simulated flight: the drone's view, with panels (map with tracks, segmentation, heatmap) in corners.

Every visual choice is a `VideoConfig` field. A new panel is a function `(FrameContext, (w, h)) -> (h, w, 3) RGB`
passed in `extra_panels`, then named in `main_panel` or `corner_panels` like the built-in ones.
Preview one frame with `FlightVideo.frame(step)` before writing the whole file with `save`.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import cv2
import numpy as np
from PIL import Image

from uavloc.data.satellite import GeoImage
from uavloc.features.segmentation import BACKGROUND
from uavloc.geo import LatLon, east_north_m_array, rotate_to_north_up
from uavloc.plotting.correlation import SEG_PALETTE
from uavloc.plotting.flight import TRACK_COLUMNS, TRACK_LABELS, TRACK_RGB
from uavloc.simulation.camera import DroneCamera
from uavloc.simulation.correctors import Fix
from uavloc.simulation.flight import FlightResult
from uavloc.simulation.route import Pose

CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")
MAP_MODES = ("follow", "route")
SEGMENTATION_SOURCES = ("last-fix", "every-frame")
BACKGROUND_RGB = (30, 30, 30)
TEXT_RGB = (255, 255, 255)
BORDER_RGB = (255, 255, 255)
REJECT_RGB = (255, 255, 255)
LINE_PX_PER_FONT_SCALE = 22
TEXT_PAD_PX = 4

Segmenter = Callable[[Image.Image, float, float], np.ndarray]   # (view, width_m, height_m) -> (H, W) class ids


@dataclass(frozen=True)
class FrameContext:
    step: int
    pose: Pose
    view: np.ndarray                 # (H, W, 3) the drone's view at the true pose
    class_map: np.ndarray | None     # (H, W) class ids of the view, from the last fix or this frame
    fix: Fix | None                  # latest fix at or before this step
    fix_step: int | None
    result: FlightResult


PanelFn = Callable[[FrameContext, tuple[int, int]], np.ndarray]


@dataclass(frozen=True)
class VideoConfig:
    fps: float = 10.0
    frame_size_px: tuple[int, int] = (1280, 720)                     # width, height
    main_panel: str = "drone"
    corner_panels: Mapping[str, str] = field(default_factory=lambda: {"top-right": "segmentation",
                                                                      "bottom-right": "map"})
    panel_width_share: float = 0.3                                    # corner panel width / frame width
    panel_margin_px: int = 12
    panel_border_px: int = 2
    panel_titles: bool = True
    map_mode: str = "follow"                                          # "follow": centred on the drone; "route": whole flight
    map_radius_m: float = 120.0                                       # follow mode: half the map panel's width
    map_route_margin_m: float = 40.0
    tracks: tuple[str, ...] = ("truth", "ins", "estimate")
    trail_s: float | None = None                                      # None: the whole track so far
    track_width_px: int = 2
    track_rgb: Mapping[str, tuple[int, int, int]] = field(default_factory=lambda: dict(TRACK_RGB))
    show_footprint: bool = True
    show_fixes: bool = True
    show_legend: bool = True
    show_hud: bool = True
    segmentation_source: str = "last-fix"                             # "every-frame" needs a segmenter
    class_rgb: np.ndarray = field(default_factory=lambda: SEG_PALETTE.copy())   # (C, 3) uint8, CLASSES order
    overlay_alpha: float = 0.0                                        # segmentation blended over the drone view
    step_stride: int = 1                                              # render every n-th pose
    codec: str = "avc1"                                               # H.264; "mp4v" if avc1 is unavailable
    font_scale: float = 0.5


class FlightVideo:
    def __init__(self, result: FlightResult, camera: DroneCamera, map_raster: GeoImage, config: VideoConfig,
                 segmenter: Segmenter | None = None, extra_panels: Mapping[str, PanelFn] | None = None) -> None:
        self.result, self.camera, self.map_raster, self.config = result, camera, map_raster, config
        self.segmenter = segmenter
        self.panels: dict[str, PanelFn] = {"drone": self.drone_panel, "map": self.map_panel,
                                           "segmentation": self.segmentation_panel, "heatmap": self.heatmap_panel,
                                           **(extra_panels or {})}
        self._validate()
        self._route_view: tuple[LatLon, float, np.ndarray] | None = None

    def _validate(self) -> None:
        cfg = self.config
        unknown = {cfg.main_panel, *cfg.corner_panels.values()} - set(self.panels)
        if unknown:
            raise ValueError(f"unknown panels {sorted(unknown)}; available: {sorted(self.panels)}")
        if set(cfg.corner_panels) - set(CORNERS):
            raise ValueError(f"corners must be among {CORNERS}")
        if cfg.map_mode not in MAP_MODES or cfg.segmentation_source not in SEGMENTATION_SOURCES:
            raise ValueError(f"map_mode in {MAP_MODES}, segmentation_source in {SEGMENTATION_SOURCES}")
        if cfg.segmentation_source == "every-frame" and self.segmenter is None:
            raise ValueError("segmentation_source='every-frame' needs a segmenter")

    # ------------------------------------------------------------------ frames

    def context(self, step: int) -> FrameContext:
        pose = self.result.poses[step]
        view = np.asarray(self.camera.view(pose).convert("RGB"))
        steps = self.result.fix_steps
        at = bisect.bisect_right(steps, step) - 1
        fix_step = steps[at] if at >= 0 else None
        fix = self.result.fixes[fix_step] if fix_step is not None else None
        if self.config.segmentation_source == "every-frame":
            class_map = self.segmenter(Image.fromarray(view), *self.camera.footprint_m(pose.altitude_m))
        else:
            class_map = fix.class_map if fix is not None else None
        return FrameContext(step, pose, view, class_map, fix, fix_step, self.result)

    def frame(self, step: int) -> np.ndarray:
        """One composed video frame, `(H, W, 3)` RGB."""
        cfg = self.config
        ctx = self.context(step)
        width, height = cfg.frame_size_px
        canvas = np.ascontiguousarray(self.panels[cfg.main_panel](ctx, (width, height)))
        for corner, name in cfg.corner_panels.items():
            panel_w = round(cfg.panel_width_share * width)
            panel_h = round(panel_w / self._panel_aspect(name, ctx))
            x, y = corner_origin(corner, (panel_w, panel_h), (width, height), cfg.panel_margin_px)
            panel = self.panels[name](ctx, (panel_w, panel_h))
            if cfg.panel_titles:
                draw_text(panel, [name], (TEXT_PAD_PX, TEXT_PAD_PX), cfg.font_scale)
            canvas[y:y + panel_h, x:x + panel_w] = panel
            cv2.rectangle(canvas, (x, y), (x + panel_w - 1, y + panel_h - 1), BORDER_RGB, cfg.panel_border_px)
        if cfg.show_hud:
            self._hud(canvas, ctx)
        return canvas

    def save(self, path: Path) -> Path:
        """Write every `step_stride`-th pose as one frame."""
        cfg = self.config
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*cfg.codec), cfg.fps, cfg.frame_size_px)
        if not writer.isOpened():
            raise RuntimeError(f"OpenCV cannot write codec {cfg.codec!r} to {path}; try codec='mp4v'")
        try:
            for step in range(0, len(self.result.poses), cfg.step_stride):
                writer.write(cv2.cvtColor(self.frame(step), cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        return path

    def _panel_aspect(self, name: str, ctx: FrameContext) -> float:
        return 1.0 if name == "map" else ctx.view.shape[1] / ctx.view.shape[0]

    def _hud(self, canvas: np.ndarray, ctx: FrameContext) -> None:
        row = self.result.track.iloc[ctx.step]
        lines = [f"t {row['t_s']:.1f} s   alt {ctx.pose.altitude_m:.0f} m   yaw {ctx.pose.yaw_deg:.0f} deg",
                 f"error  INS only {row['ins_error_m']:.1f} m   corrected {row['est_error_m']:.1f} m",
                 f"last fix: {ctx.fix.decision} at t {self.result.track['t_s'].iloc[ctx.fix_step]:.1f} s"
                 if ctx.fix else "last fix: none yet"]
        height = canvas.shape[0]
        line_px = round(LINE_PX_PER_FONT_SCALE * self.config.font_scale)
        draw_text(canvas, lines, (self.config.panel_margin_px, height - self.config.panel_margin_px - line_px * len(lines)),
                  self.config.font_scale)

    # ------------------------------------------------------------------ panels

    def drone_panel(self, ctx: FrameContext, size: tuple[int, int]) -> np.ndarray:
        """The drone view; with `overlay_alpha`, every class except background tinted over it."""
        view = ctx.view
        if self.config.overlay_alpha > 0 and ctx.class_map is not None:
            class_map = cv2.resize(ctx.class_map.astype(np.uint8), (view.shape[1], view.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            blended = cv2.addWeighted(view, 1 - self.config.overlay_alpha, self.config.class_rgb[class_map],
                                      self.config.overlay_alpha, 0)
            view = np.where((class_map != BACKGROUND)[..., None], blended, view)
        return fit(view, size)

    def segmentation_panel(self, ctx: FrameContext, size: tuple[int, int]) -> np.ndarray:
        if ctx.class_map is None:
            panel = np.full((size[1], size[0], 3), BACKGROUND_RGB, np.uint8)
            draw_text(panel, ["", "no segmentation yet"], (TEXT_PAD_PX, TEXT_PAD_PX), self.config.font_scale)
            return panel
        return fit(self.config.class_rgb[ctx.class_map], size, cv2.INTER_NEAREST)

    def heatmap_panel(self, ctx: FrameContext, size: tuple[int, int]) -> np.ndarray:
        if ctx.fix is None or ctx.fix.heatmap is None:
            return np.full((size[1], size[0], 3), BACKGROUND_RGB, np.uint8)
        heat = ctx.fix.heatmap
        finite = np.isfinite(heat)
        scaled = np.zeros(heat.shape, np.uint8)
        if finite.any() and np.ptp(heat[finite]) > 0:
            scaled[finite] = np.round((heat[finite] - heat[finite].min()) / np.ptp(heat[finite]) * 255).astype(np.uint8)
        colours = cv2.cvtColor(cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
        colours[~finite] = BACKGROUND_RGB
        return fit(colours, size, cv2.INTER_NEAREST)

    def map_panel(self, ctx: FrameContext, size: tuple[int, int]) -> np.ndarray:
        cfg = self.config
        center, px_per_m, panel = self._map_background(ctx, size)
        first = self._trail_start(ctx.step)
        for track in cfg.tracks:
            lat_column, lon_column = TRACK_COLUMNS[track]
            rows = self.result.track.iloc[first:ctx.step + 1]
            points = to_panel_px(center, px_per_m, size, rows[lat_column].to_numpy(), rows[lon_column].to_numpy())
            cv2.polylines(panel, [points], False, cfg.track_rgb[track], cfg.track_width_px, cv2.LINE_AA)
            cv2.circle(panel, tuple(int(v) for v in points[-1]), cfg.track_width_px + 3, cfg.track_rgb[track], -1,
                       cv2.LINE_AA)
        if cfg.show_fixes:
            self._draw_fixes(panel, ctx.step, center, px_per_m, size)
        if cfg.show_footprint:
            self._draw_footprint(panel, ctx.pose, center, px_per_m, size)
        if cfg.show_legend:
            legend = [(TRACK_LABELS[t], cfg.track_rgb[t]) for t in cfg.tracks]
            line_px = round(LINE_PX_PER_FONT_SCALE * cfg.font_scale)
            for k, (label, rgb) in enumerate(legend):
                draw_text(panel, [label], (TEXT_PAD_PX, size[1] - TEXT_PAD_PX - line_px * (len(legend) - k)),
                          cfg.font_scale, rgb)
        return panel

    def _map_background(self, ctx: FrameContext, size: tuple[int, int]) -> tuple[LatLon, float, np.ndarray]:
        """North-up map crop: `(centre, px per metre, pixels)`. The route view is computed once."""
        if self.config.map_mode == "follow":
            px_per_m = size[0] / (2 * self.config.map_radius_m)
            return ctx.pose.position, px_per_m, self._map_crop(ctx.pose.position, px_per_m, size)
        if self._route_view is None:
            center, px_per_m = self._route_frame(size)
            self._route_view = (center, px_per_m, self._map_crop(center, px_per_m, size))
        center, px_per_m, pixels = self._route_view
        return center, px_per_m, pixels.copy()

    def _map_crop(self, center: LatLon, px_per_m: float, size: tuple[int, int]) -> np.ndarray:
        width_m, height_m = size[0] / px_per_m, size[1] / px_per_m
        crop = self.map_raster.crop_rotated(center, 0.0, width_m, height_m, out_mpp=1 / px_per_m, allow_outside=True)
        return np.ascontiguousarray(cv2.resize(crop[..., :3], size, interpolation=cv2.INTER_AREA))

    def _route_frame(self, size: tuple[int, int]) -> tuple[LatLon, float]:
        """Centre and scale that fit every track of the whole flight, plus a margin."""
        lats = np.concatenate([self.result.track[TRACK_COLUMNS[t][0]] for t in self.config.tracks])
        lons = np.concatenate([self.result.track[TRACK_COLUMNS[t][1]] for t in self.config.tracks])
        center = LatLon((lats.min() + lats.max()) / 2, (lons.min() + lons.max()) / 2)
        east_m, north_m = east_north_m_array(center, lats, lons)
        span_m = 2 * (max(np.abs(east_m).max(), np.abs(north_m).max() * size[0] / size[1])
                      + self.config.map_route_margin_m)
        return center, size[0] / span_m

    def _trail_start(self, step: int) -> int:
        if self.config.trail_s is None:
            return 0
        times = self.result.track["t_s"].to_numpy()
        return int(np.searchsorted(times, times[step] - self.config.trail_s))

    def _draw_fixes(self, panel: np.ndarray, step: int, center: LatLon, px_per_m: float, size: tuple[int, int]) -> None:
        for fix_step in self.result.fix_steps:
            if fix_step > step:
                break
            fix = self.result.fixes[fix_step]
            x, y = to_panel_px(center, px_per_m, size, np.array([fix.position.lat]), np.array([fix.position.lon]))[0]
            if fix.applied:
                cv2.circle(panel, (int(x), int(y)), 3, self.config.track_rgb["estimate"], -1, cv2.LINE_AA)
            else:
                cv2.drawMarker(panel, (int(x), int(y)), REJECT_RGB, cv2.MARKER_TILTED_CROSS, 7, 1, cv2.LINE_AA)

    def _draw_footprint(self, panel: np.ndarray, pose: Pose, center: LatLon, px_per_m: float,
                        size: tuple[int, int]) -> None:
        """What the camera sees, at the true pose."""
        width_m, height_m = self.camera.footprint_m(pose.altitude_m)
        corners_m = [rotate_to_north_up(u, v, pose.yaw_deg)
                     for u, v in ((-width_m / 2, -height_m / 2), (width_m / 2, -height_m / 2),
                                  (width_m / 2, height_m / 2), (-width_m / 2, height_m / 2))]
        east_m, north_m = east_north_m_array(center, np.array([pose.position.lat]), np.array([pose.position.lon]))
        points = np.array([[size[0] / 2 + (east_m[0] + right) * px_per_m, size[1] / 2 - (north_m[0] - down) * px_per_m]
                           for right, down in corners_m], np.int32)
        cv2.polylines(panel, [points], True, TEXT_RGB, 1, cv2.LINE_AA)


def to_panel_px(center: LatLon, px_per_m: float, size: tuple[int, int], lats: np.ndarray,
                lons: np.ndarray) -> np.ndarray:
    """`(N, 2)` int32 pixel positions in a north-up panel centred on `center`."""
    east_m, north_m = east_north_m_array(center, lats, lons)
    return np.stack([size[0] / 2 + east_m * px_per_m, size[1] / 2 - north_m * px_per_m], axis=1).round().astype(np.int32)


def corner_origin(corner: str, panel_size: tuple[int, int], frame_size: tuple[int, int], margin_px: int) -> tuple[int, int]:
    (panel_w, panel_h), (frame_w, frame_h) = panel_size, frame_size
    x = margin_px if corner.endswith("left") else frame_w - margin_px - panel_w
    y = margin_px if corner.startswith("top") else frame_h - margin_px - panel_h
    return x, y


def fit(image: np.ndarray, size: tuple[int, int], interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """Letterbox `image` into `size` (w, h), keeping its aspect ratio."""
    width, height = size
    scale = min(width / image.shape[1], height / image.shape[0])
    new_w, new_h = max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))
    if interpolation == cv2.INTER_AREA and scale > 1:
        interpolation = cv2.INTER_LINEAR
    resized = cv2.resize(np.ascontiguousarray(image), (new_w, new_h), interpolation=interpolation)
    canvas = np.full((height, width, 3), BACKGROUND_RGB, np.uint8)
    x, y = (width - new_w) // 2, (height - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def draw_text(image: np.ndarray, lines: list[str], origin: tuple[int, int], font_scale: float,
              rgb: tuple[int, int, int] = TEXT_RGB) -> None:
    """Lines of text on a dark box, top-left at `origin`."""
    line_px = round(LINE_PX_PER_FONT_SCALE * font_scale)
    x, y = origin
    for k, line in enumerate(lines):
        if not line:
            continue
        (text_w, text_h), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        top = y + k * line_px
        cv2.rectangle(image, (x, top), (x + text_w + 2 * TEXT_PAD_PX, top + line_px), BACKGROUND_RGB, -1)
        cv2.putText(image, line, (x + TEXT_PAD_PX, top + text_h + (line_px - text_h) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, rgb, 1, cv2.LINE_AA)

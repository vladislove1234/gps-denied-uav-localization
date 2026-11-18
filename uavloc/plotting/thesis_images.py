"""Real-image figures of the thesis: panel grids of crops, class maps and masks, heatmaps, particle clouds, routes on maps.

Every figure is a grid of equal panels with equal gaps, drawn by `create_panel_grid`; column titles sit above the
columns, and any figure that shows classes carries the same class legend (`CLASS_COLOURS`). The module draws only:
what is drawn comes from `uavloc.simulation.figure_data` and the datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from PIL import Image

from uavloc.data.satellite import GeoImage
from uavloc.geo import LatLon, east_north_m_array, offset_latlon, rotate_to_north_up
from uavloc.plotting.thesis_style import FULL_WIDTH_CM, new_figure, save_thesis_figure

CM_PER_INCH = 2.54
NO_LABEL_COLOUR = "#ffffff"
IGNORE_LABEL = 255

# Fixed class palette, in the order of `uavloc.features.classes.CLASSES`: background, building, track, woody, field, water.
CLASS_NAMES = ("Тло", "Будівлі", "Дороги", "Дерева", "Поле", "Вода")
CLASS_COLOURS = ("#d9d8d2", "#d62728", "#f2a900", "#2e7d32", "#9467bd", "#1f77b4")
CLASS_RGB = np.array([[int(colour[i:i + 2], 16) for i in (1, 3, 5)] for colour in CLASS_COLOURS], np.uint8)
NO_LABEL_RGB = (255, 255, 255)
SKELETON_RGB = (0, 0, 0)

# Route colours, as on the route maps of the other thesis figures: truth green, INS red, the system blue.
TRUTH_COLOUR = "#2ca02c"
INS_COLOUR = "#d62728"
SYSTEM_COLOUR = "#1f77b4"
WINDOW_COLOUR = "#ffd400"
FRAME_EDGE_COLOUR = "#9a9a96"
HEATMAP_CMAP = "inferno"
HEATMAP_FLOOR_PERCENTILE = 2.0
HEATMAP_ALPHA_MIN = 0.1
HEATMAP_ALPHA_MAX = 0.85
MARKER_EDGE_COLOUR = "white"
PARTICLE_SIZE_PT2 = 2.0
MAX_PNG_BYTES = 1_900_000
ROUTE_PANEL_ASPECT = 0.75
PALETTE_COLOURS = 256
DPI = 300
OVERVIEW_HEIGHT_CM = 4.6
OVERVIEW_MARGIN_M = 350.0

TITLE_ROW_CM = 0.6
TITLE_PAD_PT = 3.0
COLOURBAR_STRIP_CM = 2.45         # legend line, then a colourbar with its ticks and label above
COLOURBAR_Y_CM = 1.35
LINE_CM = 0.45                     # height of one legend or label line
FRAME_LINE_PT = 0.5
SCALE_BAR_HEIGHT_FRAC = 0.018


@dataclass(frozen=True)
class PanelLayout:
    """A grid of `n_rows x n_cols` equal panels, `gap_cm` apart in both directions, with strips for what surrounds it."""

    n_rows: int
    n_cols: int
    panel_aspect: float                 # height / width of one panel
    title_cm: float = TITLE_ROW_CM      # strip above the first row for the column titles
    bottom_cm: float = 0.0              # strip below the last row for legends and colourbars
    left_cm: float = 0.0                # strip left of the first column for row labels
    gap_cm: float = 0.2
    pad_cm: float = 0.1
    header_cm: float = 0.0              # a full-width panel above the grid, added by the caller with `header_axes`

    @property
    def panel_width_cm(self) -> float:
        free = FULL_WIDTH_CM - 2 * self.pad_cm - self.left_cm - (self.n_cols - 1) * self.gap_cm
        return free / self.n_cols

    @property
    def height_cm(self) -> float:
        rows = self.n_rows * self.panel_width_cm * self.panel_aspect + (self.n_rows - 1) * self.gap_cm
        return 2 * self.pad_cm + self.header_cm + self.title_cm + rows + self.bottom_cm


def create_panel_grid(layout: PanelLayout) -> tuple[Figure, np.ndarray]:
    """Full-width figure and its `(n_rows, n_cols)` axes, placed by hand so every gap is exactly `gap_cm`."""
    fig, first = new_figure("full", layout.height_cm / FULL_WIDTH_CM)
    first.remove()
    fig.set_layout_engine("none")
    width_cm, height_cm = FULL_WIDTH_CM, layout.height_cm
    panel_w, panel_h = layout.panel_width_cm, layout.panel_width_cm * layout.panel_aspect
    axes = np.empty((layout.n_rows, layout.n_cols), object)
    for row in range(layout.n_rows):
        for col in range(layout.n_cols):
            left = layout.pad_cm + layout.left_cm + col * (panel_w + layout.gap_cm)
            top = layout.pad_cm + layout.header_cm + layout.title_cm + row * (panel_h + layout.gap_cm)
            axes[row, col] = fig.add_axes((left / width_cm, 1 - (top + panel_h) / height_cm,
                                           panel_w / width_cm, panel_h / height_cm))
            frame_panel(axes[row, col])
    return fig, axes


def header_axes(fig: Figure, layout: PanelLayout) -> Axes:
    """The full-width panel above the grid; its bottom edge is `gap_cm` above the column titles."""
    height_cm = layout.height_cm
    inner_cm = layout.header_cm - layout.gap_cm
    ax = fig.add_axes((layout.pad_cm / FULL_WIDTH_CM, 1 - (layout.pad_cm + inner_cm) / height_cm,
                       (FULL_WIDTH_CM - 2 * layout.pad_cm) / FULL_WIDTH_CM, inner_cm / height_cm))
    frame_panel(ax)
    return ax


def frame_panel(ax: Axes) -> None:
    """No ticks or grid, a thin grey frame."""
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(FRAME_LINE_PT)
        spine.set_edgecolor(FRAME_EDGE_COLOUR)


def class_rgb(class_ids: np.ndarray) -> np.ndarray:
    """`(H, W)` class ids -> `(H, W, 3)` uint8 in the fixed palette; unlabelled pixels (255) white."""
    out = np.full((*class_ids.shape, 3), NO_LABEL_RGB, np.uint8)
    known = class_ids != IGNORE_LABEL
    out[known] = CLASS_RGB[class_ids[known]]
    return out


def show_photo(ax: Axes, image: np.ndarray, extent: tuple[float, float, float, float] | None = None) -> AxesImage:
    return ax.imshow(image, extent=extent, interpolation="lanczos" if image.shape[0] < 200 else "antialiased")


def show_class_map(ax: Axes, class_ids: np.ndarray,
                   extent: tuple[float, float, float, float] | None = None) -> AxesImage:
    return ax.imshow(class_rgb(class_ids), extent=extent, interpolation="nearest")


def set_column_titles(axes_row: Sequence[Axes], titles: Sequence[str]) -> None:
    for ax, title in zip(axes_row, titles):
        ax.set_title(title, pad=TITLE_PAD_PT)


def set_row_labels(axes_column: Sequence[Axes], labels: Sequence[str]) -> None:
    for ax, label in zip(axes_column, labels):
        ax.set_ylabel(label, labelpad=TITLE_PAD_PT)


def add_class_legend(fig: Figure, class_ids: Sequence[int], y_cm: float = 0.1, with_no_label: bool = False) -> None:
    """One row of class swatches along the bottom edge, `y_cm` above it."""
    handles = [Patch(facecolor=CLASS_COLOURS[i], edgecolor=FRAME_EDGE_COLOUR, linewidth=FRAME_LINE_PT,
                     label=CLASS_NAMES[i]) for i in class_ids]
    if with_no_label:
        handles.append(Patch(facecolor=NO_LABEL_COLOUR, edgecolor=FRAME_EDGE_COLOUR, linewidth=FRAME_LINE_PT,
                             label="Без розмітки"))
    add_marker_legend(fig, handles, y_cm)


def add_marker_legend(fig: Figure, handles: Sequence, y_cm: float = 0.1) -> None:
    height_cm = fig.get_figheight() * CM_PER_INCH
    fig.legend(handles=list(handles), loc="lower center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, y_cm / height_cm), handlelength=1.4, columnspacing=1.6, handletextpad=0.5,
               borderaxespad=0.0)


def add_colourbar(fig: Figure, mappable: ScalarMappable, label: str, y_cm: float, width_frac: float = 0.4) -> None:
    """A thin horizontal colourbar centred along the bottom, its label above it."""
    height_cm = fig.get_figheight() * CM_PER_INCH
    bar_h_cm = 0.22
    cax = fig.add_axes(((1 - width_frac) / 2, y_cm / height_cm, width_frac, bar_h_cm / height_cm))
    bar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    bar.outline.set_linewidth(FRAME_LINE_PT)
    bar.set_label(label, labelpad=2)
    bar.ax.xaxis.set_label_position("top")


def add_scale_bar(ax: Axes, length_m: float, units_per_m: float = 1.0, loc: str = "lower left") -> None:
    """A black scale bar with a white halo in data units, labelled in metres (`м`)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    margin_x, margin_y = abs(x1 - x0) * 0.04, abs(y1 - y0) * 0.05
    left = min(x0, x1) + margin_x if "left" in loc else max(x0, x1) - margin_x - length_m * units_per_m
    bottom = min(y0, y1) + margin_y if "lower" in loc else max(y0, y1) - margin_y
    ax.plot([left, left + length_m * units_per_m], [bottom, bottom], color="black", lw=2.2, solid_capstyle="butt",
            path_effects=_halo(), zorder=9)
    ax.text(left + length_m * units_per_m / 2, bottom + abs(y1 - y0) * 0.02, f"{length_m:g} м", ha="center",
            va="bottom", color="black", path_effects=_halo(), zorder=9)


def _halo() -> list:
    from matplotlib import patheffects
    return [patheffects.withStroke(linewidth=2.6, foreground="white")]


def rotated_box_en_m(centre_en_m: tuple[float, float], size_m: tuple[float, float], yaw_deg: float) -> np.ndarray:
    """`(4, 2)` east/north corners of a heading-up box of `size_m = (width, height)` rotated to `yaw_deg`."""
    half_w, half_h = size_m[0] / 2, size_m[1] / 2
    corners = []
    for right, down in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)):
        east, south = rotate_to_north_up(right, down, yaw_deg)
        corners.append((centre_en_m[0] + east, centre_en_m[1] - south))
    return np.array(corners)


def north_up_crop(raster: GeoImage, centre: LatLon, size_m: tuple[float, float]) -> tuple[np.ndarray, tuple]:
    """North-up pixels around `centre` and their extent `(left, right, bottom, top)` in metres from `centre`.

    Ground outside the raster is white, not black.
    """
    pixels = raster.crop_rotated(centre, 0.0, *size_m, allow_outside=True)
    covered = raster.with_pixels(np.full(raster.pixels.shape[:2], 255, np.uint8))
    outside = covered.crop_rotated(centre, 0.0, *size_m, allow_outside=True) == 0
    pixels[outside] = NO_LABEL_RGB
    half_w, half_h = size_m[0] / 2, size_m[1] / 2
    return pixels, (-half_w, half_w, -half_h, half_h)


def keep_inside_raster(centre_en_m: np.ndarray, size_m: tuple[float, float], raster: GeoImage,
                       origin: LatLon) -> np.ndarray:
    """Shift a window's centre (east/north from `origin`) so the window stays on `raster`, where it is small enough."""
    corners = (raster.top_left, raster.bottom_right)
    east, north = east_north_m_array(origin, np.array([c.lat for c in corners]), np.array([c.lon for c in corners]))
    half = np.array(size_m) / 2
    low, high = np.array([east[0], north[1]]) + half, np.array([east[1], north[0]]) - half
    return np.where(low <= high, np.clip(centre_en_m, low, high), centre_en_m)


def fit_aspect(size_m: tuple[float, float], aspect: float) -> tuple[float, float]:
    """Grow the shorter side of `size_m` so that height / width equals `aspect`."""
    width, height = size_m
    return (width, width * aspect) if height / width < aspect else (height / aspect, height)


def draw_footprint_map(ax: Axes, raster: GeoImage, truth: LatLon, prior: LatLon, yaw_deg: float,
                       footprint_m: tuple[float, float], window_m: tuple[float, float], view_m: tuple[float, float]) -> None:
    """Satellite crop centred on the true position with the camera footprint, the search window and both positions."""
    pixels, extent = north_up_crop(raster, truth, view_m)
    show_photo(ax, pixels, extent)
    prior_en = tuple(east_north_m_array(truth, np.array([prior.lat]), np.array([prior.lon])))
    prior_en = (float(prior_en[0][0]), float(prior_en[1][0]))
    footprint = rotated_box_en_m((0.0, 0.0), footprint_m, yaw_deg)
    window = rotated_box_en_m(prior_en, window_m, yaw_deg)
    ax.add_patch(Polygon(window, closed=True, fill=False, edgecolor=WINDOW_COLOUR, linewidth=2.0, linestyle=(0, (5, 3))))
    ax.add_patch(Polygon(footprint, closed=True, facecolor=TRUTH_COLOUR, alpha=0.18, edgecolor="none"))
    ax.add_patch(Polygon(footprint, closed=True, fill=False, edgecolor=TRUTH_COLOUR, linewidth=2.0))
    front = footprint[[0, 1]].mean(axis=0) - footprint.mean(axis=0)
    ax.annotate("", xy=front * 1.0, xytext=(0, 0), arrowprops={"arrowstyle": "-|>", "color": TRUTH_COLOUR, "lw": 1.6})
    ax.plot(0, 0, "o", color=TRUTH_COLOUR, markeredgecolor=MARKER_EDGE_COLOUR, markersize=6, zorder=6)
    ax.plot(*prior_en, "P", color=INS_COLOUR, markeredgecolor=MARKER_EDGE_COLOUR, markersize=8, zorder=6)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    add_scale_bar(ax, 20.0)


def footprint_map_handles() -> list:
    return [Line2D([], [], color=TRUTH_COLOUR, lw=2.0, label="Кадр камери"),
            Line2D([], [], color=WINDOW_COLOUR, lw=2.0, linestyle=(0, (5, 3)), label="Вікно пошуку"),
            Line2D([], [], color=TRUTH_COLOUR, marker="o", markeredgecolor="white", linestyle="", label="Істинна позиція"),
            Line2D([], [], color=INS_COLOUR, marker="P", markeredgecolor="white", linestyle="",
                   label="Приблизна позиція")]


def overlay_heatmap(ax: Axes, heatmap: np.ndarray, extent: tuple[float, float, float, float],
                    norm: Normalize) -> AxesImage:
    """The score surface over the satellite patch; the higher the score, the more opaque, on one scale for all panels."""
    scaled = np.clip(np.nan_to_num(norm(np.ma.masked_invalid(heatmap)).filled(0.0)), 0.0, 1.0)
    rgba = cm.get_cmap(HEATMAP_CMAP)(scaled)
    rgba[..., 3] = HEATMAP_ALPHA_MIN + (HEATMAP_ALPHA_MAX - HEATMAP_ALPHA_MIN) * scaled
    return ax.imshow(rgba, extent=extent, interpolation="nearest")


def route_legend_handles() -> list:
    return [Line2D([], [], color=TRUTH_COLOUR, lw=2.0, label="Істинний маршрут"),
            Line2D([], [], color=INS_COLOUR, lw=2.0, linestyle=(0, (4, 2)), label="ІНС"),
            Line2D([], [], color=SYSTEM_COLOUR, lw=2.0, label="Система"),
            Line2D([], [], color="white", marker="^", markeredgecolor="black", linestyle="", label="Початок")]


def draw_route_panel(ax: Axes, basemap: GeoImage, tracks_en_m: dict[str, np.ndarray], origin: LatLon,
                     margin_m: float) -> None:
    """Truth, INS and system tracks (`(N, 2)` east/north from `origin`) over the map, framed with `margin_m` around them."""
    points = np.concatenate(list(tracks_en_m.values()))
    low, high = points.min(axis=0) - margin_m, points.max(axis=0) + margin_m
    size_m = fit_aspect(tuple(float(v) for v in (high - low)), axes_aspect(ax))
    centre_en = keep_inside_raster((low + high) / 2, size_m, basemap, origin)
    centre = offset_latlon(origin, float(centre_en[0]), float(centre_en[1]))
    pixels, extent = north_up_crop(basemap, centre, size_m)
    extent = (extent[0] + centre_en[0], extent[1] + centre_en[0], extent[2] + centre_en[1], extent[3] + centre_en[1])
    show_photo(ax, pixels, extent)
    styles = {"truth": (TRUTH_COLOUR, "-", 2.4), "ins": (INS_COLOUR, (0, (4, 2)), 2.0), "system": (SYSTEM_COLOUR, "-", 1.8)}
    for name, track in tracks_en_m.items():
        colour, style, width = styles[name]
        ax.plot(track[:, 0], track[:, 1], color=colour, linestyle=style, linewidth=width, solid_capstyle="round")
        ax.plot(*track[-1], "o", color=colour, markeredgecolor="white", markersize=5, zorder=6)
    ax.plot(*tracks_en_m["truth"][0], "^", color="white", markeredgecolor="black", markersize=7, zorder=7)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    add_scale_bar(ax, nice_length_m(size_m[0] / 4))


def nice_length_m(target_m: float) -> float:
    """The largest of 1, 2, 5 times a power of ten that does not exceed `target_m`."""
    power = 10 ** np.floor(np.log10(target_m))
    return float(max(m * power for m in (1, 2, 5) if m * power <= target_m))



@dataclass(frozen=True)
class LabelledCrop:
    title: str
    image: np.ndarray       # (H, W, 3) uint8
    label: np.ndarray       # (H, W) class ids


def draw_dataset_labels(crops: Sequence[LabelledCrop]) -> Figure:
    """One column per village: the crop above, its label below."""
    layout = PanelLayout(2, len(crops), crops[0].image.shape[0] / crops[0].image.shape[1], bottom_cm=LINE_CM + 0.3, left_cm=0.6)
    fig, axes = create_panel_grid(layout)
    for col, crop in enumerate(crops):
        show_photo(axes[0, col], crop.image)
        show_class_map(axes[1, col], crop.label)
    set_column_titles(axes[0], [crop.title for crop in crops])
    set_row_labels(axes[:, 0], ["Знімок", "Розмітка"])
    shown = sorted({int(i) for crop in crops for i in np.unique(crop.label) if i != IGNORE_LABEL})
    add_class_legend(fig, shown)
    return fig


@dataclass(frozen=True)
class DatedPair:
    place: str
    dates: tuple[str, str]
    images: tuple[np.ndarray, np.ndarray]


def corner_label(ax: Axes, text: str) -> None:
    ax.text(0.03, 0.05, text, transform=ax.transAxes, ha="left", va="bottom", color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0})


def draw_crossdate_pairs(pairs: Sequence[DatedPair]) -> Figure:
    """One row per place, the same ground on an older and a newer date."""
    first = pairs[0].images[0]
    layout = PanelLayout(len(pairs), 2, first.shape[0] / first.shape[1], left_cm=0.6)
    fig, axes = create_panel_grid(layout)
    for row, pair in enumerate(pairs):
        for col in range(2):
            show_photo(axes[row, col], pair.images[col])
            corner_label(axes[row, col], pair.dates[col])
    set_column_titles(axes[0], ["Давніша дата", "Новіша дата"])
    set_row_labels(axes[:, 0], [pair.place for pair in pairs])
    return fig


def draw_road_skeleton(image: np.ndarray, road_mask: np.ndarray, skeleton: np.ndarray) -> Figure:
    """Crop, road mask, and the mask with its skeleton on top."""
    layout = PanelLayout(1, 3, road_mask.shape[0] / road_mask.shape[1], title_cm=TITLE_ROW_CM, bottom_cm=LINE_CM + 0.3)
    fig, axes = create_panel_grid(layout)
    road = np.where(road_mask, 2, 0).astype(np.uint8)
    show_photo(axes[0, 0], image)
    show_class_map(axes[0, 1], road)
    rgb = class_rgb(road)
    rgb[skeleton] = SKELETON_RGB
    axes[0, 2].imshow(rgb, interpolation="nearest")
    set_column_titles(axes[0], ["Знімок", "Маска дороги", "Скелет дороги"])
    add_marker_legend(fig, [Patch(facecolor=CLASS_COLOURS[0], edgecolor=FRAME_EDGE_COLOUR, label="Тло"),
                            Patch(facecolor=CLASS_COLOURS[2], edgecolor=FRAME_EDGE_COLOUR, label="Дороги"),
                            Patch(facecolor="black", label="Скелет")])
    return fig


@dataclass(frozen=True)
class LossMaskRow:
    image: np.ndarray
    label: np.ndarray
    predictions: tuple[np.ndarray, ...]     # one class map per loss configuration


def draw_loss_masks(rows: Sequence[LossMaskRow], loss_titles: Sequence[str]) -> Figure:
    """Per crop: image, label, then the prediction of each loss configuration."""
    layout = PanelLayout(len(rows), 2 + len(loss_titles), rows[0].image.shape[0] / rows[0].image.shape[1], title_cm=0.95, bottom_cm=LINE_CM + 0.3)
    fig, axes = create_panel_grid(layout)
    for row, sample in enumerate(rows):
        show_photo(axes[row, 0], sample.image)
        show_class_map(axes[row, 1], sample.label)
        for col, prediction in enumerate(sample.predictions, start=2):
            show_class_map(axes[row, col], prediction)
    set_column_titles(axes[0], ["Знімок", "Розмітка", *loss_titles])
    shown = {int(i) for r in rows for a in (r.label, *r.predictions) for i in np.unique(a) if i != IGNORE_LABEL}
    add_class_legend(fig, sorted(shown), with_no_label=any((r.label == IGNORE_LABEL).any() for r in rows))
    return fig


def draw_class_maps(frame: np.ndarray, frame_ids: np.ndarray, patch: np.ndarray, patch_ids: np.ndarray) -> Figure:
    """Onboard frame and satellite patch of the same ground, each with its class map."""
    layout = PanelLayout(2, 2, frame.shape[0] / frame.shape[1], left_cm=0.6, bottom_cm=LINE_CM + 0.3)
    fig, axes = create_panel_grid(layout)
    show_photo(axes[0, 0], frame)
    show_photo(axes[0, 1], patch)
    show_class_map(axes[1, 0], frame_ids)
    show_class_map(axes[1, 1], patch_ids)
    set_column_titles(axes[0], ["Бортовий кадр", "Супутниковий знімок"])
    set_row_labels(axes[:, 0], ["Зображення", "Карта класів"])
    add_class_legend(fig, sorted({int(i) for i in np.unique(np.concatenate([frame_ids.ravel(), patch_ids.ravel()]))}))
    return fig


def draw_sim_frames(columns: dict[str, Sequence[np.ndarray]]) -> Figure:
    """Simulated onboard frames: one column per terrain, one row per village."""
    n_rows = len(next(iter(columns.values())))
    first = next(iter(columns.values()))[0]
    layout = PanelLayout(n_rows, len(columns), first.shape[0] / first.shape[1])
    fig, axes = create_panel_grid(layout)
    for col, frames in enumerate(columns.values()):
        for row, frame in enumerate(frames):
            show_photo(axes[row, col], frame)
    set_column_titles(axes[0], list(columns))
    return fig


def patch_axes_limits(ax: Axes, patch_size_m: tuple[float, float]) -> tuple[float, float, float, float]:
    """Show a heading-up patch in metres from its centre, `down` increasing downwards; returns its imshow extent."""
    half_w, half_h = patch_size_m[0] / 2, patch_size_m[1] / 2
    ax.set_xlim(-half_w, half_w)
    ax.set_ylim(half_h, -half_h)
    return -half_w, half_w, half_h, -half_h


def mark_positions(ax: Axes, true_m: tuple[float, float], found_m: tuple[float, float] | None) -> None:
    ax.plot(*true_m, "o", markerfacecolor="none", markeredgecolor=TRUTH_COLOUR, markeredgewidth=2.2, markersize=13,
            zorder=8)
    if found_m is not None:
        ax.plot(*found_m, "X", color=SYSTEM_COLOUR, markeredgecolor="white", markersize=9, zorder=8)


def position_legend_handles() -> list:
    return [Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor=TRUTH_COLOUR, markeredgewidth=2.2,
                   markersize=10, linestyle="", label="Істинна позиція"),
            Line2D([], [], marker="X", color=SYSTEM_COLOUR, markeredgecolor="white", markersize=8, linestyle="",
                   label="Знайдена позиція")]


def draw_heatmap_figure(frame: np.ndarray, patch: np.ndarray, patch_size_m: tuple[float, float],
                        heatmap: np.ndarray, heatmap_extent_m: tuple[float, float, float, float],
                        true_m: tuple[float, float], found_m: tuple[float, float]) -> Figure:
    """Onboard frame; satellite patch; the head's heatmap over the patch with the true and the found position."""
    layout = PanelLayout(1, 3, patch_size_m[1] / patch_size_m[0], title_cm=TITLE_ROW_CM,
                         bottom_cm=COLOURBAR_STRIP_CM)
    fig, axes = create_panel_grid(layout)
    show_photo(axes[0, 0], frame)
    extent = patch_axes_limits(axes[0, 1], patch_size_m)
    show_photo(axes[0, 1], patch, extent)
    patch_axes_limits(axes[0, 2], patch_size_m)
    show_photo(axes[0, 2], patch, extent)
    norm = Normalize(vmin=float(np.nanpercentile(heatmap, HEATMAP_FLOOR_PERCENTILE)), vmax=float(np.nanmax(heatmap)))
    overlay_heatmap(axes[0, 2], heatmap, heatmap_extent_m, norm)
    for ax in axes[0, 1:]:
        mark_positions(ax, true_m, found_m)
    set_column_titles(axes[0], ["Бортовий кадр", "Супутниковий знімок", "Теплова карта"])
    add_marker_legend(fig, position_legend_handles(), y_cm=0.1)
    add_colourbar(fig, ScalarMappable(norm=norm, cmap=HEATMAP_CMAP), "Оцінка збігу (z-оцінка)", y_cm=COLOURBAR_Y_CM)
    return fig


def draw_particle_cloud(cloud_before_m: np.ndarray, weights_before: np.ndarray, cloud_after_m: np.ndarray,
                        weights_after: np.ndarray, patch: np.ndarray, patch_size_m: tuple[float, float],
                        heatmap: np.ndarray, heatmap_extent_m: tuple[float, float, float, float],
                        true_m: tuple[float, float]) -> Figure:
    """The cloud before and after one update, over the satellite patch and the heatmap that updated it."""
    layout = PanelLayout(1, 2, patch_size_m[1] / patch_size_m[0], bottom_cm=COLOURBAR_STRIP_CM)
    fig, axes = create_panel_grid(layout)
    norm = Normalize(vmin=float(np.nanpercentile(heatmap, HEATMAP_FLOOR_PERCENTILE)), vmax=float(np.nanmax(heatmap)))
    for ax, cloud, weights in zip(axes[0], (cloud_before_m, cloud_after_m), (weights_before, weights_after)):
        extent = patch_axes_limits(ax, patch_size_m)
        show_photo(ax, patch, extent)
        overlay_heatmap(ax, heatmap, heatmap_extent_m, norm)
        sizes = PARTICLE_SIZE_PT2 * weights / weights.mean()
        ax.scatter(cloud[:, 0], cloud[:, 1], s=sizes, facecolor="white", edgecolor="none", alpha=0.7,
                   zorder=7)
        mark_positions(ax, true_m, tuple(weights @ cloud))
    set_column_titles(axes[0], ["До оновлення", "Після оновлення"])
    add_marker_legend(fig, [Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor="black",
                                   markeredgewidth=0.4, markersize=5, linestyle="", label="Частинки"),
                            *position_legend_handles()[:1],
                            Line2D([], [], marker="X", color=SYSTEM_COLOUR, markeredgecolor="white", markersize=8,
                                   linestyle="", label="Оцінка (середнє частинок)")], y_cm=0.1)
    add_colourbar(fig, ScalarMappable(norm=norm, cmap=HEATMAP_CMAP), "Оцінка збігу (z-оцінка)", y_cm=COLOURBAR_Y_CM)
    return fig


@dataclass(frozen=True)
class RoutePanel:
    title: str
    basemap: GeoImage
    tracks_en_m: dict[str, np.ndarray]     # "truth", "ins", "system": (N, 2) east/north from `origin`
    origin: LatLon


def draw_route_tracks(panels: Sequence[RoutePanel], margin_m: float) -> Figure:
    """One column per simulated route, on the satellite map: the true route, the INS and the system."""
    layout = PanelLayout(1, len(panels), ROUTE_PANEL_ASPECT, title_cm=0.95, bottom_cm=LINE_CM + 0.3)
    fig, axes = create_panel_grid(layout)
    for ax, panel in zip(axes[0], panels):
        draw_route_panel(ax, panel.basemap, panel.tracks_en_m, panel.origin, margin_m)
    set_column_titles(axes[0], [panel.title for panel in panels])
    add_marker_legend(fig, route_legend_handles())
    return fig


@dataclass(frozen=True)
class RouteZoom:
    """A stretch of a long route shown enlarged around sample `index`."""

    title: str
    basemap: GeoImage
    index: int


def axes_aspect(ax: Axes) -> float:
    """Height over width of the axes box as drawn."""
    box = ax.get_window_extent()
    return float(box.height / box.width)


def zoom_window(tracks_en_m: dict[str, np.ndarray], index: int, width_m: float, aspect: float) -> tuple:
    """`(centre_en_m, size_m)` of the window holding the truth, INS and system positions at sample `index`."""
    points = np.array([track[index] for track in tracks_en_m.values()])
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    return centre, (width_m, width_m * aspect)


def draw_zoom_panel(ax: Axes, zoom: RouteZoom, tracks_en_m: dict[str, np.ndarray], origin: LatLon,
                    width_m: float, trail_samples: int) -> None:
    """Window around the three positions at `zoom.index`, with the last `trail_samples` of each track."""
    centre, size_m = zoom_window(tracks_en_m, zoom.index, width_m, axes_aspect(ax))
    pixels, extent = north_up_crop(zoom.basemap, offset_latlon(origin, *map(float, centre)), size_m)
    extent = (extent[0] + centre[0], extent[1] + centre[0], extent[2] + centre[1], extent[3] + centre[1])
    show_photo(ax, pixels, extent)
    styles = {"truth": (TRUTH_COLOUR, "-", 2.6), "ins": (INS_COLOUR, (0, (4, 2)), 2.0), "system": (SYSTEM_COLOUR, "-", 1.8)}
    for name, track in tracks_en_m.items():
        colour, style, width = styles[name]
        trail = track[max(zoom.index - trail_samples, 0):zoom.index + 1]
        ax.plot(trail[:, 0], trail[:, 1], color=colour, linestyle=style, linewidth=width)
        ax.plot(*track[zoom.index], "o", color=colour, markeredgecolor="white", markersize=7, zorder=6)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    add_scale_bar(ax, nice_length_m(width_m / 4))


def draw_long_route_map(overview: GeoImage, tracks_en_m: dict[str, np.ndarray], origin: LatLon,
                        zooms: Sequence[RouteZoom], zoom_width_m: float, trail_samples: int) -> Figure:
    """The whole route on a coarse map above, and enlarged stretches below where the three positions are drawn apart."""
    layout = PanelLayout(1, len(zooms), 0.75, title_cm=0.95, header_cm=OVERVIEW_HEIGHT_CM + 0.2,
                         bottom_cm=LINE_CM + 0.3)
    fig, axes = create_panel_grid(layout)
    top = header_axes(fig, layout)
    points = np.concatenate(list(tracks_en_m.values()))
    low, high = points.min(axis=0) - OVERVIEW_MARGIN_M, points.max(axis=0) + OVERVIEW_MARGIN_M
    centre_en = (low + high) / 2
    size_m = fit_aspect(tuple(map(float, high - low)), axes_aspect(top))
    pixels, extent = north_up_crop(overview, offset_latlon(origin, *map(float, centre_en)), size_m)
    extent = (extent[0] + centre_en[0], extent[1] + centre_en[0], extent[2] + centre_en[1], extent[3] + centre_en[1])
    show_photo(top, pixels, extent)
    for name, (colour, style, width) in {"truth": (TRUTH_COLOUR, "-", 2.4), "ins": (INS_COLOUR, (0, (4, 2)), 1.8),
                                         "system": (SYSTEM_COLOUR, "-", 1.4)}.items():
        top.plot(tracks_en_m[name][:, 0], tracks_en_m[name][:, 1], color=colour, linestyle=style, linewidth=width)
    top.plot(*tracks_en_m["truth"][0], "^", color="white", markeredgecolor="black", markersize=7, zorder=7)
    for number, zoom in enumerate(zooms):
        mark = tracks_en_m["truth"][zoom.index]
        top.annotate(str(number + 1), mark, xytext=(0, 9), textcoords="offset points", ha="center", va="bottom",
                     path_effects=_halo(), zorder=9)
        top.plot(*mark, "o", color="white", markeredgecolor="black", markersize=4, zorder=8)
    top.set_xlim(extent[0], extent[1])
    top.set_ylim(extent[2], extent[3])
    add_scale_bar(top, 2000.0, loc="lower right")
    for number, (ax, zoom) in enumerate(zip(axes[0], zooms)):
        draw_zoom_panel(ax, zoom, tracks_en_m, origin, zoom_width_m, trail_samples)
        ax.set_title(f"{number + 1}. {zoom.title}", pad=TITLE_PAD_PT)
    add_marker_legend(fig, route_legend_handles())
    return fig


def save_figure_png(fig: Figure, path: Path, max_bytes: int = MAX_PNG_BYTES) -> tuple[int, int]:
    """`save_thesis_figure`, then a 256-colour palette if the photo-heavy PNG is larger than `max_bytes`."""
    size_px = save_thesis_figure(fig, path)
    if path.stat().st_size > max_bytes:
        palette = Image.open(path).convert("RGB").quantize(colors=PALETTE_COLOURS, dither=Image.Dither.FLOYDSTEINBERG)
        palette.save(path, optimize=True, dpi=(DPI, DPI))
    return size_px

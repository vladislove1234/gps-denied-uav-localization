"""Step-by-step pictures of one position correction: inputs, feature maps, heatmap, match, map.

Patch panels are drawn in metres of the heading-up patch centred on the prior
(`right_m`, `down_m`), so the heatmap, footprints and true position overlay exactly.
"""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import Circle, Ellipse, Polygon, Rectangle

from uavloc.data.drone_frames import DroneFrame
from uavloc.data.satellite import GeoImage
from uavloc.features.segmentation import CLASSES
from uavloc.geo import LatLon, east_north_m, haversine_m, rotate_to_north_up
from uavloc.matching.correlator import Correction, FeatureExtractor, class_ncc
from uavloc.matching.replay import Replay

SEG_PALETTE = np.array([[40, 40, 40], [230, 30, 30], [255, 215, 0], [20, 160, 40], [200, 140, 255], [30, 110, 255]],
                       np.uint8)  # background, building, track, woody, field, water
TRUE_COLOUR = "lime"
PRIOR_COLOUR = "deepskyblue"
CORRECTED_COLOURS = ("red", "orange", "magenta", "yellow")
HEATMAP_ALPHA = 0.5
DRONE_ONLY_COLOUR = (255, 255, 255)
MAP_ONLY_COLOUR = (120, 120, 120)
MARKER_SIZE = 9
GRID_UPSCALE = 3
GRID_COLUMNS = 4
GRID_GAP_PX = 4
TRUE_RGB = (255, 255, 255)
ARGMAX_RGB = (255, 40, 40)


def max_activation(fmap: torch.Tensor) -> np.ndarray:
    """`(1, C, H, W)` → `(H, W)` max over channels, the 2025 feature debug view."""
    return fmap[0].max(0).values.float().cpu().numpy()


def class_colours(fmap: torch.Tensor, class_ids: list[int]) -> np.ndarray:
    """`(1, C, H, W)` class probabilities → `(H, W, 3)` argmax class colours."""
    return SEG_PALETTE[np.asarray(class_ids)[fmap[0].argmax(0).cpu().numpy()]]


def patch_extent_m(size_m: tuple[float, float]) -> tuple[float, float, float, float]:
    width_m, height_m = size_m
    return -width_m / 2, width_m / 2, height_m / 2, -height_m / 2


def show_patch(ax: Axes, image: np.ndarray, size_m: tuple[float, float], title: str) -> AxesImage:
    shown = ax.imshow(image, extent=patch_extent_m(size_m))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("right, m", fontsize=8)
    ax.set_ylabel("down, m", fontsize=8)
    return shown


def mark_points(ax: Axes, correction: Correction, true_offset_m: tuple[float, float], colour: str) -> None:
    ax.plot(0, 0, "+", color=PRIOR_COLOUR, ms=MARKER_SIZE + 4, mew=2, label="prior")
    ax.plot(*true_offset_m, "o", color=TRUE_COLOUR, ms=MARKER_SIZE, mfc="none", mew=2, label="true")
    ax.plot(*correction.offset_m, "x", color=colour, ms=MARKER_SIZE, mew=2, label="corrected")


def plot_surface(ax: Axes, patch_rgb: np.ndarray, surface: np.ndarray, correction: Correction,
                 true_offset_m: tuple[float, float], colour: str, title: str, **imshow_kwargs) -> AxesImage:
    """Any score surface over the search range (same cells as `correction.heatmap`), on the patch."""
    show_patch(ax, patch_rgb, correction.patch_size_m, title)
    shown = ax.imshow(surface, extent=correction.heatmap_extent_m, alpha=HEATMAP_ALPHA, **{"cmap": "jet", **imshow_kwargs})
    mark_points(ax, correction, true_offset_m, colour)
    ax.set_xlim(*patch_extent_m(correction.patch_size_m)[:2])
    ax.set_ylim(*patch_extent_m(correction.patch_size_m)[2:])
    return shown


def plot_heatmap(ax: Axes, patch_rgb: np.ndarray, correction: Correction, true_offset_m: tuple[float, float],
                 colour: str, title: str) -> AxesImage:
    return plot_surface(ax, patch_rgb, correction.heatmap, correction, true_offset_m, colour, title)


def plot_uncertainty(ax: Axes, correction: Correction, colour: str, n_sigma: float = 2.0,
                     search_radius_m: float | None = None) -> None:
    """The posterior's `n_sigma` ellipse around its mean, its best cell (square), and a round window's edge."""
    s1, s2, e1, _ = correction.posterior.axes
    ax.add_patch(Ellipse(tuple(correction.posterior.mean_m), 2 * n_sigma * s1, 2 * n_sigma * s2,
                         angle=float(np.degrees(np.arctan2(e1[1], e1[0]))), fill=False, ec=colour, lw=1.5))
    ax.plot(*correction.map_offset_m, "s", color=colour, ms=MARKER_SIZE, mfc="none", mew=1.5, label="posterior best")
    if search_radius_m:
        ax.add_patch(Circle((0, 0), search_radius_m, fill=False, ec="white", ls="--", lw=1))


def footprint_rectangle(centre_m: tuple[float, float], footprint_m: tuple[float, float], colour: str,
                        style: str = "-") -> Rectangle:
    width_m, height_m = footprint_m
    return Rectangle((centre_m[0] - width_m / 2, centre_m[1] - height_m / 2), width_m, height_m,
                     fill=False, edgecolor=colour, lw=2, ls=style)


def plot_match(ax: Axes, patch_rgb: np.ndarray, correction: Correction, true_offset_m: tuple[float, float],
               footprint_m: tuple[float, float], colour: str, title: str) -> None:
    """The drone footprint where the correlator put it, where it really is, and where the prior said."""
    show_patch(ax, patch_rgb, correction.patch_size_m, title)
    ax.add_patch(footprint_rectangle((0, 0), footprint_m, PRIOR_COLOUR, "--"))
    ax.add_patch(footprint_rectangle(true_offset_m, footprint_m, TRUE_COLOUR))
    ax.add_patch(footprint_rectangle(correction.offset_m, footprint_m, colour))
    mark_points(ax, correction, true_offset_m, colour)


def search_window_polygon(frame: DroneFrame, prior: LatLon, patch_size_m: tuple[float, float]) -> Polygon:
    """The rotated satellite patch outline, in metres east/north of the true position."""
    prior_east_m, prior_north_m = east_north_m(frame.position, prior)
    half_w_m, half_h_m = patch_size_m[0] / 2, patch_size_m[1] / 2
    corners = []
    for du_m, dv_m in ((-half_w_m, -half_h_m), (half_w_m, -half_h_m), (half_w_m, half_h_m), (-half_w_m, half_h_m)):
        dx_m, dy_m = rotate_to_north_up(du_m, dv_m, frame.yaw_deg)
        corners.append((prior_east_m + dx_m, prior_north_m - dy_m))
    return Polygon(corners, fill=False, edgecolor=PRIOR_COLOUR, lw=1.5, ls="--")


def plot_overview(ax: Axes, satellite: GeoImage, frame: DroneFrame, prior: LatLon, corrected: dict[str, LatLon],
                  patch_size_m: tuple[float, float], margin_m: float) -> None:
    """North-up map around the truth: prior, search window, and each method's answer."""
    region = satellite.crop_rotated(frame.position, 0.0, 2 * margin_m, 2 * margin_m)
    ax.imshow(region, extent=(-margin_m, margin_m, -margin_m, margin_m))
    ax.add_patch(search_window_polygon(frame, prior, patch_size_m))
    ax.plot(*east_north_m(frame.position, prior), "+", color=PRIOR_COLOUR, ms=MARKER_SIZE + 4, mew=2, label="prior")
    ax.plot(0, 0, "o", color=TRUE_COLOUR, ms=MARKER_SIZE, mfc="none", mew=2, label="true")
    for (name, position), colour in zip(corrected.items(), CORRECTED_COLOURS):
        ax.plot(*east_north_m(frame.position, position), "x", color=colour, ms=MARKER_SIZE, mew=2, label=name)
    ax.set_xlabel("east, m", fontsize=8)
    ax.set_ylabel("north, m", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")


def plot_class_surface(ax: Axes, surface: np.ndarray, correction: Correction, true_offset_m: tuple[float, float],
                       colour: str, title: str) -> AxesImage:
    """One class's NCC over the search range only, opaque, on a symmetric scale of its own."""
    limit = max(float(np.abs(surface).max()), 1e-6)
    shown = ax.imshow(surface, extent=correction.heatmap_extent_m, cmap="RdBu_r", vmin=-limit, vmax=limit)
    mark_points(ax, correction, true_offset_m, colour)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("drone centre: right of prior, m", fontsize=8)
    ax.set_ylabel("down, m", fontsize=8)
    return shown


def class_overlay(drone: torch.Tensor, reference: torch.Tensor, track: int, woody: int) -> np.ndarray:
    """RGB overlay of two aligned class maps: track yellow (drone) / cyan (map) / white (both); woody orange / green / pale."""
    drone_ids, map_ids = drone[0].argmax(0).cpu().numpy(), reference[0].argmax(0).cpu().numpy()
    out = np.full((*drone_ids.shape, 3), 40, np.uint8)
    for cls, d_colour, m_colour, both in ((woody, (255, 140, 0), (20, 160, 40), (200, 255, 200)),
                                           (track, (255, 215, 0), (0, 200, 255), (255, 255, 255))):
        d, m = drone_ids == cls, map_ids == cls
        out[d & ~m], out[m & ~d], out[d & m] = d_colour, m_colour, both
    return out


def plot_review(r: Replay, satellite: GeoImage, title: str, map_label: str, margin_m: float) -> Figure:
    """The 2025 review sequence for one correction: patches, class maps, heatmap on the patch, positions on the map."""
    n_classes = r.drone_map.shape[1]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    patch = satellite.crop_rotated(r.prior, r.frame.yaw_deg, *r.correction.patch_size_m)
    show_patch(axes[0, 0], patch, r.correction.patch_size_m, f"Satellite Patch ({map_label})")
    for ax, image, name in [(axes[0, 1], r.frame.image(), "Drone Image"),
                            (axes[0, 2], class_colours(r.drone_map, list(range(n_classes))), "Drone Class Map"),
                            (axes[1, 0], class_colours(r.satellite_map, list(range(n_classes))), "Satellite Class Map")]:
        ax.imshow(image); ax.set_title(name, fontsize=9); ax.axis("off")
    fig.colorbar(plot_heatmap(axes[1, 1], patch, r.correction, r.true_offset_m, CORRECTED_COLOURS[0],
                              f"Match Heatmap, peak z {r.correction.peak_z:.2f}"), ax=axes[1, 1], shrink=.8)
    plot_overview(axes[1, 2], satellite, r.frame, r.prior, {"Corrected": r.correction.position}, r.correction.patch_size_m, margin_m)
    axes[1, 2].set_title("Estimate vs Corrected vs True", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.legend([Rectangle((0, 0), 1, 1, color=SEG_PALETTE[c] / 255) for c in range(len(SEG_PALETTE))],
               CLASSES, loc="lower center", ncol=len(CLASSES))
    fig.tight_layout(rect=(0, .03, 1, .97))
    return fig


def class_agreement(drone: torch.Tensor, reference: torch.Tensor) -> np.ndarray:
    """`(H, W, 3)` for two aligned class maps: class colour where both agree, white = drone only, grey = map only."""
    drone_ids, map_ids = drone[0].argmax(0).cpu().numpy(), reference[0].argmax(0).cpu().numpy()
    out = np.zeros((*drone_ids.shape, 3), np.uint8)
    both = (drone_ids == map_ids) & (drone_ids != 0)
    out[both] = SEG_PALETTE[drone_ids[both]]
    out[(drone_ids != 0) & ~both] = DRONE_ONLY_COLOUR
    out[(drone_ids == 0) & (map_ids != 0)] = MAP_ONLY_COLOUR
    return out


def aligned_class_scores(drone: torch.Tensor, reference: torch.Tensor, class_weights: tuple[float, ...]) -> dict[str, float]:
    """Per-class NCC of two aligned `(1, C, H, W)` maps, for the weighted classes, as the correlator scores one shift."""
    return {CLASSES[c]: float(class_ncc(drone[0, c].float().cpu(), reference[0, c].float().cpu())[0, 0])
            for c, w in enumerate(class_weights) if w > 0}


def plot_class_match(r: Replay, features: FeatureExtractor, class_weights: tuple[float, ...]) -> Figure:
    """How the drone class map lines up with the map's classes at the corrected and at the true position."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 7), gridspec_kw={"width_ratios": [1, 1, 1, .8]})
    for row, (name, position) in zip(axes, [("CORRECTED", r.correction.position), ("TRUE", r.frame.position)]):
        reference = features.satellite_patch(position, r.frame.yaw_deg, *r.footprint_m).cpu()
        scores = aligned_class_scores(r.drone_map, reference, class_weights)
        for ax, image, title in [(row[0], class_colours(r.drone_map, list(range(len(CLASSES)))), "drone classes"),
                                 (row[1], class_colours(reference, list(range(len(CLASSES)))), f"map classes at {name} position"),
                                 (row[2], class_agreement(r.drone_map, reference), "agreement: colour both, white drone only, grey map only")]:
            ax.imshow(image); ax.set_title(title, fontsize=9); ax.axis("off")
        row[3].barh(list(scores), list(scores.values()), color=[SEG_PALETTE[CLASSES.index(n)] / 255 for n in scores])
        row[3].set_xlim(-1, 1); row[3].axvline(0, color="k", lw=.5); row[3].set_title(f"class NCC at {name} position", fontsize=9)
        row[0].set_ylabel(name, fontsize=10)
    fig.tight_layout()
    return fig


def plot_model_comparison(replays: dict[str, Replay], satellite: GeoImage, title: str, margin_m: float) -> Figure:
    """One draw corrected by several models: a row of class maps and heatmap per model, every answer on one map."""
    first = next(iter(replays.values()))
    n_classes = first.drone_map.shape[1]
    patch = satellite.crop_rotated(first.prior, first.frame.yaw_deg, *first.correction.patch_size_m)
    fig = plt.figure(figsize=(24, 4.5 * (len(replays) + 1)))
    grid = fig.add_gridspec(len(replays) + 1, 4, width_ratios=(1, 1, 1, 1.4))
    top = fig.add_subplot(grid[0, 0])
    top.imshow(first.frame.image()); top.set_title("Drone Image", fontsize=9); top.axis("off")
    show_patch(fig.add_subplot(grid[0, 1]), patch, first.correction.patch_size_m, "Satellite Patch (search window)")
    answers = {}
    for i, ((name, r), colour) in enumerate(zip(replays.items(), CORRECTED_COLOURS), start=1):
        error_m = haversine_m(r.correction.position, r.frame.position)
        answers[f"{name}: {error_m:.1f} m, {r.correction.decision}"] = r.correction.position
        for j, (fmap, what) in enumerate(((r.drone_map, "drone classes"), (r.satellite_map, "map classes"))):
            ax = fig.add_subplot(grid[i, j])
            ax.imshow(class_colours(fmap, list(range(n_classes))))
            ax.set_title(f"{name}: {what}", fontsize=9); ax.axis("off")
        ax = fig.add_subplot(grid[i, 2])
        fig.colorbar(plot_heatmap(ax, patch, r.correction, r.true_offset_m, colour,
                                  f"{name}: peak z {r.correction.peak_z:.2f}, {r.correction.decision}"), ax=ax, shrink=.8)
    overview = fig.add_subplot(grid[:, 3])
    plot_overview(overview, satellite, first.frame, first.prior, answers, first.correction.patch_size_m, margin_m)
    overview.set_title("Prior vs corrected vs true", fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.legend([Rectangle((0, 0), 1, 1, color=SEG_PALETTE[c] / 255) for c in range(len(SEG_PALETTE))],
               CLASSES, loc="lower center", ncol=len(CLASSES))
    fig.tight_layout(rect=(0, .03, 1, .97))
    return fig



def grid_px(row: float, col: float) -> tuple[int, int]:
    """Pixel `(x, y)` of a surface cell's centre in a `heatmap_grid` tile."""
    return int((col + 0.5) * GRID_UPSCALE), int((row + 0.5) * GRID_UPSCALE)


def heatmap_grid(logits: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """`(N, H, W)` surface logits -> `(H', W', 3)` uint8 grid of softmax maps for TensorBoard.

    White circle: true cell. Red cross: argmax. Each tile is scaled to its own maximum.
    """
    tiles = []
    for surface, (row, col) in zip(logits, cells):
        relative_p = np.exp(surface - surface.max())
        tile = (plt.get_cmap("viridis")(relative_p)[..., :3] * 255).astype(np.uint8)
        tile = np.ascontiguousarray(cv2.resize(tile, None, fx=GRID_UPSCALE, fy=GRID_UPSCALE,
                                               interpolation=cv2.INTER_NEAREST))
        cv2.circle(tile, grid_px(row, col), 2 * GRID_UPSCALE, TRUE_RGB, 1)
        cv2.drawMarker(tile, grid_px(*np.unravel_index(surface.argmax(), surface.shape)), ARGMAX_RGB,
                       cv2.MARKER_CROSS, 3 * GRID_UPSCALE, 1)
        tiles.append(np.pad(tile, ((GRID_GAP_PX, GRID_GAP_PX), (GRID_GAP_PX, GRID_GAP_PX), (0, 0)), constant_values=255))
    tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % GRID_COLUMNS)
    rows = [np.concatenate(tiles[i:i + GRID_COLUMNS], 1) for i in range(0, len(tiles), GRID_COLUMNS)]
    return np.concatenate(rows, 0)

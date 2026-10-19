from __future__ import annotations

import dataclasses
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from nibabel.gifti.gifti import GiftiImage
from nibabel.loadsave import load as nib_load
from nilearn import plotting
import numpy as np

from yawn.plots.layout import add_colorbar
from yawn.plots.layout import annotate_time
from yawn.plots.layout import build_side_axes
from yawn.plots.layout import draw_label_legend


class FrameMapSource(Protocol):
    n_frames: int

    def get_frame_maps(self, frame_index: int) -> tuple[np.ndarray, np.ndarray]: ...


@dataclasses.dataclass(frozen=True)
class SurfaceOverlay:
    left_plot: list[np.ndarray]
    right_plot: list[np.ndarray]
    output_tag: str
    # for Cifti atlases
    cmap: Colormap | str | None = None
    labels: list[tuple[int, str, tuple[float, float, float, float]]] | None = None
    # for Gifti ROIs
    labels_left: list[tuple[int, str, tuple[float, float, float, float]]] | None = None
    labels_right: list[tuple[int, str, tuple[float, float, float, float]]] | None = None


@dataclasses.dataclass(frozen=True)
class SurfaceCameraConfig:
    elev: float | None = None
    azim: float | None = None


@dataclasses.dataclass(frozen=True)
class SurfaceRenderOptions:
    views: list[str]
    figure_size: tuple[float, float]
    dpi: int
    ncols: int | None = None
    cmap: str = "coolwarm"
    vmin: float | None = None
    vmax: float | None = None
    mesh_alpha: float = 1.0
    surf_zoom: float = 1.8
    black_bg: bool = False
    colorbar: bool = True
    colorbar_side: str = "right"
    title_template: str | None = None
    source_format: str = "unknown"
    atlas_ignore_zero: bool = True
    atlas_legend: bool = False
    atlas_legend_max_items: int = 40
    roi_legend: bool = False
    exclude_medial_wall: bool = False
    time_annotate: bool = False
    time_s: float | None = None
    threshold: float | None = None
    left_camera: SurfaceCameraConfig = dataclasses.field(
        default_factory=SurfaceCameraConfig
    )
    right_camera: SurfaceCameraConfig = dataclasses.field(
        default_factory=SurfaceCameraConfig
    )


@dataclasses.dataclass(frozen=True)
class RenderedSurfaceFigure:
    figure: Figure
    panel_axes: list[Axes]
    colorbar_ax: Axes | None
    legend_ax: Axes | None


def _sample_finite_values(
    data: np.ndarray, *, max_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample finite values from a large array without loading everything."""
    flat = np.asarray(data, dtype=float).ravel()
    n = int(flat.size)
    if n == 0 or max_samples <= 0:
        return np.asarray([], dtype=float)
    if n <= max_samples:
        vals = flat[np.isfinite(flat)]
        return vals.astype(float, copy=False)

    collected: list[np.ndarray] = []
    remaining = max_samples
    for _ in range(25):
        if remaining <= 0:
            break
        draw = min(max(remaining * 2, 1024), 200_000)
        idx = rng.integers(0, n, size=draw, dtype=np.int64)
        vals = flat[idx]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if vals.size > remaining:
            vals = vals[:remaining]
        collected.append(vals.astype(float, copy=False))
        remaining -= int(vals.size)

    if not collected:
        return np.asarray([], dtype=float)
    return np.concatenate(collected, axis=0)


def compute_intensity_bounds(
    metric_source: FrameMapSource | None,
    *,
    selected_index: int,
    intensity_mode: str,
    p_low: float,
    p_high: float,
    max_samples: int,
    max_total_samples: int,
    vmin_arg: float | None,
    vmax_arg: float | None,
) -> tuple[float | None, float | None]:
    """Estimate intensity bounds for a surface metric source."""
    if metric_source is None:
        return None, None
    if vmin_arg is not None and vmax_arg is not None:
        return float(vmin_arg), float(vmax_arg)

    rng = np.random.default_rng(0)
    vmin: float | None = None
    vmax: float | None = None
    if intensity_mode == "global" and metric_source.n_frames > 1:
        samples_per_frame = max(
            1, int(max_total_samples // max(metric_source.n_frames, 1))
        )
        collected: list[np.ndarray] = []
        for frame_index in range(metric_source.n_frames):
            left_map, right_map = metric_source.get_frame_maps(frame_index)
            collected.append(
                _sample_finite_values(
                    np.concatenate((left_map, right_map), axis=0),
                    max_samples=samples_per_frame,
                    rng=rng,
                )
            )
        all_samples = (
            np.concatenate([sample for sample in collected if sample.size > 0], axis=0)
            if collected
            else np.asarray([], dtype=float)
        )
        if all_samples.size > 0:
            vmin = float(np.percentile(all_samples, p_low))
            vmax = float(np.percentile(all_samples, p_high))
    else:
        left_map, right_map = metric_source.get_frame_maps(selected_index)
        samples = _sample_finite_values(
            np.concatenate((left_map, right_map), axis=0),
            max_samples=max_samples,
            rng=rng,
        )
        if samples.size > 0:
            vmin = float(np.percentile(samples, p_low))
            vmax = float(np.percentile(samples, p_high))
    if vmin_arg is not None:
        vmin = float(vmin_arg)
    if vmax_arg is not None:
        vmax = float(vmax_arg)
    return vmin, vmax


def validate_map_against_mesh(
    values: np.ndarray | list[np.ndarray],
    surf_mesh: Any,
    *,
    kind: str,
    source_type: Literal["metric", "overlay"],
    source_format: Literal["gifti", "cifti"],
) -> None:
    """Ensure a surface map matches the vertex count of its mesh."""
    if source_format not in ("gifti", "cifti"):
        raise ValueError(f"Unsupported source format: {source_format}")
    # Cifti atlas files are a single array
    if source_format == "cifti" or source_type == "metric":
        _values_array = [values]
    # gifti ROI files are a list of arrays, one per ROI
    else:
        _values_array = values

    for _values in _values_array:
        if not isinstance(_values, np.ndarray):
            raise ValueError(
                f"{kind} {source_format} values must be a numpy array, got {type(_values)}"
            )
        coords = np.asarray(surf_mesh[0])
        if int(_values.size) != int(coords.shape[0]):
            raise ValueError(
                f"{kind} {source_format} vertex count ({int(_values.size)}) does "
                f"not match mesh vertices ({int(coords.shape[0])})"
            )


@lru_cache(maxsize=1)
def _load_medial_wall_mask_pair() -> tuple[np.ndarray, np.ndarray]:
    """Load the cached medial-wall masks for both hemispheres."""
    root = Path(__file__).resolve().parent.parent.parent
    left_path = (
        root / "template" / "fsLR_hemi-L_den-32k_desc-nomedialwall_dparc.label.gii"
    )
    right_path = (
        root / "template" / "fsLR_hemi-R_den-32k_desc-nomedialwall_dparc.label.gii"
    )
    left_img = nib_load(str(left_path))
    right_img = nib_load(str(right_path))
    left_gifti = cast(GiftiImage, left_img)
    right_gifti = cast(GiftiImage, right_img)
    return (
        np.asarray(left_gifti.darrays[0].data, dtype=float).ravel(),
        np.asarray(right_gifti.darrays[0].data, dtype=float).ravel(),
    )


def _apply_medial_wall_mask_to_map(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mask out values that fall outside the medial wall keep region."""
    if values.shape != mask.shape:
        return values
    masked = np.asarray(values, dtype=float).copy()
    masked[np.asarray(mask, dtype=float) <= 0] = np.nan
    return masked


def _apply_surf_zoom(ax: Axes, zoom: float) -> None:
    """Apply a lightweight zoom adjustment to a 3D surface axis."""
    try:
        z = float(zoom)
    except Exception:
        return
    if not np.isfinite(z) or z <= 0:
        return
    try:
        box_aspect = getattr(cast(Any, ax), "set_box_aspect", None)
        if callable(box_aspect):
            box_aspect(None, zoom=z)
            return
        if hasattr(cast(Any, ax), "dist"):
            setattr(cast(Any, ax), "dist", max(1.0, 10.0 / z))
    except Exception:
        return


def _resolve_contour_colors(
    roi_map: np.ndarray,
    labels: list[tuple[int, str, tuple[float, float, float, float]]],
) -> tuple[list[float], list[tuple[float, float, float, float]]] | None:
    """Build explicit contour levels and colors for atlas overlays."""
    roi_values = np.asarray(roi_map, dtype=float)
    levels = [float(value) for value in np.unique(roi_values[np.isfinite(roi_values)])]
    if not levels:
        return None

    # Contours need at least 2 levels to draw boundaries; fall back to cmap for single-level
    if len(levels) < 2:
        return None

    label_colors = {
        int(key): cast(tuple[float, float, float, float], rgba)
        for key, _name, rgba in labels
    }
    colors: list[tuple[float, float, float, float]] = []
    for level in levels:
        color = label_colors.get(int(level))
        if color is None:
            return None
        colors.append(color)
    return levels, colors


def _render_surface_panels(
    *,
    axes: list[Axes],
    surf_left_mesh: Any,
    surf_right_mesh: Any,
    surf_left_sulc: np.ndarray,
    surf_right_sulc: np.ndarray,
    stat_map_left: np.ndarray | None,
    stat_map_right: np.ndarray | None,
    overlay: SurfaceOverlay | None,
    options: SurfaceRenderOptions,
    index: int,
) -> None:
    """Render each requested view/hemi panel into the provided axes."""
    # Load medial-wall masks if requested.
    medial_wall_masks: tuple[np.ndarray, np.ndarray] | None = None
    if bool(options.exclude_medial_wall):
        medial_wall_masks = _load_medial_wall_mask_pair()
    # all panels are rendered in a single loop, left and right hemispheres interleaved
    panels = [(str(view), hemi) for view in options.views for hemi in ("L", "R")]
    for panel_idx, (view, hemi) in enumerate(panels):
        if panel_idx >= len(axes):
            break
        ax = axes[panel_idx]
        surf_mesh = surf_left_mesh if hemi == "L" else surf_right_mesh
        surf_sulc = surf_left_sulc if hemi == "L" else surf_right_sulc
        stat_map = stat_map_left if hemi == "L" else stat_map_right
        # apply medial-wall mask to stat map if requested
        if medial_wall_masks is not None and stat_map is not None:
            stat_map = _apply_medial_wall_mask_to_map(
                stat_map, medial_wall_masks[0 if hemi == "L" else 1]
            )
        # resolve the view into a concrete camera setting if provided
        resolved_view = _resolve_surface_view(options, view=str(view), hemi=str(hemi))
        # compute the title for this panel if requested
        title = None
        if options.title_template:
            title = str(options.title_template).format(
                index=int(index),
                panel=int(panel_idx),
                view=str(view),
                hemi=str(hemi),
                time=("" if options.time_s is None else f"{float(options.time_s):.3f}"),
                format=options.source_format,
            )
        # render the stat map if provided, otherwise render the sulcal map
        if stat_map is not None:
            plotting.plot_surf_stat_map(
                surf_mesh,
                stat_map,
                hemi=("left" if hemi == "L" else "right"),
                view=cast(Any, resolved_view),
                cmap=str(options.cmap),
                alpha=float(options.mesh_alpha),
                vmin=options.vmin,
                vmax=options.vmax,
                colorbar=False,
                title=title,
                figure=ax.figure,
                axes=ax,
                bg_map=surf_sulc,
                bg_on_data=False,
                threshold=options.threshold,
            )
        elif title is not None:
            ax.set_title(title)

        # plot the overlays if provided - Gifti ROIs and Cifti atlases are handled differently
        if overlay is not None:
            overlay_maps = overlay.left_plot if hemi == "L" else overlay.right_plot
            # if no stat map is provided, render the sulcal map as a background
            if stat_map is None:
                plotting.plot_surf(
                    surf_mesh,
                    bg_map=surf_sulc,
                    hemi=("left" if hemi == "L" else "right"),
                    view=cast(Any, resolved_view),
                    alpha=None,
                    bg_on_data=False,
                    colorbar=False,
                    title=None,
                    figure=ax.figure,
                    axes=ax,
                )
            # plot atlas overlays as contours on top of the sulcal map
            if overlay.output_tag == "atlas":
                # overlay maps are expected to be a single map for atlas overlays, and labels are required
                labels = overlay.labels
                if labels is None or len(labels) == 0:
                    raise ValueError(
                        "Atlas overlay requires labels, but none were provided"
                    )

                # resolve contour levels and colors for the atlas overlay
                contour_spec = _resolve_contour_colors(overlay_maps[0], labels)
                if contour_spec is None:
                    raise ValueError(
                        "Atlas contour overlays require explicit colors for each network level"
                    )
                contour_levels, contour_colors = contour_spec
                # plot the atlas overlay as contours on top of the sulcal map
                plotting.plot_surf_contours(
                    surf_mesh,
                    overlay_maps[0],
                    hemi=("left" if hemi == "L" else "right"),
                    view=cast(Any, resolved_view),
                    levels=contour_levels,
                    colors=contour_colors,
                    title=None,
                    figure=ax.figure,
                    axes=ax,
                )
            # plot Gifti ROI overlays as contours on top of the sulcal map
            elif overlay.output_tag == "roi":
                # get labels for the current hemisphere if provided
                labels = overlay.labels_left if hemi == "L" else overlay.labels_right
                if labels is None or len(labels) == 0:
                    raise ValueError(
                        f"ROI overlay requires labels for the {'left' if hemi == 'L' else 'right'} hemisphere, but none were provided"
                    )
                # loop through overlay maps and render them
                for overlay_map, label in zip(overlay_maps, labels):
                    if medial_wall_masks is not None:
                        overlay_map = _apply_medial_wall_mask_to_map(
                            overlay_map, medial_wall_masks[0 if hemi == "L" else 1]
                        )
                    plotting.plot_surf_contours(
                        surf_mesh,
                        overlay_map,
                        hemi=("left" if hemi == "L" else "right"),
                        view=cast(Any, resolved_view),
                        levels=[1.0],  # contour level for ROI
                        labels=[label[1]],  # label for this ROI
                        colors=[label[2]],  # color for this ROI
                        title=None,
                        figure=ax.figure,
                        axes=ax,
                    )
            else:
                raise ValueError(
                    f"Unsupported overlay output tag: {overlay.output_tag}"
                )

        _apply_surf_zoom(ax, float(options.surf_zoom))


def _resolve_surface_view(
    options: SurfaceRenderOptions, *, view: str, hemi: str
) -> str | tuple[float, float]:
    """Resolve a view name into a concrete camera setting if provided."""
    hemi_camera = options.left_camera if hemi == "L" else options.right_camera
    elev = hemi_camera.elev
    azim = hemi_camera.azim
    if elev is None and azim is None:
        return view
    if elev is None or azim is None:
        raise ValueError(
            "Explicit surface camera control requires both elevation and azimuth. "
            "Provide both global camera values."
        )
    return float(elev), float(azim)


def render_surface_figure(
    *,
    surf_left_mesh: Any,
    surf_right_mesh: Any,
    surf_left_sulc: np.ndarray,
    surf_right_sulc: np.ndarray,
    stat_map_left: np.ndarray | None,
    stat_map_right: np.ndarray | None,
    overlay: SurfaceOverlay | None,
    options: SurfaceRenderOptions,
    index: int,
    legend_kwargs: dict[str, Any] | None = None,
    colorbar_kwargs: dict[str, Any] | None = None,
) -> RenderedSurfaceFigure:
    """Render a complete surface snapshot figure."""
    panels = [(str(view), hemi) for view in options.views for hemi in ("L", "R")]
    if not panels:
        raise RuntimeError("No surface panels to render")

    ncols = int(options.ncols) if options.ncols is not None else len(panels)
    if ncols <= 0:
        raise ValueError(f"--ncols must be positive, got {ncols}")
    nrows_maps = int(np.ceil(len(panels) / ncols))

    overlay_is_atlas = overlay is not None and overlay.output_tag == "atlas"
    if overlay_is_atlas:
        legend_enabled = bool(options.atlas_legend and overlay and overlay.labels)
        legend_labels = overlay.labels if overlay is not None else []
        legend_max_items = int(options.atlas_legend_max_items)
        ignore_zero = bool(options.atlas_ignore_zero)
    else:
        legend_enabled = bool(options.roi_legend and overlay and overlay.labels)
        legend_labels = []
        if overlay is not None and overlay.labels_left is not None:
            legend_labels.extend(overlay.labels_left)
        if overlay is not None and overlay.labels_right is not None:
            legend_labels.extend(overlay.labels_right)
        legend_max_items = None
        ignore_zero = False

    want_cbar = bool(options.colorbar) and (
        options.vmin is not None and options.vmax is not None
    )
    want_legend = legend_enabled
    side = str(options.colorbar_side)
    want_side_col = want_cbar or want_legend
    ncols_total = ncols + (1 if want_side_col else 0)
    surf_col0 = 1 if (want_side_col and side == "left") else 0
    side_col = 0 if (want_side_col and side == "left") else (ncols_total - 1)

    side_width_ratio = 0.12
    if want_legend and want_cbar:
        side_width_ratio = 0.18
    elif want_legend:
        side_width_ratio = 0.22

    fig = plt.figure(figsize=options.figure_size)
    if options.black_bg:
        fig.patch.set_facecolor("black")

    width_ratios, gs, ax_cbar, ax_legend = build_side_axes(
        fig=fig,
        nrows_maps=nrows_maps,
        ncols=ncols,
        ncols_total=ncols_total,
        side_col=side_col,
        side=side,
        want_side_col=want_side_col,
        want_cbar=want_cbar,
        want_legend=want_legend,
        side_width_ratio=side_width_ratio,
    )
    _ = width_ratios

    axes_flat = [
        fig.add_subplot(gs[row, col + surf_col0], projection="3d")
        for row in range(nrows_maps)
        for col in range(ncols)
    ]

    if options.time_annotate:
        if options.time_s is None:
            raise ValueError("time_annotate requires time_s to be provided")
        annotate_time(fig=fig, time_s=float(options.time_s), black_bg=options.black_bg)
    _render_surface_panels(
        axes=axes_flat,
        surf_left_mesh=surf_left_mesh,
        surf_right_mesh=surf_right_mesh,
        surf_left_sulc=surf_left_sulc,
        surf_right_sulc=surf_right_sulc,
        stat_map_left=stat_map_left,
        stat_map_right=stat_map_right,
        overlay=overlay,
        options=options,
        index=index,
    )

    if ax_cbar is not None and options.vmin is not None and options.vmax is not None:
        add_colorbar(
            fig=fig,
            cax=ax_cbar,
            cmap=str(options.cmap),
            vmin=float(options.vmin),
            vmax=float(options.vmax),
            colorbar_kwargs=colorbar_kwargs,
        )
    if ax_legend is not None and legend_labels:
        draw_label_legend(
            ax_legend,
            legend_labels,
            ignore_zero=ignore_zero,
            max_items=legend_max_items,
            legend_kwargs=legend_kwargs,
        )

    fig.subplots_adjust(left=0.02, right=0.9, bottom=0.06, top=0.95)
    return RenderedSurfaceFigure(
        figure=fig,
        panel_axes=axes_flat[: len(panels)],
        colorbar_ax=ax_cbar,
        legend_ax=ax_legend,
    )


def render_surface_into_axes(
    *,
    axes: list[Axes],
    colorbar_axis: Axes | None = None,
    legend_axis: Axes | None = None,
    legend_kwargs: dict[str, Any] | None = None,
    colorbar_kwargs: dict[str, Any] | None = None,
    surf_left_mesh: Any,
    surf_right_mesh: Any,
    surf_left_sulc: np.ndarray,
    surf_right_sulc: np.ndarray,
    stat_map_left: np.ndarray | None,
    stat_map_right: np.ndarray | None,
    overlay: SurfaceOverlay | None,
    options: SurfaceRenderOptions,
    index: int,
) -> list[Axes]:
    """Render a surface snapshot into existing axes."""
    panels = [(str(view), hemi) for view in options.views for hemi in ("L", "R")]
    if not panels:
        raise RuntimeError("No surface panels to render")
    if len(axes) < len(panels):
        raise ValueError(f"Expected at least {len(panels)} axes, got {len(axes)}")

    if options.time_annotate:
        if options.time_s is None:
            raise ValueError("time_annotate requires time_s to be provided")
        annotate_time(
            fig=cast(Figure, axes[0].figure),
            time_s=float(options.time_s),
            black_bg=options.black_bg,
        )

    _render_surface_panels(
        axes=axes,
        surf_left_mesh=surf_left_mesh,
        surf_right_mesh=surf_right_mesh,
        surf_left_sulc=surf_left_sulc,
        surf_right_sulc=surf_right_sulc,
        stat_map_left=stat_map_left,
        stat_map_right=stat_map_right,
        overlay=overlay,
        options=options,
        index=index,
    )
    if (
        colorbar_axis is not None
        and options.vmin is not None
        and options.vmax is not None
    ):
        add_colorbar(
            fig=cast(Figure, colorbar_axis.figure),
            cax=colorbar_axis,
            cmap=str(options.cmap),
            vmin=float(options.vmin),
            vmax=float(options.vmax),
            colorbar_kwargs=colorbar_kwargs,
        )

    overlay_is_atlas = overlay is not None and overlay.output_tag == "atlas"
    if overlay_is_atlas:
        legend_enabled = bool(options.atlas_legend and overlay and overlay.labels)
        legend_labels = overlay.labels if overlay is not None else []
        legend_max_items = int(options.atlas_legend_max_items)
        ignore_zero = bool(options.atlas_ignore_zero)
    else:
        legend_enabled = bool(options.roi_legend and overlay and overlay.labels)
        legend_labels = []
        if overlay is not None and overlay.labels_left is not None:
            legend_labels.extend(overlay.labels_left)
        if overlay is not None and overlay.labels_right is not None:
            legend_labels.extend(overlay.labels_right)
        legend_max_items = None
        ignore_zero = False

    if legend_axis is not None and legend_enabled and legend_labels:
        draw_label_legend(
            legend_axis,
            legend_labels,
            ignore_zero=ignore_zero,
            max_items=legend_max_items,
            legend_kwargs=legend_kwargs,
        )
    return axes[: len(panels)]

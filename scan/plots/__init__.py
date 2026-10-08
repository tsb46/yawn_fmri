"""Plotting utilities for scan_physio."""

from scan.plots.network_summary import plot_network_summary
from scan.plots.snapshot import (
    SurfaceSnapshotPlotter,
    build_snapshot_plotter,
    load_snapshot_scene,
    select_snapshot_output_path,
)
from scan.plots.surface import (
    RenderedSurfaceFigure,
    SurfaceCameraConfig,
    SurfaceOverlay,
    SurfaceRenderOptions,
    compute_intensity_bounds,
    render_surface_figure,
    validate_map_against_mesh,
)
from scan.plots.timecourse import (
    TimeSeriesMarker,
    build_timecourse_marker,
    markers_from_serialized_annotations,
    nearest_sample_index,
    plot_timeseries,
    serialize_timecourse_annotations,
)
from scan.plots.timecourse_dash import (
    create_timecourse_annotation_app,
    run_timecourse_annotation_app,
)

__all__ = [
    "RenderedSurfaceFigure",
    "SurfaceCameraConfig",
    "SurfaceOverlay",
    "SurfaceRenderOptions",
    "SurfaceSnapshotPlotter",
    "TimeSeriesMarker",
    "build_snapshot_plotter",
    "build_timecourse_marker",
    "compute_intensity_bounds",
    "create_timecourse_annotation_app",
    "load_snapshot_scene",
    "markers_from_serialized_annotations",
    "nearest_sample_index",
    "plot_network_summary",
    "plot_timeseries",
    "render_surface_figure",
    "run_timecourse_annotation_app",
    "select_snapshot_output_path",
    "serialize_timecourse_annotations",
    "validate_map_against_mesh",
]

from __future__ import annotations

"""Interactive time-series plotting helpers built on Plotly.

The public helper keeps Plotly imports local so the core package remains usable
without the optional visualization dependency installed.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSeriesPlotData:
    """Normalized time-series data ready for plotting."""

    time: np.ndarray
    values: np.ndarray
    labels: tuple[str, ...]


@dataclass(frozen=True)
class TimeSeriesMarker:
    """Metadata for a manually annotated sample."""

    sample_index: int
    time_seconds: float
    values: tuple[float, ...]
    label: str | None = None


def _to_1d_array(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array


def _coerce_labels(
    data: np.ndarray | pd.DataFrame,
    column_labels: Sequence[str] | None,
) -> tuple[str, ...]:
    if isinstance(data, pd.DataFrame):
        labels = tuple(str(label) for label in data.columns)
        if column_labels is None:
            return labels
        column_labels = tuple(str(label) for label in column_labels)
        if len(column_labels) != data.shape[1]:
            raise ValueError(
                "column_labels must match the number of columns in the DataFrame."
            )
        return column_labels

    if column_labels is None:
        raise ValueError(
            "column_labels are required when data is provided as an array."
        )

    column_labels = tuple(str(label) for label in column_labels)
    if len(column_labels) != data.shape[1]:
        raise ValueError("column_labels must match the second dimension of data.")
    return column_labels


def _coerce_values(data: np.ndarray | pd.DataFrame) -> np.ndarray:
    if isinstance(data, pd.DataFrame):
        values = data.to_numpy()
    else:
        values = np.asarray(data)
    if values.ndim != 2:
        raise ValueError("data must be two-dimensional with shape (time, signals).")
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("data must contain numeric values.") from exc


def _coerce_time(
    data: np.ndarray | pd.DataFrame,
    time: Sequence[Any] | np.ndarray | None,
    n_samples: int,
) -> np.ndarray:
    if time is None:
        if isinstance(data, pd.DataFrame):
            return np.asarray(data.index)
        return np.arange(n_samples)

    array = _to_1d_array(time, name="time")
    if array.shape[0] != n_samples:
        raise ValueError(
            "time must have the same length as the number of rows in data."
        )
    return array


def _scale_column(
    values: np.ndarray, scale_mode: Literal["none", "zscore", "minmax"]
) -> np.ndarray:
    column = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(column)
    if scale_mode == "none":
        return column.copy()
    if not finite_mask.any():
        return np.zeros_like(column, dtype=float)

    scaled = column.copy()
    finite_values = column[finite_mask]

    if scale_mode == "zscore":
        center = float(np.mean(finite_values))
        spread = float(np.std(finite_values))
        if spread == 0.0:
            scaled[:] = 0.0
        else:
            scaled = (scaled - center) / spread
    elif scale_mode == "minmax":
        lower = float(np.min(finite_values))
        upper = float(np.max(finite_values))
        span = upper - lower
        if span == 0.0:
            scaled[:] = 0.0
        else:
            scaled = (scaled - lower) / span
    else:  # pragma: no cover - exhaustive enum guard
        raise ValueError(f"Unsupported scale mode: {scale_mode!r}")

    scaled[~finite_mask] = np.nan
    return scaled


def _scale_values(
    values: np.ndarray, scale_mode: Literal["none", "zscore", "minmax"]
) -> np.ndarray:
    if scale_mode == "none":
        return np.asarray(values, dtype=float).copy()

    # Scale each signal independently so traces can be compared visually even
    # when they live on very different raw value ranges.
    scaled_columns = [
        _scale_column(values[:, index], scale_mode) for index in range(values.shape[1])
    ]
    return np.column_stack(scaled_columns)


def nearest_sample_index(time: Sequence[Any] | np.ndarray, target_time: Any) -> int:
    """Return the index of the sample nearest to an x-axis position."""

    time_values = _to_1d_array(time, name="time")
    try:
        numeric_time = np.asarray(time_values, dtype=float)
        target = float(target_time)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "time and target_time must be numeric for annotation."
        ) from exc

    if numeric_time.size == 0:
        raise ValueError("time cannot be empty.")
    return int(np.argmin(np.abs(numeric_time - target)))


def build_timecourse_marker(
    normalized: TimeSeriesPlotData,
    sample_index: int,
    *,
    label: str | None = None,
) -> TimeSeriesMarker:
    """Create a serializable marker for one sample in a normalized timecourse."""

    if sample_index < 0 or sample_index >= normalized.values.shape[0]:
        raise IndexError("sample_index is out of range for the timecourse.")

    time_seconds = float(np.asarray(normalized.time)[sample_index])
    values = tuple(float(value) for value in normalized.values[sample_index])
    return TimeSeriesMarker(
        sample_index=sample_index,
        time_seconds=time_seconds,
        values=values,
        label=label,
    )


def serialize_timecourse_annotations(
    normalized: TimeSeriesPlotData,
    markers: Sequence[TimeSeriesMarker],
    *,
    source_name: str | None = None,
) -> dict[str, Any]:
    """Serialize markers and their context to a JSON-friendly payload."""

    return {
        "source_name": source_name,
        "signal_labels": list(normalized.labels),
        "markers": [
            {
                "sample_index": marker.sample_index,
                "time_seconds": marker.time_seconds,
                "values": {
                    label: value
                    for label, value in zip(
                        normalized.labels, marker.values, strict=True
                    )
                },
                "label": marker.label,
            }
            for marker in markers
        ],
    }


def markers_from_serialized_annotations(
    payload: Mapping[str, Any],
) -> tuple[TimeSeriesMarker, ...]:
    """Rebuild markers from a serialized annotation payload."""

    signal_labels = tuple(str(label) for label in payload.get("signal_labels", ()))
    marker_records = payload.get("markers", ())
    markers: list[TimeSeriesMarker] = []
    for record in marker_records:
        values_by_label = record["values"]
        markers.append(
            TimeSeriesMarker(
                sample_index=int(record["sample_index"]),
                time_seconds=float(record["time_seconds"]),
                values=tuple(float(values_by_label[label]) for label in signal_labels),
                label=record.get("label"),
            )
        )
    return tuple(markers)


def _add_vertical_marker(
    figure: Any,
    *,
    marker: TimeSeriesMarker,
    line_color: str,
    line_dash: str,
    line_width: float,
) -> None:
    annotation_text = marker.label if marker.label else None
    figure.add_vline(
        x=marker.time_seconds,
        line_color=line_color,
        line_dash=line_dash,
        line_width=float(line_width),
        annotation_text=annotation_text,
    )


def normalize_timeseries(
    data: np.ndarray | pd.DataFrame,
    *,
    column_labels: Sequence[str] | None = None,
    time: Sequence[Any] | np.ndarray | None = None,
    scale_mode: Literal["none", "zscore", "minmax"] = "none",
) -> TimeSeriesPlotData:
    """Coerce a NumPy array or DataFrame into a validated plotting payload.

    Parameters
    ----------
    data:
        A 2D array with shape ``(time, signals)`` or a DataFrame with matching
        columns.
    column_labels:
        Required for array inputs. When provided for DataFrames, the labels are
        validated against the second dimension.
    time:
        Optional explicit x-axis values. If omitted, DataFrame indices are used
        and array inputs fall back to ``arange``.
    scale_mode:
        Per-signal scaling mode applied before plotting.

    Returns
    -------
    TimeSeriesPlotData
        The normalized time vector, numeric values, and column labels.
    """
    values = _coerce_values(data)
    labels = _coerce_labels(data, column_labels)
    x_values = _coerce_time(data, time, values.shape[0])
    scaled = _scale_values(values, scale_mode)
    return TimeSeriesPlotData(time=x_values, values=scaled, labels=labels)


def _plotly_imports() -> tuple[Any, Any]:
    try:
        import plotly.graph_objects as go  # pyright: ignore[reportMissingImports]
        from plotly.subplots import (
            make_subplots,  # pyright: ignore[reportMissingImports]
        )
    except ImportError as exc:
        raise ImportError(
            "plotly is required for plot_timeseries; install the project with the viz extra."
        ) from exc
    return go, make_subplots


def _trace_factory(
    go: Any, *, name: str, time: np.ndarray, values: np.ndarray, line_width: float
) -> Any:
    """Build the trace object used by both stacked and overlay modes."""
    return go.Scattergl(
        x=time,
        y=values,
        mode="lines",
        name=name,
        line={"width": float(line_width)},
    )


def plot_timeseries(
    data: np.ndarray | pd.DataFrame,
    *,
    column_labels: Sequence[str] | None = None,
    time: Sequence[Any] | np.ndarray | None = None,
    overlay: bool = False,
    scale_mode: Literal["none", "zscore", "minmax"] = "none",
    markers: Sequence[TimeSeriesMarker] | None = None,
    title: str | None = None,
    xaxis_title: str = "Time",
    yaxis_title: str | None = None,
    legend_title: str = "Signal",
    show_legend: bool = True,
    line_width: float = 1.5,
    template: str = "plotly_white",
    height: int | None = None,
    width: int | None = None,
    vertical_spacing: float = 0.03,
    uirevision: Any | None = None,
) -> Any:
    """Create an interactive Plotly time-series figure.

    The helper supports either vertically stacked subplots or a single overlay
    panel. Overlay mode intentionally uses ``Scattergl`` so large traces stay
    responsive.
    """
    normalized = normalize_timeseries(
        data,
        column_labels=column_labels,
        time=time,
        scale_mode=scale_mode,
    )
    go, make_subplots = _plotly_imports()
    marker_list = tuple(markers or ())

    axis_label = yaxis_title or ("Scaled value" if scale_mode != "none" else "Value")

    if overlay:
        figure = go.Figure()
        for label, column in zip(normalized.labels, normalized.values.T):
            # WebGL keeps the overlay mode usable when each trace is large.
            figure.add_trace(
                _trace_factory(
                    go,
                    name=label,
                    time=normalized.time,
                    values=column,
                    line_width=line_width,
                )
            )
        figure.update_layout(
            template=template,
            title=title,
            showlegend=show_legend,
            hovermode="x unified",
            dragmode="pan",
            legend_title_text=legend_title,
            xaxis_title=xaxis_title,
            yaxis_title=axis_label,
            height=height,
            width=width,
            uirevision=uirevision,
        )
        for marker in marker_list:
            _add_vertical_marker(
                figure,
                marker=marker,
                line_color="#b22222",
                line_dash="dash",
                line_width=1.5,
            )
        figure.update_xaxes(rangeslider_visible=True)
        return figure

    rows = normalized.values.shape[1]
    figure = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=vertical_spacing,
        row_titles=list(normalized.labels),
    )
    for row_index, (label, column) in enumerate(
        zip(normalized.labels, normalized.values.T), start=1
    ):
        figure.add_trace(
            _trace_factory(
                go,
                name=label,
                time=normalized.time,
                values=column,
                line_width=line_width,
            ),
            row=row_index,
            col=1,
        )

    # Keep the rangeslider on the bottom axis only so the stack remains tidy.
    figure.update_xaxes(rangeslider_visible=False)
    figure.update_xaxes(rangeslider_visible=True, row=rows, col=1)
    figure.update_layout(
        template=template,
        title=title,
        showlegend=show_legend,
        hovermode="x unified",
        dragmode="pan",
        legend_title_text=legend_title,
        xaxis_title=xaxis_title,
        yaxis_title=axis_label,
        height=height or max(250, 180 * rows),
        width=width,
        uirevision=uirevision,
    )
    for marker in marker_list:
        _add_vertical_marker(
            figure,
            marker=marker,
            line_color="#b22222",
            line_dash="dash",
            line_width=1.5,
        )
    return figure

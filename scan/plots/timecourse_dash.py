from __future__ import annotations

"""Dash-based manual annotation workflow for time-series plots."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from scan.plots.timecourse import (
    TimeSeriesMarker,
    build_timecourse_marker,
    markers_from_serialized_annotations,
    nearest_sample_index,
    normalize_timeseries,
    plot_timeseries,
    serialize_timecourse_annotations,
)


def _dash_imports() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    try:
        from dash import (  # pyright: ignore[reportMissingImports]
            Dash,
            Input,
            Output,
            State,
            ctx,
            dcc,
            html,
        )
        from dash.exceptions import (  # pyright: ignore[reportMissingImports]
            PreventUpdate,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "dash is required for the annotation app; install the project with the viz extra."
        ) from exc
    return Dash, Input, Output, State, ctx, dcc, html, PreventUpdate


def _marker_status(marker: TimeSeriesMarker) -> str:
    values = ", ".join(f"{value:.6g}" for value in marker.values)
    label_text = f" [{marker.label}]" if marker.label else ""
    return (
        f"Marked sample {marker.sample_index}{label_text} at {marker.time_seconds:.3f} s "
        f"with values [{values}]"
    )


def _removed_marker_status(count: int) -> str:
    if count == 0:
        return "No markers to remove."
    if count == 1:
        return "Removed the last marker."
    return "Cleared all markers."


def _render_timecourse_figure(
    data: np.ndarray | pd.DataFrame,
    *,
    column_labels: Sequence[str] | None,
    time: Sequence[Any] | np.ndarray | None,
    scale_mode: Literal["none", "zscore", "minmax"],
    markers: Sequence[TimeSeriesMarker],
    source_name: str | None,
    overlay: bool = False,
    xaxis_range: Sequence[Any] | None = None,
) -> Any:
    uirevision = source_name or "timecourse"
    figure = plot_timeseries(
        data,
        column_labels=column_labels,
        time=time,
        overlay=overlay,
        scale_mode=scale_mode,
        markers=markers,
        title=source_name,
        xaxis_title="Time (s)",
        uirevision=uirevision,
    )
    if xaxis_range is not None and len(xaxis_range) == 2:
        figure.update_xaxes(range=list(xaxis_range), autorange=False)
    return figure


def _serialize_annotation_state(
    normalized,
    markers: Sequence[TimeSeriesMarker],
    *,
    source_name: str | None,
) -> dict[str, Any]:
    return serialize_timecourse_annotations(
        normalized, markers, source_name=source_name
    )


def create_timecourse_annotation_app(
    data: np.ndarray | pd.DataFrame,
    *,
    column_labels: Sequence[str] | None = None,
    time: Sequence[Any] | np.ndarray | None = None,
    source_name: str | None = None,
    scale_mode: Literal["none", "zscore", "minmax"] = "none",
    overlay: bool = False,
    labels: Sequence[str] | None = None,
    default_output_path: str | Path | None = None,
) -> Any:
    """Build a Dash app for clicking markers onto a timecourse and saving JSON.

    When ``labels`` is provided, the UI adds a selector that stamps each new
    marker with the currently selected label. Undo still removes the most
    recent marker regardless of label.
    """

    Dash, Input, Output, State, ctx, dcc, html, PreventUpdate = _dash_imports()

    label_choices = tuple(str(label) for label in labels or ())
    default_label = label_choices[0] if label_choices else None

    normalized = normalize_timeseries(
        data,
        column_labels=column_labels,
        time=time,
        scale_mode=scale_mode,
    )
    initial_payload = serialize_timecourse_annotations(
        normalized, (), source_name=source_name
    )
    default_path = Path(default_output_path or f"{source_name or 'timecourse'}.json")

    app = Dash(__name__)
    app.title = source_name or "Timecourse annotation"

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H2("Timecourse annotation"),
                    html.P(
                        "Click the trace to drop a vertical marker, then save the session as JSON."
                    ),
                ]
            ),
            html.Div(
                [
                    html.Label("Marker label", htmlFor="label-selector"),
                    dcc.Dropdown(
                        id="label-selector",
                        options=[
                            {"label": label, "value": label} for label in label_choices
                        ],
                        value=default_label,
                        clearable=False,
                        disabled=not label_choices,
                        style={"width": "12rem"},
                    ),
                ],
                style={"margin": "0.5rem 0 1rem"},
            ),
            dcc.Graph(
                id="timecourse-graph",
                figure=_render_timecourse_figure(
                    data,
                    column_labels=column_labels,
                    time=time,
                    scale_mode=scale_mode,
                    markers=(),
                    source_name=source_name,
                    overlay=overlay,
                ),
                clear_on_unhover=True,
            ),
            dcc.Store(id="annotation-store", data=initial_payload),
            html.Div(
                [
                    dcc.Input(
                        id="output-path",
                        type="text",
                        value=str(default_path),
                        style={"width": "32rem"},
                    ),
                    html.Button(
                        "Undo last marker", id="undo-marker-button", n_clicks=0
                    ),
                    html.Button("Clear markers", id="clear-markers-button", n_clicks=0),
                    html.Button("Save JSON", id="save-button", n_clicks=0),
                ],
                style={"display": "flex", "gap": "0.75rem", "alignItems": "center"},
            ),
            html.Pre(id="click-status", style={"whiteSpace": "pre-wrap"}),
            html.Pre(id="save-status", style={"whiteSpace": "pre-wrap"}),
        ],
        style={"maxWidth": "1100px", "margin": "0 auto", "padding": "1rem"},
    )

    @app.callback(
        Output("timecourse-graph", "figure"),
        Output("annotation-store", "data"),
        Output("click-status", "children"),
        Input("timecourse-graph", "clickData"),
        Input("undo-marker-button", "n_clicks"),
        Input("clear-markers-button", "n_clicks"),
        State("label-selector", "value"),
        State("annotation-store", "data"),
        State("timecourse-graph", "relayoutData"),
        prevent_initial_call=True,
    )
    def update_markers(
        click_data: dict[str, Any] | None,
        undo_clicks: int,
        clear_clicks: int,
        selected_label: str | None,
        store_data: dict[str, Any],
        relayout_data: dict[str, Any] | None,
    ) -> tuple[Any, dict[str, Any], str]:
        markers = list(markers_from_serialized_annotations(store_data))
        triggered_id = ctx.triggered_id

        xaxis_range = None
        if relayout_data:
            if "xaxis.range[0]" in relayout_data and "xaxis.range[1]" in relayout_data:
                xaxis_range = [
                    relayout_data["xaxis.range[0]"],
                    relayout_data["xaxis.range[1]"],
                ]
            elif "xaxis.range" in relayout_data and isinstance(
                relayout_data["xaxis.range"], (list, tuple)
            ):
                xaxis_range = list(relayout_data["xaxis.range"])

        if triggered_id == "timecourse-graph":
            if not click_data:
                raise PreventUpdate
            point = click_data["points"][0]
            sample_index = nearest_sample_index(normalized.time, point["x"])
            marker = build_timecourse_marker(
                normalized, sample_index, label=selected_label
            )
            markers.append(marker)
            status = _marker_status(marker)
        elif triggered_id == "undo-marker-button":
            if not markers:
                raise PreventUpdate
            markers.pop()
            status = _removed_marker_status(1)
        elif triggered_id == "clear-markers-button":
            if not markers:
                raise PreventUpdate
            removed_count = len(markers)
            markers = []
            status = _removed_marker_status(removed_count)
        else:
            raise PreventUpdate

        payload = _serialize_annotation_state(
            normalized, markers, source_name=source_name
        )

        return (
            _render_timecourse_figure(
                data,
                column_labels=column_labels,
                time=time,
                scale_mode=scale_mode,
                markers=markers,
                source_name=source_name,
                xaxis_range=xaxis_range,
            ),
            payload,
            status,
        )

    @app.callback(
        Output("save-status", "children"),
        Input("save-button", "n_clicks"),
        State("annotation-store", "data"),
        State("output-path", "value"),
        prevent_initial_call=True,
    )
    def save_annotations(
        n_clicks: int, store_data: dict[str, Any], output_path: str
    ) -> str:
        if not store_data:
            raise PreventUpdate

        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(store_data, indent=2), encoding="utf-8")
        return f"Saved {len(store_data.get('markers', ()))} marker(s) to {path}"

    return app


def run_timecourse_annotation_app(
    data: np.ndarray | pd.DataFrame,
    *,
    overlay: bool = False,
    column_labels: Sequence[str] | None = None,
    time: Sequence[Any] | np.ndarray | None = None,
    source_name: str | None = None,
    scale_mode: Literal["none", "zscore", "minmax"] = "none",
    labels: Sequence[str] | None = None,
    default_output_path: str | Path | None = None,
    debug: bool = False,
) -> None:
    """Launch the Dash annotation app.

    Pass ``labels`` to enable multi-label annotation markers in the UI.
    """

    app = create_timecourse_annotation_app(
        data,
        column_labels=column_labels,
        time=time,
        overlay=overlay,
        source_name=source_name,
        scale_mode=scale_mode,
        labels=labels,
        default_output_path=default_output_path,
    )
    app.run(debug=debug)

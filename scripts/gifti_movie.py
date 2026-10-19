"""Render paired GIFTI metric time series to frames and optionally encode an MP4.

This is the movie-oriented sibling to scripts/gifti_snapshot.py and preserves the
same base surface visualization defaults for GIFTI metric inputs.

- Input: paired GIFTI metric files (.func.gii)
- Optional overlay: one or more paired GIFTI label files (.label.gii)
- Surface rendering: inflated fsLR32k GIFTI surfaces

Video encoding uses an ffmpeg executable. The script will try, in order:
1) --ffmpeg path (if provided)
2) system ffmpeg on PATH
3) imageio-ffmpeg (if installed; see optional dependency extra `viz`)

Examples
--------
# Basic movie
uv run python scripts/gifti_movie.py --input sub-01_bold_lh.func.gii --output bold.mp4

# Restrict frames and keep rendered PNGs
uv run python scripts/gifti_movie.py --input sub-01_bold_lh.func.gii --start 0 --stop 20 \
  --keep-frames --frames-dir tmp/gifti_frames

# Multiple views with explicit camera control
uv run python scripts/gifti_movie.py --input-left lh.func.gii --input-right rh.func.gii \
  --surf-views lateral medial --surf-elev 15 --surf-azim 210 --output bold_views.mp4
"""

from __future__ import annotations

import argparse
import dataclasses
import shutil
import subprocess
import tempfile
import time
import sys
from pathlib import Path
from typing import Any, Callable, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from nibabel.gifti.gifti import GiftiImage
from nibabel.loadsave import load as nib_load
import numpy as np
from nilearn import surface

ROOT = Path(__file__).resolve().parent.parent
try:
    from yawn.plots import SurfaceCameraConfig
    from yawn.plots import SurfaceRenderOptions
    from yawn.plots import SurfaceOverlay
    from yawn.plots import compute_intensity_bounds
    from yawn.plots import render_surface_figure
    from yawn.plots import validate_map_against_mesh
except ModuleNotFoundError:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from yawn.plots import SurfaceCameraConfig
    from yawn.plots import SurfaceRenderOptions
    from yawn.plots import SurfaceOverlay
    from yawn.plots import compute_intensity_bounds
    from yawn.plots import render_surface_figure
    from yawn.plots import validate_map_against_mesh


@dataclasses.dataclass(frozen=True)
class _MetricSource:
    n_frames: int
    output_hint: Path
    get_frame_maps: Callable[[int], tuple[np.ndarray, np.ndarray]]


@dataclasses.dataclass(frozen=True)
class _GiftiLabelPair:
    left_path: Path | None
    right_path: Path | None


def _validate_percentiles(p_low: float, p_high: float) -> tuple[float, float]:
    if not (0.0 <= p_low <= 100.0 and 0.0 <= p_high <= 100.0):
        raise ValueError(f"Percentiles must be in [0, 100], got {p_low}, {p_high}")
    if p_low >= p_high:
        raise ValueError(
            f"Lower percentile must be < upper percentile, got {p_low}, {p_high}"
        )
    return float(p_low), float(p_high)


def _sample_finite_values(
    data: np.ndarray, *, max_samples: int, rng: np.random.Generator
) -> np.ndarray:
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


def _load_gifti_image(path: Path, *, kind: str) -> GiftiImage:
    if not path.exists():
        raise FileNotFoundError(str(path))
    loaded = nib_load(str(path))
    if not isinstance(loaded, GiftiImage):
        raise TypeError(f"Expected {kind} GIFTI image, got {type(loaded)}")
    return loaded


def _get_gifti_frame(img: GiftiImage, index: int) -> np.ndarray:
    if index < 0 or index >= len(img.darrays):
        raise ValueError(f"frame index out of range: {index}")
    return np.asarray(img.darrays[index].data, dtype=float).ravel()


def _toggle_hemi_token(name: str) -> str | None:
    replacements = (
        ("lh.", "rh."),
        ("rh.", "lh."),
        ("_lh_", "_rh_"),
        ("_rh_", "_lh_"),
        ("_lh.", "_rh."),
        ("_rh.", "_lh."),
        ("-lh.", "-rh."),
        ("-rh.", "-lh."),
        (".lh_", ".rh_"),
        (".rh_", ".lh_"),
        (".L.", ".R."),
        (".R.", ".L."),
        ("_L_", "_R_"),
        ("_R_", "_L_"),
        ("_L.", "_R."),
        ("_R.", "_L."),
    )
    for old, new in replacements:
        if old in name:
            return name.replace(old, new, 1)
    return None


def _detect_hemi_token(name: str) -> str | None:
    left_tokens = ("lh.", "_lh_", "_lh.", "-lh.", ".lh_", ".L.", "_L_", "_L.")
    right_tokens = ("rh.", "_rh_", "_rh.", "-rh.", ".rh_", ".R.", "_R_", "_R.")
    if any(token in name for token in left_tokens):
        return "left"
    if any(token in name for token in right_tokens):
        return "right"
    return None


def _pair_from_single_path(path: Path, *, suffix: str) -> tuple[Path, Path]:
    if path.exists():
        mate_name = _toggle_hemi_token(path.name)
        if mate_name is None:
            raise ValueError(
                f"Could not infer hemisphere mate from {path}. Provide explicit left/right files."
            )
        mate = path.with_name(mate_name)
        if not mate.exists():
            raise FileNotFoundError(f"Could not find inferred hemisphere mate: {mate}")
        hemi = _detect_hemi_token(path.name)
        if hemi == "left":
            return path, mate
        if hemi == "right":
            return mate, path
        raise ValueError(
            f"Could not determine left/right hemisphere from {path}. Provide explicit left/right files."
        )

    candidate_sets = (
        (
            path.with_name(f"{path.name}_lh{suffix}"),
            path.with_name(f"{path.name}_rh{suffix}"),
        ),
        (
            path.with_name(f"{path.name}.lh{suffix}"),
            path.with_name(f"{path.name}.rh{suffix}"),
        ),
        (
            path.with_name(f"{path.name}-lh{suffix}"),
            path.with_name(f"{path.name}-rh{suffix}"),
        ),
    )
    for left_path, right_path in candidate_sets:
        if left_path.exists() and right_path.exists():
            return left_path, right_path
    raise FileNotFoundError(
        f"Could not resolve paired GIFTI files from {path}. Provide explicit left/right files."
    )


def _resolve_gifti_pair(
    *,
    single_path: Path | None,
    left_path: Path | None,
    right_path: Path | None,
    kind: str,
) -> tuple[Path, Path]:
    if left_path is not None or right_path is not None:
        if left_path is None or right_path is None:
            raise ValueError(
                f"Provide both --{kind}-left and --{kind}-right, or just --{kind}."
            )
        return left_path, right_path
    if single_path is None:
        raise ValueError(f"No {kind} input provided")
    suffix = ".func.gii" if kind == "input" else ".label.gii"
    return _pair_from_single_path(single_path, suffix=suffix)


def _resolve_gifti_pairs(
    *,
    single_paths: list[Path] | None,
    left_paths: list[Path] | None,
    right_paths: list[Path] | None,
    kind: str,
) -> list[_GiftiLabelPair]:
    single_paths = single_paths or []
    left_paths = left_paths or []
    right_paths = right_paths or []

    if single_paths and (left_paths or right_paths):
        raise ValueError(
            f"Do not mix --{kind} with --{kind}-left/--{kind}-right in the same command."
        )

    pairs: list[_GiftiLabelPair] = []
    if left_paths or right_paths:
        if len(left_paths) != len(right_paths):
            raise ValueError(
                f"Provide the same number of --{kind}-left and --{kind}-right arguments."
            )
        for left_path, right_path in zip(left_paths, right_paths):
            pairs.append(_GiftiLabelPair(left_path=left_path, right_path=right_path))
        return pairs

    for single_path in single_paths:
        left_path, right_path = _resolve_gifti_pair(
            single_path=single_path,
            left_path=None,
            right_path=None,
            kind=kind,
        )
        pairs.append(_GiftiLabelPair(left_path=left_path, right_path=right_path))
    return pairs


def _load_gifti_metric_source(args: argparse.Namespace) -> _MetricSource:
    left_path, right_path = _resolve_gifti_pair(
        single_path=cast(Path | None, args.input),
        left_path=cast(Path | None, args.input_left),
        right_path=cast(Path | None, args.input_right),
        kind="input",
    )
    left_img = _load_gifti_image(left_path, kind="metric")
    right_img = _load_gifti_image(right_path, kind="metric")

    n_frames = len(left_img.darrays)
    if n_frames <= 0:
        raise ValueError("No frames found in left hemisphere GIFTI")
    if len(right_img.darrays) != n_frames:
        raise ValueError(
            "Left and right hemisphere GIFTI files must have the same number of frames"
        )

    left_sizes = {int(np.asarray(darr.data).size) for darr in left_img.darrays}
    right_sizes = {int(np.asarray(darr.data).size) for darr in right_img.darrays}
    if len(left_sizes) != 1 or len(right_sizes) != 1:
        raise ValueError(
            "Each GIFTI hemisphere must have consistent vertex counts across frames"
        )

    def _get_frame_maps(frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        return _get_gifti_frame(left_img, frame_index), _get_gifti_frame(
            right_img, frame_index
        )

    return _MetricSource(
        n_frames=n_frames,
        output_hint=left_path,
        get_frame_maps=_get_frame_maps,
    )


def _load_gifti_label_overlay(args: argparse.Namespace) -> SurfaceOverlay | None:
    label_single = cast(list[Path] | None, args.label)
    label_left = cast(list[Path] | None, args.label_left)
    label_right = cast(list[Path] | None, args.label_right)
    if not label_single and not label_left and not label_right:
        return None

    label_pairs = _resolve_gifti_pairs(
        single_paths=label_single,
        left_paths=label_left,
        right_paths=label_right,
        kind="label",
    )
    if not label_pairs:
        return None

    # (name, left binary map or None, right binary map or None) per ROI
    entries: list[tuple[str, np.ndarray | None, np.ndarray | None]] = []

    for pair in label_pairs:
        left_img = (
            _load_gifti_image(pair.left_path, kind="label")
            if pair.left_path is not None
            else None
        )
        right_img = (
            _load_gifti_image(pair.right_path, kind="label")
            if pair.right_path is not None
            else None
        )
        if left_img is None and right_img is None:
            continue
        if left_img is not None and len(left_img.darrays) < 1:
            raise ValueError("Label GIFTI files must contain at least one data array")
        if right_img is not None and len(right_img.darrays) < 1:
            raise ValueError("Label GIFTI files must contain at least one data array")

        if left_img is not None:
            left_data = _get_gifti_frame(left_img, 0)
        else:
            assert right_img is not None
            left_data = np.full(
                int(np.asarray(right_img.darrays[0].data).size), np.nan, dtype=float
            )
        if right_img is not None:
            right_data = _get_gifti_frame(right_img, 0)
        else:
            assert left_img is not None
            right_data = np.full(
                int(np.asarray(left_img.darrays[0].data).size), np.nan, dtype=float
            )

        if bool(args.label_ignore_zero):
            left_data = left_data.copy()
            right_data = right_data.copy()
            left_data[left_data == 0] = np.nan
            right_data[right_data == 0] = np.nan

        pair_name = (pair.left_path or pair.right_path or Path("label")).name
        unique_values = np.unique(np.concatenate((left_data, right_data), axis=0))
        for label_value in unique_values:
            if not np.isfinite(label_value):
                continue
            left_mask = left_data == label_value
            right_mask = right_data == label_value
            if not np.any(left_mask) and not np.any(right_mask):
                continue
            entries.append(
                (
                    f"{pair_name}:{int(label_value)}",
                    np.where(left_mask, 1.0, np.nan) if np.any(left_mask) else None,
                    np.where(right_mask, 1.0, np.nan) if np.any(right_mask) else None,
                )
            )

    if not entries:
        return None

    colors = plt.get_cmap(str(args.label_cmap))(
        np.linspace(0.0, 1.0, num=len(entries), endpoint=False)
    )
    left_plot: list[np.ndarray] = []
    right_plot: list[np.ndarray] = []
    labels_left: list[tuple[int, str, tuple[float, float, float, float]]] = []
    labels_right: list[tuple[int, str, tuple[float, float, float, float]]] = []
    for idx, (name, left_map, right_map) in enumerate(entries):
        rgba = cast(
            tuple[float, float, float, float], tuple(float(c) for c in colors[idx])
        )
        if left_map is not None:
            left_plot.append(left_map)
            labels_left.append((1, name, rgba))
        if right_map is not None:
            right_plot.append(right_map)
            labels_right.append((1, name, rgba))

    return SurfaceOverlay(
        left_plot=left_plot,
        right_plot=right_plot,
        output_tag="roi",
        labels_left=labels_left,
        labels_right=labels_right,
    )


def _render_surface_image(
    *,
    stat_map_left: np.ndarray,
    stat_map_right: np.ndarray,
    surf_left_mesh: Any,
    surf_right_mesh: Any,
    surf_left_sulc: np.ndarray,
    surf_right_sulc: np.ndarray,
    overlay: SurfaceOverlay | None,
    options: SurfaceRenderOptions,
    index: int,
    figure_title: str | None = None,
    colorbar_label: str | None = None,
) -> np.ndarray:
    rendered = render_surface_figure(
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
    text_color = "white" if options.black_bg else "black"
    if figure_title:
        rendered.figure.suptitle(figure_title, color=text_color)
    if colorbar_label and rendered.colorbar_ax is not None:
        rendered.colorbar_ax.set_ylabel(colorbar_label, color=text_color)
    canvas = FigureCanvasAgg(rendered.figure)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba()).copy()
    plt.close(rendered.figure)
    return rgba


def _strip_suffixes(name: str, suffixes: tuple[str, ...]) -> str:
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _derive_default_output(path: Path) -> Path:
    stem = _strip_suffixes(
        path.name,
        (
            ".func.gii",
            ".shape.gii",
            ".nii.gz",
            ".nii",
            ".gii",
        ),
    )
    return path.with_name(f"{stem}.mp4")


def _iter_frame_indices(
    n_frames: int, start: int, stop: int | None, step: int
) -> list[int]:
    stop_ = n_frames if stop is None else min(stop, n_frames)
    if start < 0 or start >= n_frames:
        raise ValueError(f"start must be in [0, {n_frames - 1}], got {start}")
    if stop_ <= start:
        raise ValueError(f"stop must be > start; got start={start} stop={stop_}")
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    return list(range(start, stop_, step))


def _find_ffmpeg(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path

    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _run_ffmpeg(
    *,
    ffmpeg: str,
    frames_pattern: str,
    fps: float,
    output: Path,
    crf: int,
    preset: str,
    scale: str | None,
) -> None:
    cmd: list[str] = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        frames_pattern,
    ]

    vf_filters: list[str] = []
    if scale:
        vf_filters.append(f"scale={scale}")
    vf_filters.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")
    if vf_filters:
        cmd += ["-vf", ",".join(vf_filters)]

    cmd += [
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        preset,
        str(output),
    ]
    subprocess.run(cmd, check=True)


def _compute_frame_bounds(
    source: _MetricSource,
    *,
    frame_index: int,
    p_low: float,
    p_high: float,
    max_samples: int,
    rng: np.random.Generator,
    vmin_arg: float | None,
    vmax_arg: float | None,
) -> tuple[float | None, float | None]:
    if vmin_arg is not None and vmax_arg is not None:
        return float(vmin_arg), float(vmax_arg)

    left_map, right_map = source.get_frame_maps(frame_index)
    samples = _sample_finite_values(
        np.concatenate((left_map, right_map), axis=0),
        max_samples=max_samples,
        rng=rng,
    )
    vmin = float(np.percentile(samples, p_low)) if samples.size > 0 else None
    vmax = float(np.percentile(samples, p_high)) if samples.size > 0 else None
    if vmin_arg is not None:
        vmin = float(vmin_arg)
    if vmax_arg is not None:
        vmax = float(vmax_arg)
    return vmin, vmax


def _compute_global_bounds(
    source: _MetricSource,
    *,
    frame_indices: list[int],
    p_low: float,
    p_high: float,
    max_total_samples: int,
    vmin_arg: float | None,
    vmax_arg: float | None,
) -> tuple[float | None, float | None]:
    if vmin_arg is not None and vmax_arg is not None:
        return float(vmin_arg), float(vmax_arg)

    rng = np.random.default_rng(0)
    samples_per_frame = max(1, int(max_total_samples // max(len(frame_indices), 1)))
    collected: list[np.ndarray] = []
    for frame_index in frame_indices:
        left_map, right_map = source.get_frame_maps(frame_index)
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
    vmin = float(np.percentile(all_samples, p_low)) if all_samples.size > 0 else None
    vmax = float(np.percentile(all_samples, p_high)) if all_samples.size > 0 else None
    if vmin_arg is not None:
        vmin = float(vmin_arg)
    if vmax_arg is not None:
        vmax = float(vmax_arg)
    return vmin, vmax


def _default_surface_path(filename: str) -> Path:
    return Path(__file__).resolve().parent.parent / "template" / filename


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render paired GIFTI metric data to surface frames and optionally encode a movie.",
    )
    p.add_argument(
        "--input",
        required=False,
        type=Path,
        default=None,
        help="A single GIFTI hemisphere file or prefix used to infer the mate file.",
    )
    p.add_argument(
        "--input-left",
        type=Path,
        default=None,
        help="Left hemisphere metric GIFTI (.func.gii).",
    )
    p.add_argument(
        "--input-right",
        type=Path,
        default=None,
        help="Right hemisphere metric GIFTI (.func.gii).",
    )
    p.add_argument(
        "--label",
        type=Path,
        action="append",
        default=None,
        help="A GIFTI label file or prefix used to infer the mate file. Repeat to overlay multiple label pairs.",
    )
    p.add_argument(
        "--label-left",
        type=Path,
        action="append",
        default=None,
        help="Left hemisphere label GIFTI (.label.gii). Repeat alongside --label-right to overlay multiple label pairs.",
    )
    p.add_argument(
        "--label-right",
        type=Path,
        action="append",
        default=None,
        help="Right hemisphere label GIFTI (.label.gii). Repeat alongside --label-left to overlay multiple label pairs.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output movie path (.mp4). Defaults to a path derived from the metric input.",
    )
    p.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Directory to write frames. Defaults to a temp directory.",
    )
    p.add_argument(
        "--keep-frames",
        action="store_true",
        help="Do not delete frames directory when done (only applies to temp dir).",
    )
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Only render frames; do not run ffmpeg.",
    )

    p.add_argument(
        "--surf-left",
        type=Path,
        default=_default_surface_path("fsaverage.L.inflated.32k_fs_LR.surf.gii"),
        help="Left hemisphere inflated surface GIFTI.",
    )
    p.add_argument(
        "--surf-right",
        type=Path,
        default=_default_surface_path("fsaverage.R.inflated.32k_fs_LR.surf.gii"),
        help="Right hemisphere inflated surface GIFTI.",
    )
    p.add_argument(
        "--sulc-left",
        type=Path,
        default=_default_surface_path("fsaverage.L.sulc.32k_fs_LR.surf.gii"),
        help="Left hemisphere sulcal depth GIFTI.",
    )
    p.add_argument(
        "--sulc-right",
        type=Path,
        default=_default_surface_path("fsaverage.R.sulc.32k_fs_LR.surf.gii"),
        help="Right hemisphere sulcal depth GIFTI.",
    )
    p.add_argument(
        "--surf-views",
        nargs="+",
        default=["lateral", "medial"],
        help="One or more nilearn surface views. Each view renders both hemispheres as separate panels.",
    )
    p.add_argument(
        "--surf-elev",
        type=float,
        default=None,
        help=(
            "Explicit camera elevation passed through nilearn as part of a "
            "(elev, azim) view tuple. Overrides --surf-views when paired with --surf-azim."
        ),
    )
    p.add_argument(
        "--surf-azim",
        type=float,
        default=None,
        help=(
            "Explicit camera azimuth passed through nilearn as part of a "
            "(elev, azim) view tuple. Overrides --surf-views when paired with --surf-azim."
        ),
    )
    p.add_argument(
        "--surf-elev-left",
        type=float,
        default=None,
        help="Per-hemisphere override for camera elevation (LEFT).",
    )
    p.add_argument(
        "--surf-elev-right",
        type=float,
        default=None,
        help="Per-hemisphere override for camera elevation (RIGHT).",
    )
    p.add_argument(
        "--surf-azim-left",
        type=float,
        default=None,
        help="Per-hemisphere override for camera azimuth (LEFT).",
    )
    p.add_argument(
        "--surf-azim-right",
        type=float,
        default=None,
        help="Per-hemisphere override for camera azimuth (RIGHT).",
    )
    p.add_argument(
        "--surf-zoom",
        type=float,
        default=1.8,
        help="Zoom factor for mplot3d surface panels.",
    )
    p.add_argument(
        "--ncols",
        type=int,
        default=None,
        help="Number of columns when rendering multiple panels.",
    )

    p.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap name.")
    p.add_argument(
        "--label-cmap",
        default="tab20",
        help="Matplotlib colormap name for label contour overlays.",
    )
    p.add_argument(
        "--label-ignore-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat label 0 as background (default: true).",
    )
    p.add_argument("--vmin", type=float, default=None, help="Lower bound for colormap.")
    p.add_argument("--vmax", type=float, default=None, help="Upper bound for colormap.")
    p.add_argument(
        "--intensity-mode",
        choices=["global", "frame"],
        default="global",
        help="How to set vmin/vmax when not explicitly provided.",
    )
    p.add_argument(
        "--auto-percentiles",
        nargs=2,
        type=float,
        metavar=("PLOW", "PHIGH"),
        default=(1.0, 99.0),
        help="Percentiles for automatic vmin/vmax (default: 1 99).",
    )
    p.add_argument(
        "--auto-max-samples",
        type=int,
        default=200_000,
        help="Max random samples when estimating frame percentiles (default: 200000).",
    )
    p.add_argument(
        "--auto-max-total-samples",
        type=int,
        default=2_000_000,
        help="Max total random samples used to estimate global percentiles (default: 2000000).",
    )
    p.add_argument(
        "--black-bg",
        action="store_true",
        help="Use black figure background.",
    )
    p.add_argument(
        "--colorbar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show colorbar (default: true).",
    )
    p.add_argument(
        "--colorbar-side",
        choices=["right", "left"],
        default="right",
        help="Where to place the colorbar when enabled (default: right).",
    )
    p.add_argument(
        "--title",
        default=None,
        help="Optional per-panel title template. Use {index}, {panel}, {view}, {hemi}, {time}, {format} placeholders.",
    )
    p.add_argument(
        "--figure-title",
        default=None,
        help="Optional single title drawn above all panels.",
    )
    p.add_argument(
        "--colorbar-label",
        default=None,
        help="Optional label drawn next to the colorbar.",
    )
    p.add_argument(
        "--time-annotate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Annotate movie with time in seconds (default: false).",
    )
    p.add_argument("--tr", type=float, default=None, help="Repetition time in seconds.")
    p.add_argument(
        "--t0-trs",
        type=float,
        default=0.0,
        help="Starting time in TRs for index 0.",
    )

    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved frames (default: 150).",
    )
    p.add_argument(
        "--size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=(1280, 720),
        help="Frame size in pixels (default: 1280 720).",
    )

    p.add_argument(
        "--start", type=int, default=0, help="First frame index (default: 0)."
    )
    p.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Stop frame index (exclusive). Defaults to end.",
    )
    p.add_argument("--step", type=int, default=1, help="Frame step (default: 1).")

    p.add_argument(
        "--fps", type=float, default=10.0, help="Frames per second (default: 10)."
    )
    p.add_argument(
        "--ffmpeg",
        type=str,
        default=None,
        help="Path to ffmpeg executable (optional).",
    )
    p.add_argument(
        "--crf",
        type=int,
        default=23,
        help="x264 CRF quality (lower=better, larger files). Default: 23.",
    )
    p.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=[
            "ultrafast",
            "superfast",
            "veryfast",
            "faster",
            "fast",
            "medium",
            "slow",
            "slower",
            "veryslow",
        ],
        help="x264 preset (default: medium).",
    )
    p.add_argument(
        "--scale",
        type=str,
        default=None,
        help="Optional ffmpeg scale W:H (e.g. 960:-2) to downscale during encoding.",
    )
    p.add_argument(
        "--fast-render",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reuse surface artists across frames when vmin/vmax stay fixed. "
            "Automatically falls back when per-frame auto scaling is active."
        ),
    )
    p.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print periodic progress updates (default: true).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if bool(args.time_annotate) and args.tr is None:
        raise ValueError("--tr is required when using --time-annotate")

    source = _load_gifti_metric_source(args)
    overlay = _load_gifti_label_overlay(args)
    output_path = (
        Path(args.output)
        if args.output is not None
        else _derive_default_output(source.output_hint)
    )

    surf_left_mesh = surface.load_surf_mesh(str(args.surf_left))
    surf_right_mesh = surface.load_surf_mesh(str(args.surf_right))
    surf_left_sulc = np.asarray(
        surface.load_surf_data(str(args.sulc_left)), dtype=float
    )
    surf_right_sulc = np.asarray(
        surface.load_surf_data(str(args.sulc_right)), dtype=float
    )
    probe_left, probe_right = source.get_frame_maps(0)
    validate_map_against_mesh(
        probe_left,
        surf_left_mesh,
        kind="Left metric",
        source_type="metric",
        source_format="gifti",
    )
    validate_map_against_mesh(
        probe_right,
        surf_right_mesh,
        kind="Right metric",
        source_type="metric",
        source_format="gifti",
    )
    if overlay is not None:
        validate_map_against_mesh(
            overlay.left_plot,
            surf_left_mesh,
            kind="Left label",
            source_type="overlay",
            source_format="gifti",
        )
        validate_map_against_mesh(
            overlay.right_plot,
            surf_right_mesh,
            kind="Right label",
            source_type="overlay",
            source_format="gifti",
        )

    frame_indices = _iter_frame_indices(
        source.n_frames,
        int(args.start),
        cast(int | None, args.stop),
        int(args.step),
    )
    p_low, p_high = _validate_percentiles(
        float(args.auto_percentiles[0]), float(args.auto_percentiles[1])
    )
    global_vmin: float | None = None
    global_vmax: float | None = None
    if args.intensity_mode == "global":
        global_vmin, global_vmax = _compute_global_bounds(
            source,
            frame_indices=frame_indices,
            p_low=p_low,
            p_high=p_high,
            max_total_samples=int(args.auto_max_total_samples),
            vmin_arg=cast(float | None, args.vmin),
            vmax_arg=cast(float | None, args.vmax),
        )

    width_px, height_px = int(args.size[0]), int(args.size[1])
    dpi = int(args.dpi)
    figsize = (width_px / dpi, height_px / dpi)
    panels = [(str(view), hemi) for view in args.surf_views for hemi in ("L", "R")]
    if not panels:
        raise RuntimeError("No surface panels to render")
    ncols = int(args.ncols) if args.ncols is not None else len(panels)
    if ncols <= 0:
        raise ValueError(f"--ncols must be positive, got {ncols}")

    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    if args.frames_dir is None:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="gifti_frames_")
        frames_dir = Path(temp_dir_obj.name)
    else:
        frames_dir = Path(args.frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
    keep_temp_frames = bool(args.keep_frames) or bool(args.no_video)

    t_all0 = time.perf_counter()

    def _camera(hemi_name: str) -> SurfaceCameraConfig:
        # Per-hemisphere values override the global --surf-elev/--surf-azim.
        elev = getattr(args, f"surf_elev_{hemi_name}")
        azim = getattr(args, f"surf_azim_{hemi_name}")
        return SurfaceCameraConfig(
            elev=elev if elev is not None else args.surf_elev,
            azim=azim if azim is not None else args.surf_azim,
        )

    left_camera = _camera("left")
    right_camera = _camera("right")

    for frame_number, frame_index in enumerate(frame_indices):
        t0 = time.perf_counter()
        left_map, right_map = source.get_frame_maps(frame_index)
        validate_map_against_mesh(
            left_map,
            surf_left_mesh,
            kind="Left metric",
            source_type="metric",
            source_format="gifti",
        )
        validate_map_against_mesh(
            right_map,
            surf_right_mesh,
            kind="Right metric",
            source_type="metric",
            source_format="gifti",
        )

        frame_vmin = global_vmin
        frame_vmax = global_vmax
        if args.intensity_mode == "frame":
            frame_vmin, frame_vmax = compute_intensity_bounds(
                cast(Any, source),
                selected_index=frame_index,
                intensity_mode="frame",
                p_low=p_low,
                p_high=p_high,
                max_samples=int(args.auto_max_samples),
                max_total_samples=int(args.auto_max_total_samples),
                vmin_arg=cast(float | None, args.vmin),
                vmax_arg=cast(float | None, args.vmax),
            )
        if frame_vmin is None or frame_vmax is None:
            frame_vmin, frame_vmax = -1.0, 1.0

        time_sec = (
            (float(args.t0_trs) + float(frame_index)) * float(args.tr)
            if args.tr is not None
            else None
        )
        options = SurfaceRenderOptions(
            views=list(args.surf_views),
            figure_size=figsize,
            dpi=dpi,
            ncols=(int(args.ncols) if args.ncols is not None else None),
            cmap=str(args.cmap),
            vmin=float(frame_vmin),
            vmax=float(frame_vmax),
            mesh_alpha=1.0,
            surf_zoom=float(args.surf_zoom),
            black_bg=bool(args.black_bg),
            colorbar=bool(args.colorbar),
            colorbar_side=str(args.colorbar_side),
            title_template=str(args.title) if args.title else None,
            source_format="gifti",
            time_annotate=bool(args.time_annotate),
            time_s=time_sec,
            threshold=None,
            left_camera=left_camera,
            right_camera=right_camera,
        )
        frame_image = _render_surface_image(
            stat_map_left=left_map,
            stat_map_right=right_map,
            surf_left_mesh=surf_left_mesh,
            surf_right_mesh=surf_right_mesh,
            surf_left_sulc=surf_left_sulc,
            surf_right_sulc=surf_right_sulc,
            overlay=overlay,
            options=options,
            index=frame_index,
            figure_title=cast(str | None, args.figure_title),
            colorbar_label=cast(str | None, args.colorbar_label),
        )
        frame_path = frames_dir / f"frame_{frame_number:05d}.png"
        plt.imsave(str(frame_path), frame_image)
        if bool(args.progress):
            n_done = frame_number + 1
            n_total = len(frame_indices)
            if n_done == 1 or (n_done % 5) == 0 or n_done == n_total:
                elapsed = time.perf_counter() - t0
                total_elapsed = time.perf_counter() - t_all0
                avg = total_elapsed / max(n_done, 1)
                remaining = avg * (n_total - n_done)
                print(
                    f"Rendered frame {n_done}/{n_total} (index={frame_index}) in {elapsed:.2f}s; avg {avg:.2f}s/frame; ETA {remaining / 60:.1f} min"
                )

    try:
        if args.no_video:
            print(f"Rendered {len(frame_indices)} frame(s) to: {frames_dir}")
            return 0

        ffmpeg = _find_ffmpeg(cast(str | None, args.ffmpeg))
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg not found. Install system ffmpeg, or add optional extra 'viz' "
                "(imageio-ffmpeg) and re-run. You can also use --no-video to only render frames."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        _run_ffmpeg(
            ffmpeg=ffmpeg,
            frames_pattern=str(frames_dir / "frame_%05d.png"),
            fps=float(args.fps),
            output=output_path,
            crf=int(args.crf),
            preset=str(args.preset),
            scale=cast(str | None, args.scale),
        )
        print(f"Wrote movie: {output_path}")
        return 0
    finally:
        if temp_dir_obj is not None and (not keep_temp_frames):
            temp_dir_obj.cleanup()
        elif temp_dir_obj is not None and keep_temp_frames:
            print(f"Kept frames at: {frames_dir}")


if __name__ == "__main__":
    raise SystemExit(main())

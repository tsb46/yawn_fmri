from __future__ import annotations

import dataclasses
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, cast, Literal

import nibabel as nib
from nibabel.gifti.gifti import GiftiImage
from nibabel.loadsave import load as nib_load
from matplotlib.colors import ListedColormap
from matplotlib.colors import is_color_like, to_rgba
from nilearn import surface
import numpy as np

from yawn.plots.surface import RenderedSurfaceFigure
from yawn.plots.surface import SurfaceCameraConfig
from yawn.plots.surface import SurfaceOverlay
from yawn.plots.surface import SurfaceRenderOptions
from yawn.plots.surface import compute_intensity_bounds
from yawn.plots.surface import render_surface_figure
from yawn.plots.surface import render_surface_into_axes
from yawn.plots.surface import validate_map_against_mesh


@dataclasses.dataclass(frozen=True)
class _CiftiFrameSpec:
    frame_axis_first: bool
    n_frames: int


@dataclasses.dataclass(frozen=True)
class _MetricSource:
    n_frames: int
    output_hint: Path
    get_frame_maps: Callable[[int], tuple[np.ndarray, np.ndarray]]


@dataclasses.dataclass(frozen=True)
class _OverlayData:
    left_plot: list[np.ndarray]
    right_plot: list[np.ndarray]
    output_hint: Path
    output_tag: str
    # for Gifti ROIs
    labels_left: list[tuple[int, str, tuple[float, float, float, float]]] | None = None
    labels_right: list[tuple[int, str, tuple[float, float, float, float]]] | None = None
    # for CIFTI atlases
    labels: list[tuple[int, str, tuple[float, float, float, float]]] | None = None
    # for CIFTI atlases
    cmap: Any | None = None


@dataclasses.dataclass(frozen=True)
class _SceneData:
    metric_source: _MetricSource | None
    overlay: _OverlayData | None
    source_format: Literal["gifti", "cifti"]


@dataclasses.dataclass(frozen=True)
class _PairData:
    left: np.ndarray
    right: np.ndarray


def _default_surface_path(filename: str) -> Path:
    """Resolve a bundled template surface path."""
    return Path(__file__).resolve().parent.parent.parent / "template" / filename


@dataclasses.dataclass(frozen=True)
class _SnapshotParams:
    input_left: Path | None = None
    input_right: Path | None = None
    roi_left: list[Path] | None = None
    roi_right: list[Path] | None = None
    roi_left_color: list[str] | None = None
    roi_right_color: list[str] | None = None
    roi_left_label: list[str] | None = None
    roi_right_label: list[str] | None = None
    atlas: str | None = None
    atlas_label_index: list[str] | None = None
    format: str = "auto"
    index: int = 0
    output: Path | None = None
    surf_left: Path = dataclasses.field(
        default_factory=lambda: _default_surface_path(
            "fsaverage.L.inflated.32k_fs_LR.surf.gii"
        )
    )
    surf_right: Path = dataclasses.field(
        default_factory=lambda: _default_surface_path(
            "fsaverage.R.inflated.32k_fs_LR.surf.gii"
        )
    )
    sulc_left: Path = dataclasses.field(
        default_factory=lambda: _default_surface_path(
            "fsaverage.L.sulc.32k_fs_LR.surf.gii"
        )
    )
    sulc_right: Path = dataclasses.field(
        default_factory=lambda: _default_surface_path(
            "fsaverage.R.sulc.32k_fs_LR.surf.gii"
        )
    )
    surf_views: tuple[str, ...] = ("lateral", "medial")
    surf_elev: float | None = None
    surf_azim: float | None = None
    surf_elev_left: float | None = None
    surf_elev_right: float | None = None
    surf_azim_left: float | None = None
    surf_azim_right: float | None = None
    surf_zoom: float = 1.8
    ncols: int | None = None
    cmap: str = "coolwarm"
    roi_cmap: str = "tab20"
    vmin: float | None = None
    vmax: float | None = None
    intensity_mode: str = "global"
    auto_percentiles: tuple[float, float] = (1.0, 99.0)
    auto_max_samples: int = 200_000
    auto_max_total_samples: int = 2_000_000
    black_bg: bool = False
    colorbar: bool = True
    colorbar_side: str = "right"
    title: str | None = None
    time_annotate: bool = False
    tr: float | None = None
    t0_trs: float = 0.0
    dpi: int = 150
    size: tuple[int, int] = (1280, 720)
    atlas_cmap: str = "tab20"
    atlas_ignore_zero: bool = True
    atlas_legend: bool = False
    atlas_legend_max_items: int = 40
    roi_legend: bool = False
    exclude_medial_wall: bool = False
    threshold: float | None = None
    sulc_file_reverse_sign: bool = False
    mesh_alpha: float = 1.0


def _validate_percentiles(p_low: float, p_high: float) -> tuple[float, float]:
    """Validate percentile bounds for automatic intensity scaling."""
    if not (0.0 <= p_low <= 100.0 and 0.0 <= p_high <= 100.0):
        raise ValueError(f"Percentiles must be in [0, 100], got {p_low}, {p_high}")
    if p_low >= p_high:
        raise ValueError(
            f"Lower percentile must be < upper percentile, got {p_low}, {p_high}"
        )
    return float(p_low), float(p_high)


def _normalize_rgba(
    rgba4: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Normalize RGBA values to the 0..1 range when needed."""
    mx = max(rgba4)
    if mx > 1.0 and mx <= 255.0:
        return tuple(float(x) / 255.0 for x in rgba4)  # type: ignore[return-value]
    return rgba4


def _normalize_label_filter_values(values: list[str] | None) -> set[str] | None:
    """Normalize requested label names for atlas filtering."""
    if values is None:
        return None
    normalized = {str(value).strip() for value in values if str(value).strip()}
    return normalized


def _cifti_family_name(parcel_name: str) -> str:
    """Extract the family name from a CIFTI parcel label."""
    parts = str(parcel_name).split("_")
    if len(parts) >= 3 and parts[0] == "7Networks":
        return str(parts[2])
    if len(parts) >= 2:
        return str(parts[1])
    return str(parcel_name)


def _gordon_cifti_roi_network_name(roi_name: str) -> str:
    """Convert a Gordon ROI label into its network name.

    Gordon CIFTI labels in this repository are encoded as hemisphere-prefixed
    ROI identifiers, for example ``L_Default_2`` or ``R_DorsAttn_Post_12``.
    The leading hemisphere token and trailing ROI index are not part of the
    display name. This helper removes those pieces and returns the network name.
    """
    parts = [part for part in str(roi_name).split("_") if part]
    if len(parts) < 2:
        return str(roi_name)

    hemisphere_tokens = {"l", "r", "lh", "rh", "left", "right"}
    if parts[0].lower() in hemisphere_tokens and len(parts) >= 3:
        parts = parts[1:]

    if parts and parts[-1].isdigit():
        parts = parts[:-1]

    if not parts:
        return str(roi_name)

    return "_".join(parts)


def _yeo_cifti_network_name(parcel_name: str) -> str:
    """Map a Yeo parcel label to its displayed network name."""
    return _cifti_family_name(parcel_name)


def _is_medial_wall_label_name(name: str) -> bool:
    """Return whether a label name refers to the medial wall."""
    normalized = str(name).strip().lower().replace("_", "").replace("-", "")
    return normalized in {"???", "none", "medialwall"}


def _sample_overlay_colors(
    cmap_name: str, roi_pairs: list[tuple[Path | None, Path | None]]
) -> list[tuple[float, float, float, float]]:
    """Sample distinct overlay colors from a named colormap."""
    # get number of non-null ROIs across all pairs
    count = sum(1 for pair in roi_pairs for roi in pair if roi is not None)
    if count <= 0:
        return []
    import matplotlib.pyplot as plt

    cmap_obj = plt.get_cmap(cmap_name)

    def _rgba(value: float) -> tuple[float, float, float, float]:
        color = cmap_obj(value)
        return (
            float(color[0]),
            float(color[1]),
            float(color[2]),
            float(color[3]),
        )

    if count == 1:
        return [_rgba(0.5)]
    samples = np.linspace(0.0, 1.0, num=count, endpoint=False)
    return [_rgba(float(sample)) for sample in samples]


def _infer_cifti_axes(
    img: nib.cifti2.cifti2.Cifti2Image,
) -> tuple[Any, Any, _CiftiFrameSpec]:
    """Infer the frame and brain-model axes from a CIFTI image."""
    from nibabel.cifti2 import cifti2_axes

    ax0 = cifti2_axes.from_index_mapping(img.header.get_index_map(0))
    ax1 = cifti2_axes.from_index_mapping(img.header.get_index_map(1))
    if isinstance(ax0, cifti2_axes.BrainModelAxis) and not isinstance(
        ax1, cifti2_axes.BrainModelAxis
    ):
        frame_axis, brain_axis = ax1, ax0
        frame_spec = _CiftiFrameSpec(
            frame_axis_first=False, n_frames=int(frame_axis.size)
        )
    elif isinstance(ax1, cifti2_axes.BrainModelAxis) and not isinstance(
        ax0, cifti2_axes.BrainModelAxis
    ):
        frame_axis, brain_axis = ax0, ax1
        frame_spec = _CiftiFrameSpec(
            frame_axis_first=True, n_frames=int(frame_axis.size)
        )
    else:
        raise ValueError(
            "Expected one BrainModelAxis and one frame axis (SeriesAxis/ScalarAxis). "
            f"Got axis0={type(ax0)} axis1={type(ax1)}"
        )
    return frame_axis, brain_axis, frame_spec


def _get_cifti_frame_vector(
    img: nib.cifti2.cifti2.Cifti2Image,
    *,
    frame_index: int,
    frame_spec: _CiftiFrameSpec,
) -> np.ndarray:
    """Extract one CIFTI frame as a flat floating-point vector."""
    dataobj = img.dataobj
    if frame_spec.frame_axis_first:
        vec = np.asanyarray(dataobj[frame_index, :])
    else:
        vec = np.asanyarray(dataobj[:, frame_index])
    return np.asarray(vec, dtype=float).ravel()


def _extract_cortex_structures(brain_axis: Any) -> dict[str, tuple[slice, Any]]:
    """Collect cortex structures from a CIFTI brain-model axis."""
    structures: dict[str, tuple[slice, Any]] = {}
    for struct_name, struct_slice, struct_bm in brain_axis.iter_structures():
        structures[str(struct_name)] = (struct_slice, struct_bm)
    return structures


def _brain_to_hemi_vertices(
    *,
    frame_vec: np.ndarray,
    structures: dict[str, tuple[slice, Any]],
    structure_name: str,
) -> np.ndarray:
    """Project CIFTI brain-model values into a hemisphere vertex map."""
    if structure_name not in structures:
        raise ValueError(f"Structure {structure_name} not found in CIFTI brain models")
    struct_slice, struct_bm = structures[structure_name]
    vals = np.asarray(frame_vec[struct_slice], dtype=float)
    struct_bm_any = cast(Any, struct_bm)
    vertex = np.asarray(struct_bm_any.vertex, dtype=np.int64)
    nverts_dict = getattr(struct_bm_any, "nvertices", None)
    if isinstance(nverts_dict, dict) and structure_name in nverts_dict:
        n_verts = int(nverts_dict[structure_name])
    else:
        n_verts = int(vertex.max()) + 1 if vertex.size else int(vals.size)
    if int(vertex.size) != int(vals.size):
        raise ValueError(
            f"BrainModel vertex index length ({int(vertex.size)}) does not match values length ({int(vals.size)})"
        )
    out = np.full(n_verts, np.nan, dtype=float)
    out[vertex] = vals
    return out


def _extract_cifti_label_legend_from_label_axis(
    atlas_img: nib.cifti2.cifti2.Cifti2Image,
    *,
    name_transform: Callable[[str], str] | None = None,
) -> list[tuple[int, str, tuple[float, float, float, float]]]:
    """Read legend entries from a CIFTI label axis."""
    from nibabel.cifti2 import cifti2_axes

    ax0 = cifti2_axes.from_index_mapping(atlas_img.header.get_index_map(0))
    ax1 = cifti2_axes.from_index_mapping(atlas_img.header.get_index_map(1))
    if isinstance(ax0, cifti2_axes.LabelAxis):
        label_axis = ax0
    elif isinstance(ax1, cifti2_axes.LabelAxis):
        label_axis = ax1
    else:
        raise ValueError("Expected a CIFTI label axis for atlas labels")

    labels_any = cast(Any, getattr(label_axis, "label"))
    if isinstance(labels_any, dict):
        label_dict = labels_any
    elif hasattr(labels_any, "__len__") and len(labels_any) > 0:
        label_dict = cast(Any, labels_any[0])
        if not isinstance(label_dict, dict):
            raise ValueError("Expected CIFTI atlas labels to be stored as a mapping")
    else:
        raise ValueError("Expected CIFTI atlas labels to be stored as a mapping")

    entries: list[tuple[int, str, tuple[float, float, float, float]]] = []
    for key, value in label_dict.items():
        name, rgba = value
        rgba4 = tuple(float(x) for x in rgba)
        if len(rgba4) != 4:
            continue
        if _is_medial_wall_label_name(str(name)):
            continue
        display_name = (
            name_transform(str(name)) if name_transform is not None else str(name)
        )
        entries.append((int(key), display_name, _normalize_rgba(cast(Any, rgba4))))

    entries.sort(key=lambda item: item[0])
    return entries


def _build_cifti_network_overlay(
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    labels: list[tuple[int, str, tuple[float, float, float, float]]],
    *,
    keep_names: set[str] | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[int, str, tuple[float, float, float, float]]],
]:
    """Merge ROI parcels into per-network atlas contour masks."""
    if keep_names is None:
        keep_names = {str(name) for _key, name, _rgba in labels}

    selected_labels: list[tuple[int, str, tuple[float, float, float, float]]] = []
    seen_names: set[str] = set()
    for key, name, rgba in labels:
        if name not in keep_names or name in seen_names:
            continue
        seen_names.add(name)
        selected_labels.append((int(key), str(name), rgba))
    if not selected_labels:
        return (
            _empty_overlay_like(left_labels.size),
            _empty_overlay_like(right_labels.size),
            [],
        )

    name_to_roi_keys: dict[str, list[int]] = {}
    for key, name, _rgba in labels:
        name_to_roi_keys.setdefault(str(name), []).append(int(key))

    def _merge(values: np.ndarray) -> np.ndarray:
        source = np.asarray(values, dtype=float)
        merged = np.full(source.shape, np.nan, dtype=float)
        if source.size == 0:
            return merged
        for new_key, (_old_key, name, _rgba) in enumerate(selected_labels, start=1):
            roi_keys = name_to_roi_keys.get(name, [])
            if not roi_keys:
                continue
            merged[np.isin(source, roi_keys)] = float(new_key)
        return merged

    remapped_labels = [
        (int(idx + 1), name, rgba)
        for idx, (_key, name, rgba) in enumerate(selected_labels)
    ]
    return _merge(left_labels), _merge(right_labels), remapped_labels


def _load_medial_wall_mask_pair() -> _PairData:
    """Load the paired medial-wall masks used for surface masking."""
    root = Path(__file__).resolve().parent.parent.parent
    left_path = (
        root / "template" / "fsLR_hemi-L_den-32k_desc-nomedialwall_dparc.label.gii"
    )
    right_path = (
        root / "template" / "fsLR_hemi-R_den-32k_desc-nomedialwall_dparc.label.gii"
    )
    left_img = _load_gifti_image(left_path, kind="left medial wall mask")
    right_img = _load_gifti_image(right_path, kind="right medial wall mask")
    if len(left_img.darrays) == 0 or len(right_img.darrays) == 0:
        raise ValueError(
            "Medial wall mask GIFTI files must contain at least one data array"
        )
    return _PairData(
        left=np.asarray(left_img.darrays[0].data, dtype=float).ravel(),
        right=np.asarray(right_img.darrays[0].data, dtype=float).ravel(),
    )


def _apply_medial_wall_mask(
    left: np.ndarray, right: np.ndarray, mask: _PairData | None
) -> tuple[np.ndarray, np.ndarray]:
    """Mask vertices outside the medial wall keep region."""
    if mask is None:
        return left, right
    left_keep = np.asarray(mask.left, dtype=float) > 0
    right_keep = np.asarray(mask.right, dtype=float) > 0
    if left_keep.shape == left.shape:
        left = np.asarray(left, dtype=float).copy()
        left[~left_keep] = np.nan
    if right_keep.shape == right.shape:
        right = np.asarray(right, dtype=float).copy()
        right[~right_keep] = np.nan
    return left, right


def _load_named_cifti_overlay(
    args: _SnapshotParams, *, atlas_name: str
) -> _OverlayData | None:
    """Build an atlas overlay from a named CIFTI template."""
    root = Path(__file__).resolve().parent.parent.parent
    if atlas_name == "yeo":
        atlas_path = root / "template" / "Yeo2011_7Networks.split_components.dlabel.nii"
    elif atlas_name == "gordon":
        atlas_path = (
            root / "template" / "Gordon333_FreesurferSubcortical.32k_fs_LR.dlabel.nii"
        )
    else:
        raise ValueError(f"Unsupported atlas selection: {atlas_name}")
    if not atlas_path.exists():
        raise FileNotFoundError(str(atlas_path))
    atlas_img = nib_load(str(atlas_path))
    if not isinstance(atlas_img, nib.cifti2.cifti2.Cifti2Image):
        raise TypeError(f"Expected atlas CIFTI-2 image, got {type(atlas_img)}")

    _frame_axis, brain_axis, frame_spec = _infer_cifti_axes(atlas_img)
    frame_vec = _get_cifti_frame_vector(atlas_img, frame_index=0, frame_spec=frame_spec)
    structures = _extract_cortex_structures(brain_axis)
    labels = (
        _extract_cifti_label_legend_from_label_axis(
            atlas_img, name_transform=_yeo_cifti_network_name
        )
        if atlas_name == "yeo"
        else _extract_cifti_label_legend_from_label_axis(
            atlas_img, name_transform=_gordon_cifti_roi_network_name
        )
    )

    left_labels = _brain_to_hemi_vertices(
        frame_vec=frame_vec,
        structures=structures,
        structure_name="CIFTI_STRUCTURE_CORTEX_LEFT",
    )
    right_labels = _brain_to_hemi_vertices(
        frame_vec=frame_vec,
        structures=structures,
        structure_name="CIFTI_STRUCTURE_CORTEX_RIGHT",
    )
    left_labels, right_labels, labels = _build_cifti_network_overlay(
        left_labels,
        right_labels,
        labels,
        keep_names=_normalize_label_filter_values(
            cast(list[str] | None, args.atlas_label_index)
        ),
    )
    medial_wall_mask = (
        _load_medial_wall_mask_pair() if bool(args.exclude_medial_wall) else None
    )
    return _prepare_atlas_overlay(
        left_labels=left_labels,
        right_labels=right_labels,
        labels=labels,
        cmap_name=str(args.atlas_cmap),
        ignore_zero=bool(args.atlas_ignore_zero),
        ignore_medial_wall=False,
        ignore_label_values=None,
        medial_wall_mask=medial_wall_mask,
        use_label_table=bool(args.atlas_legend) and len(labels) > 0,
        output_hint=atlas_path,
        output_tag="atlas",
    )


def _load_gifti_image(path: Path, *, kind: str) -> GiftiImage:
    """Load a GIFTI image and verify its type."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    loaded = nib_load(str(path))
    if not isinstance(loaded, GiftiImage):
        raise TypeError(f"Expected {kind} GIFTI image, got {type(loaded)}")
    return loaded


def _get_gifti_frame(img: GiftiImage, index: int) -> np.ndarray:
    """Return one GIFTI data array as a flat floating-point vector."""
    if index < 0 or index >= len(img.darrays):
        raise ValueError(f"frame index out of range: {index}")
    return np.asarray(img.darrays[index].data, dtype=float).ravel()


def _strip_suffixes(name: str, suffixes: tuple[str, ...]) -> str:
    """Remove the first matching suffix from a filename."""
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _load_gifti_metric_source(args: _SnapshotParams) -> _MetricSource:
    """Load paired GIFTI metrics into a frame source."""
    left_path, right_path = args.input_left, args.input_right
    if left_path is None or right_path is None:
        raise ValueError("Both --input-left and --input-right must be provided")

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


def _copy_cmap_with_transparent_bad(cmap: Any) -> Any:
    """Copy a colormap and make its bad values transparent."""
    if isinstance(cmap, str):
        import matplotlib.pyplot as plt

        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap
    try:
        cmap_obj = cast(Any, cmap_obj).copy()
    except Exception:
        pass
    try:
        cast(Any, cmap_obj).set_bad((0.0, 0.0, 0.0, 0.0))
    except Exception:
        pass
    return cmap_obj


def _empty_overlay_like(size: int) -> np.ndarray:
    """Create an all-NaN overlay vector of the requested size."""
    return np.full((size,), np.nan, dtype=float)


def _prepare_atlas_overlay(
    *,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    labels: list[tuple[int, str, tuple[float, float, float, float]]],
    cmap_name: str,
    ignore_zero: bool,
    ignore_medial_wall: bool,
    ignore_label_values: frozenset[int] | None,
    medial_wall_mask: _PairData | None,
    use_label_table: bool,
    output_hint: Path,
    output_tag: str,
) -> _OverlayData:
    """Normalize CIFTI atlas label data into a renderable overlay payload."""
    left = np.asarray(left_labels, dtype=float).copy()
    right = np.asarray(right_labels, dtype=float).copy()
    if ignore_zero:
        left[left == 0] = np.nan
        right[right == 0] = np.nan

    if ignore_label_values:
        ignore_ids = np.asarray(sorted(ignore_label_values), dtype=np.int64)
        if ignore_ids.size > 0:
            left_finite = np.isfinite(left)
            right_finite = np.isfinite(right)
            if np.any(left_finite):
                left_mask = np.zeros(left.shape, dtype=bool)
                left_mask[left_finite] = np.isin(
                    left[left_finite].astype(np.int64, copy=False), ignore_ids
                )
                left[left_mask] = np.nan
            if np.any(right_finite):
                right_mask = np.zeros(right.shape, dtype=bool)
                right_mask[right_finite] = np.isin(
                    right[right_finite].astype(np.int64, copy=False), ignore_ids
                )
                right[right_mask] = np.nan

    if medial_wall_mask is not None:
        left_keep = np.asarray(medial_wall_mask.left, dtype=float) > 0
        right_keep = np.asarray(medial_wall_mask.right, dtype=float) > 0
        if left_keep.shape == left.shape:
            left[~left_keep] = np.nan
        if right_keep.shape == right.shape:
            right[~right_keep] = np.nan

    if ignore_medial_wall and len(labels) > 0:
        background_keys = [
            int(key) for key, name, _rgba in labels if _is_medial_wall_label_name(name)
        ]
        if background_keys:
            left[np.isin(left, background_keys)] = np.nan
            right[np.isin(right, background_keys)] = np.nan

    if use_label_table and len(labels) > 0:
        entries = labels
        if ignore_zero:
            entries = [entry for entry in entries if int(entry[0]) != 0]
        if ignore_medial_wall:
            entries = [
                entry for entry in entries if not _is_medial_wall_label_name(entry[1])
            ]
        keys = np.array([int(key) for key, _name, _rgba in entries], dtype=np.int64)
        colors = [rgba for _key, _name, rgba in entries]
        if keys.size > 0:
            order = np.argsort(keys)
            keys = keys[order]
            colors = [colors[int(i)] for i in order]
        cmap = _copy_cmap_with_transparent_bad(
            ListedColormap(colors, name=f"{output_tag}_label_table")
        )

        def _remap(arr: np.ndarray) -> np.ndarray:
            values = np.asarray(arr, dtype=float)
            out = np.full(values.shape, np.nan, dtype=float)
            finite_mask = np.isfinite(values)
            if not np.any(finite_mask) or keys.size == 0:
                return out
            label_vals = values[finite_mask].astype(np.int64, copy=False)
            idx = np.searchsorted(keys, label_vals)
            ok = (idx >= 0) & (idx < keys.size) & (keys[idx] == label_vals)
            tmp = np.full(label_vals.shape, np.nan, dtype=float)
            tmp[ok] = idx[ok].astype(float) + 1.0
            out[finite_mask] = tmp
            return out

        left_plot = _remap(left)
        right_plot = _remap(right)
        return _OverlayData(
            left_plot=[left_plot],
            right_plot=[right_plot],
            cmap=cmap,
            labels=entries,
            output_hint=output_hint,
            output_tag=output_tag,
        )

    cmap = _copy_cmap_with_transparent_bad(cmap_name)
    finite_values = np.concatenate(
        (left[np.isfinite(left)], right[np.isfinite(right)]), axis=0
    )
    if finite_values.size > 0:
        min_value = float(np.min(finite_values))
        max_value = float(np.max(finite_values))
        if min_value == max_value:
            min_value -= 0.5
            max_value += 0.5
    else:
        min_value = None
        max_value = None
    return _OverlayData(
        left_plot=[left],
        right_plot=[right],
        cmap=cmap,
        labels=labels,
        output_hint=output_hint,
        output_tag=output_tag,
    )


def _load_gifti_overlay(args: _SnapshotParams) -> _OverlayData | None:
    """Build a label overlay from one or more GIFTI label inputs."""
    # If no label inputs are provided, return None to indicate no overlay.
    if not args.roi_left and not args.roi_right:
        return None

    # Pair up left and right label files, allowing for missing sides.
    roi_pairs = list(zip_longest(args.roi_left or [], args.roi_right or []))

    # If there are no label pairs, return None to indicate no overlay.
    if not roi_pairs:
        return None

    # Pair up left and right color labels, allowing for missing sides.
    if args.roi_left_color or args.roi_right_color:
        # convert color strings to RGBA tuples
        _roi_left_color = [
            _convert_color_to_rgba(color) for color in (args.roi_left_color or [])
        ]
        _roi_right_color = [
            _convert_color_to_rgba(color) for color in (args.roi_right_color or [])
        ]
        color_pairs = list(zip_longest(_roi_left_color, _roi_right_color))
    else:
        # if no colors are provided, sample colors from the requested colormap
        colors = _sample_overlay_colors(str(args.roi_cmap), roi_pairs)
        color_pairs = []
        for _ in roi_pairs:
            color_pairs.append((colors.pop(0), colors.pop(0)))

    # Pair up left and right label names, allowing for missing sides.
    if args.roi_left_label or args.roi_right_label:
        label_pairs = list(
            zip_longest(args.roi_left_label or [], args.roi_right_label or [])
        )
    else:
        # create label pairs with enumerated names unique to each left and right hemi
        # if no label names are provided
        _roi_flat = [roi for pair in roi_pairs for roi in pair if roi is not None]
        _roi_labels = [f"ROI_{i + 1}" for i in range(len(_roi_flat))]
        label_pairs = []
        for _ in roi_pairs:
            label_pairs.append((_roi_labels.pop(0), _roi_labels.pop(0)))

    # if colors are provided for each label, check their length matches the number of label pairs
    if color_pairs and len(color_pairs) != len(roi_pairs):
        raise ValueError(
            f"Number of label colors ({len(color_pairs)}) does not match number of label pairs ({len(roi_pairs)})"
        )

    # if label names are provided for each label, check their length matches the number of label pairs
    if label_pairs and len(label_pairs) != len(roi_pairs):
        raise ValueError(
            f"Number of label names ({len(label_pairs)}) does not match number of label pairs ({len(roi_pairs)})"
        )

    # Load the medial wall mask if requested, otherwise set it to None.
    medial_wall_mask = (
        _load_medial_wall_mask_pair() if bool(args.exclude_medial_wall) else None
    )

    # Initialize ROI shapes and other variables for processing.
    left_shape = None
    right_shape = None
    roi_entries_left: list[tuple[int, str, tuple[float, float, float, float]]] = []
    roi_entries_right: list[tuple[int, str, tuple[float, float, float, float]]] = []
    roi_data_left: list[np.ndarray] = []
    roi_data_right: list[np.ndarray] = []

    # Process each pair of roi files, loading the GIFTI images and extracting label data.
    for pair_index, (pair, color_pair, label_pair) in enumerate(
        zip(roi_pairs, color_pairs, label_pairs)
    ):
        # Load the left and right GIFTI images for the current pair, if they exist.
        left_img = (
            _load_gifti_image(pair[0], kind="label") if pair[0] is not None else None
        )
        right_img = (
            _load_gifti_image(pair[1], kind="label") if pair[1] is not None else None
        )
        # If both images are missing, skip this pair.
        if left_img is None and right_img is None:
            continue
        # Validate that each GIFTI image contains at least one data array.
        if left_img is not None and len(left_img.darrays) < 1:
            raise ValueError("Label GIFTI files must contain at least one data array")
        if right_img is not None and len(right_img.darrays) < 1:
            raise ValueError("Label GIFTI files must contain at least one data array")

        # Extract the label data from the first data array of each GIFTI image, or create an empty overlay if one side is missing.
        if left_img is not None:
            left_data = np.asarray(left_img.darrays[0].data, dtype=float).ravel()
        else:
            assert right_img is not None
            left_data = _empty_overlay_like(
                int(np.asarray(right_img.darrays[0].data).size)
            )
        if right_img is not None:
            right_data = np.asarray(right_img.darrays[0].data, dtype=float).ravel()
        else:
            assert left_img is not None
            right_data = _empty_overlay_like(
                int(np.asarray(left_img.darrays[0].data).size)
            )

        if pair_index == 0:
            left_shape = tuple(left_data.shape)
            right_shape = tuple(right_data.shape)
        else:
            assert left_shape is not None and right_shape is not None
            if left_shape != left_data.shape or right_shape != right_data.shape:
                raise ValueError(
                    "All label GIFTI files must have matching vertex counts"
                )

        # set all non-zero values to 1.0
        non_zero_left_mask = np.isfinite(left_data) & (left_data != 0)
        non_zero_right_mask = np.isfinite(right_data) & (right_data != 0)
        left_data[non_zero_left_mask] = 1.0
        left_data[~non_zero_left_mask] = np.nan
        right_data[non_zero_right_mask] = 1.0
        right_data[~non_zero_right_mask] = np.nan

        if medial_wall_mask is not None:
            left_data, right_data = _apply_medial_wall_mask(
                left_data, right_data, medial_wall_mask
            )

        roi_data_left.append(left_data)
        roi_data_right.append(right_data)
        roi_entries_left.append(
            (int(pair_index + 1), str(label_pair[0]), color_pair[0])
        )
        roi_entries_right.append(
            (int(pair_index + 1), str(label_pair[1]), color_pair[1])
        )

    return _OverlayData(
        left_plot=roi_data_left,
        right_plot=roi_data_right,
        labels_left=roi_entries_left,
        labels_right=roi_entries_right,
        output_hint=roi_pairs[0][0]
        if roi_pairs[0][0] is not None
        else cast(Path, roi_pairs[0][1]),
        output_tag="roi",
    )


def _convert_color_to_rgba(color: Any) -> tuple[float, float, float, float]:
    """Convert a color specification to an RGBA tuple."""
    if not is_color_like(color):
        raise ValueError(f"Invalid color specification: {color}")
    rgba = to_rgba(color)
    return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))


def _detect_format(args: _SnapshotParams) -> Literal["gifti", "cifti"]:
    """Infer the source format from the provided snapshot arguments."""
    if args.input_left is not None or args.input_right is not None:
        return "gifti"
    if args.roi_left is not None or args.roi_right is not None:
        return "gifti"
    if args.atlas is not None:
        return "cifti"

    raise ValueError(
        "Could not infer input format. Use --format gifti or --format cifti."
    )


def load_snapshot_scene(
    *,
    input_left: Path | None = None,
    input_right: Path | None = None,
    roi_left: list[Path] | None = None,
    roi_right: list[Path] | None = None,
    atlas: str | None = None,
    atlas_label_index: list[str] | None = None,
    format: str = "auto",
    index: int = 0,
    exclude_medial_wall: bool = False,
    atlas_cmap: str = "tab20",
    atlas_ignore_zero: bool = True,
    atlas_legend: bool = False,
    atlas_legend_max_items: int = 40,
    roi_cmap: str = "tab20",
    roi_legend: bool = False,
) -> _SceneData:
    """Load the metric and overlay inputs needed for a snapshot render."""
    params = _SnapshotParams(
        input_left=input_left,
        input_right=input_right,
        roi_left=roi_left,
        roi_right=roi_right,
        atlas=atlas,
        atlas_label_index=atlas_label_index,
        format=format,
        index=index,
        exclude_medial_wall=exclude_medial_wall,
        atlas_cmap=atlas_cmap,
        atlas_ignore_zero=atlas_ignore_zero,
        atlas_legend=atlas_legend,
        atlas_legend_max_items=atlas_legend_max_items,
        roi_cmap=roi_cmap,
        roi_legend=roi_legend,
    )
    return _load_snapshot_scene(params)


def _load_snapshot_scene(args: _SnapshotParams) -> _SceneData:
    """Resolve snapshot inputs into a metric source and optional overlay."""
    source_format = _detect_format(args)
    has_cifti_overlay = args.atlas is not None
    has_gifti_overlay = any(
        value is not None for value in (args.roi_left, args.roi_right)
    )
    if has_cifti_overlay and has_gifti_overlay:
        raise ValueError("Use either --atlas or --roi/--roi-left/--roi-right, not both")
    if source_format == "cifti":
        if has_cifti_overlay:
            overlay = (
                _load_named_cifti_overlay(args, atlas_name="yeo")
                if str(args.atlas) == "yeo"
                else _load_named_cifti_overlay(args, atlas_name="gordon")
            )
        else:
            overlay = None
        return _SceneData(
            metric_source=None, overlay=overlay, source_format=source_format
        )

    metric_source = None
    if args.input_left is not None or args.input_right is not None:
        metric_source = _load_gifti_metric_source(args)

    overlay = (
        _load_named_cifti_overlay(args, atlas_name="yeo")
        if has_cifti_overlay and str(args.atlas) == "yeo"
        else _load_named_cifti_overlay(args, atlas_name="gordon")
        if has_cifti_overlay
        else _load_gifti_overlay(args)
    )
    if metric_source is None and overlay is None:
        raise ValueError("GIFTI mode requires metric input and/or label or atlas input")
    return _SceneData(
        metric_source=metric_source, overlay=overlay, source_format=source_format
    )


def _derive_default_output(
    path: Path, *, index: int, suffixes: tuple[str, ...]
) -> Path:
    """Derive a default output filename for metric snapshots."""
    return path.with_name(f"{_strip_suffixes(path.name, suffixes)}_idx-{index}.png")


def _derive_default_overlay_output(
    path: Path, *, tag: str, suffixes: tuple[str, ...]
) -> Path:
    """Derive a default output filename for overlay snapshots."""
    return path.with_name(f"{_strip_suffixes(path.name, suffixes)}_{tag}.png")


def select_snapshot_output_path(
    *,
    output: Path | None = None,
    scene: _SceneData,
    index: int = 0,
) -> Path:
    """Select the default output path for a prepared snapshot scene."""
    if output is not None:
        return Path(output)
    if scene.metric_source is not None:
        return _derive_default_output(
            scene.metric_source.output_hint,
            index=int(index),
            suffixes=(
                ".dtseries.nii",
                ".dscalar.nii",
                ".func.gii",
                ".nii.gz",
                ".nii",
                ".gii",
            ),
        )
    if scene.overlay is not None:
        return _derive_default_overlay_output(
            scene.overlay.output_hint,
            tag=scene.overlay.output_tag,
            suffixes=(
                ".dlabel.nii",
                ".label.gii",
                ".nii.gz",
                ".nii",
                ".gii",
            ),
        )
    raise RuntimeError("No metric or overlay input available to derive output path")


def build_snapshot_plotter(
    *,
    input_left: Path | None = None,
    input_right: Path | None = None,
    roi_left: list[Path] | None = None,
    roi_right: list[Path] | None = None,
    roi_left_color: list[str] | None = None,
    roi_right_color: list[str] | None = None,
    roi_left_label: list[str] | None = None,
    roi_right_label: list[str] | None = None,
    atlas: str | None = None,
    atlas_label_index: list[str] | None = None,
    index: int = 0,
    output: Path | None = None,
    surf_left: Path = _default_surface_path("fsaverage.L.inflated.32k_fs_LR.surf.gii"),
    surf_right: Path = _default_surface_path("fsaverage.R.inflated.32k_fs_LR.surf.gii"),
    sulc_left: Path = _default_surface_path("fsaverage.L.sulc.32k_fs_LR.surf.gii"),
    sulc_right: Path = _default_surface_path("fsaverage.R.sulc.32k_fs_LR.surf.gii"),
    surf_views: tuple[str, ...] = ("lateral", "medial"),
    surf_elev: float | None = None,
    surf_azim: float | None = None,
    surf_elev_left: float | None = None,
    surf_elev_right: float | None = None,
    surf_azim_left: float | None = None,
    surf_azim_right: float | None = None,
    surf_zoom: float = 1.8,
    mesh_alpha: float = 1.0,
    ncols: int | None = None,
    cmap: str = "coolwarm",
    roi_cmap: str = "tab20",
    vmin: float | None = None,
    vmax: float | None = None,
    intensity_mode: str = "global",
    auto_percentiles: tuple[float, float] = (1.0, 99.0),
    auto_max_samples: int = 200_000,
    auto_max_total_samples: int = 2_000_000,
    black_bg: bool = False,
    colorbar: bool = True,
    colorbar_side: str = "right",
    title: str | None = None,
    time_annotate: bool = False,
    tr: float | None = None,
    t0_trs: float = 0.0,
    dpi: int = 150,
    size: tuple[int, int] = (1280, 720),
    atlas_cmap: str = "tab20",
    atlas_ignore_zero: bool = True,
    atlas_legend: bool = False,
    atlas_legend_max_items: int = 40,
    roi_legend: bool = False,
    exclude_medial_wall: bool = False,
    threshold: float | None = None,
    sulc_file_reverse_sign: bool = True,
) -> SurfaceSnapshotPlotter:
    """Build a surface snapshot plotter from surface data and rendering options.

    Parameters
    ----------
    input_left : Path | None, optional
        Left-hemisphere metric Gifti file. Use together with ``input_right``.
    input_right : Path | None, optional
        Right-hemisphere metric Gifti file. Use together with ``input_left``.
    roi_left : list[Path] | None, optional
        Left-hemisphere label ROI GIFTI files to render as overlays. The label gifti
        is expected to have a single non-zero integer for a single ROI.
    roi_right : list[Path] | None, optional
        Right-hemisphere label ROI GIFTI files to render as overlays. The label gifti
        is expected to have a single non-zero integer for a single ROI.
    roi_left_color: list[str] | None, optional
        Optional list of color specifications for each left-hemisphere ROI file. Should be in HEX or any matplotlib-compatible color format.
        If passed, the list must be the same length as ``roi_left``. If passed, the roi_cmap argument is ignored.
    roi_right_color: list[str] | None, optional
        Optional list of color specifications for each right-hemisphere ROI file. Should be in HEX or any matplotlib-compatible color format.
        If passed, the list must be the same length as ``roi_right``. If passed, the roi_cmap argument is ignored.
    roi_right_label: list[str] | None, optional
        Optional list of label names for each right-hemisphere ROI file. Should be in string format.
        If passed, the list must be the same length as ``roi_right``. If not provided, ROI labels will be autogenerated.
        Note, roi_legend must be True for the labels to be displayed in the legend.
    roi_left_label: list[str] | None, optional
        Optional list of label names for each left-hemisphere ROI file. Should be in string format.
        If passed, the list must be the same length as ``roi_left``. If not provided, ROI labels will be autogenerated.
        Note, roi_legend must be True for the labels to be displayed in the legend.
    atlas : str | None, optional
        Named CIFTI atlas overlay to use instead of label GIFTI overlays. The
        supported values are currently ``"yeo"`` and ``"gordon"``.
    atlas_label_index : list[str] | None, optional
        Subset of atlas network labels to keep when rendering the atlas overlay.
        Labels are matched against the atlas network names after hemisphere and
        parcel suffixes are removed. If omitted, all atlas networks are merged
        and rendered as network-level contours.
    format : str, default="auto"
        Input format to use when loading the metric source. Use ``"gifti"`` or
        ``"cifti"`` to force a mode, or ``"auto"`` to infer it from the input
        paths.
    index : int, default=0
        Frame index to render from multi-frame inputs.
    output : Path | None, optional
        Output path hint used when deriving default filenames. The returned
        plotter does not write files itself.
    surf_left : Path, default=template fsaverage left inflated surface
        Left-hemisphere mesh used for rendering.
    surf_right : Path, default=template fsaverage right inflated surface
        Right-hemisphere mesh used for rendering.
    sulc_left : Path, default=template fsaverage left sulcal map
        Left-hemisphere sulcal background map.
    sulc_right : Path, default=template fsaverage right sulcal map
        Right-hemisphere sulcal background map.
    surf_views : tuple[str, ...], default=("lateral", "medial")
        Surface views to render for each hemisphere.
    surf_elev : float | None, optional
        Shared camera elevation for both hemispheres. If set without
        ``surf_azim``, a ValueError is raised when rendering.
    surf_azim : float | None, optional
        Shared camera azimuth for both hemispheres. If set without
        ``surf_elev``, a ValueError is raised when rendering.
    surf_elev_left : float | None, optional
        Per-hemisphere elevation override for the left hemisphere.
    surf_elev_right : float | None, optional
        Per-hemisphere elevation override for the right hemisphere.
    surf_azim_left : float | None, optional
        Per-hemisphere azimuth override for the left hemisphere.
    surf_azim_right : float | None, optional
        Per-hemisphere azimuth override for the right hemisphere.
    surf_zoom : float, default=1.8
        Camera zoom factor applied to each 3D axis after rendering.
    mesh_alpha : float, default=1.0
        Alpha used when drawing the statistical surface map.
    ncols : int | None, optional
        Number of columns in the panel layout. If omitted, all panels are placed
        on a single row.
    cmap : str, default="coolwarm"
        Colormap used for the statistical metric data.
    roi_cmap : str, default="tab20"
        Colormap used for ROI overlays when a label table is not embedded.
    vmin : float | None, optional
        Lower bound for the metric color scale. If omitted, the bound may be
        inferred from the metric data or overlay data.
    vmax : float | None, optional
        Upper bound for the metric color scale. If omitted, the bound may be
        inferred from the metric data or overlay data.
    intensity_mode : str, default="global"
        Strategy used to estimate automatic metric bounds. ``"global"`` samples
        across all frames; other modes use the selected frame only.
    auto_percentiles : tuple[float, float], default=(1.0, 99.0)
        Percentiles used when inferring metric bounds automatically.
    auto_max_samples : int, default=200_000
        Maximum number of samples drawn when estimating per-frame intensity
        bounds.
    auto_max_total_samples : int, default=2_000_000
        Maximum number of samples drawn across all frames when using global
        intensity mode.
    black_bg : bool, default=False
        Whether to render figures on a black background.
    colorbar : bool, default=True
        Whether to draw a colorbar when metric bounds are available.
    colorbar_side : str, default="right"
        Side on which the colorbar or legend column is placed.
    title : str | None, optional
        Optional title template passed through to the surface renderer.
    time_annotate : bool, default=False
        Whether to annotate the figure with a time label.
    tr : float | None, optional
        Repetition time in seconds. Required when ``time_annotate`` is true.
    t0_trs : float, default=0.0
        Starting TR offset used when computing the annotated time.
    dpi : int, default=150
        Figure DPI.
    size : tuple[int, int], default=(1280, 720)
        Output figure size in pixels.
    atlas_cmap : str, default="tab20"
        Colormap used for atlas overlays.
    atlas_ignore_zero : bool, default=True
        Whether to treat atlas label value 0 as background.
    atlas_legend : bool, default=False
        Whether to draw a legend for atlas overlays.
    atlas_legend_max_items : int, default=40
        Maximum number of atlas legend entries to show.
    roi_legend : bool, default=False
        Whether to draw a legend for ROI overlays.
    exclude_medial_wall : bool, default=False
        Whether to mask medial wall vertices before rendering metric and overlay
        data.
    threshold : float | None, optional
        Optional threshold passed to nilearn surface plotting for metric data.
    sulc_file_reverse_sign : bool, default=False
        Whether to negate the sulcal background maps before rendering.

    Returns
    -------
    SurfaceSnapshotPlotter
        Configured plotter ready to render a figure or draw into existing axes.
    """
    params = _SnapshotParams(
        input_left=input_left,
        input_right=input_right,
        roi_left=roi_left,
        roi_right=roi_right,
        roi_left_color=roi_left_color,
        roi_right_color=roi_right_color,
        roi_left_label=roi_left_label,
        roi_right_label=roi_right_label,
        atlas=atlas,
        atlas_label_index=atlas_label_index,
        index=index,
        output=output,
        surf_left=surf_left,
        surf_right=surf_right,
        sulc_left=sulc_left,
        sulc_right=sulc_right,
        surf_views=surf_views,
        surf_elev=surf_elev,
        surf_azim=surf_azim,
        surf_elev_left=surf_elev_left,
        surf_elev_right=surf_elev_right,
        surf_azim_left=surf_azim_left,
        surf_azim_right=surf_azim_right,
        surf_zoom=surf_zoom,
        mesh_alpha=mesh_alpha,
        ncols=ncols,
        cmap=cmap,
        roi_cmap=roi_cmap,
        vmin=vmin,
        vmax=vmax,
        intensity_mode=intensity_mode,
        auto_percentiles=auto_percentiles,
        auto_max_samples=auto_max_samples,
        auto_max_total_samples=auto_max_total_samples,
        black_bg=black_bg,
        colorbar=colorbar,
        colorbar_side=colorbar_side,
        title=title,
        time_annotate=time_annotate,
        tr=tr,
        t0_trs=t0_trs,
        dpi=dpi,
        size=size,
        atlas_cmap=atlas_cmap,
        atlas_ignore_zero=atlas_ignore_zero,
        atlas_legend=atlas_legend,
        atlas_legend_max_items=atlas_legend_max_items,
        roi_legend=roi_legend,
        exclude_medial_wall=exclude_medial_wall,
        threshold=threshold,
        sulc_file_reverse_sign=sulc_file_reverse_sign,
    )
    scene = _load_snapshot_scene(params)
    surf_left_mesh = surface.load_surf_mesh(str(params.surf_left))
    surf_right_mesh = surface.load_surf_mesh(str(params.surf_right))
    surf_left_sulc = surface.load_surf_data(str(params.sulc_left))
    surf_right_sulc = surface.load_surf_data(str(params.sulc_right))
    if bool(params.sulc_file_reverse_sign):
        surf_left_sulc = -surf_left_sulc
        surf_right_sulc = -surf_right_sulc

    selected_left: np.ndarray | None = None
    selected_right: np.ndarray | None = None
    if scene.metric_source is not None:
        selected_left, selected_right = scene.metric_source.get_frame_maps(
            int(params.index)
        )
        validate_map_against_mesh(
            selected_left,
            surf_left_mesh,
            kind="Left metric",
            source_type="metric",
            source_format=scene.source_format,
        )
        validate_map_against_mesh(
            selected_right,
            surf_right_mesh,
            kind="Right metric",
            source_type="metric",
            source_format=scene.source_format,
        )
    if scene.overlay is not None:
        validate_map_against_mesh(
            scene.overlay.left_plot,
            surf_left_mesh,
            kind="Left overlay",
            source_type="overlay",
            source_format=scene.source_format,
        )
        validate_map_against_mesh(
            scene.overlay.right_plot,
            surf_right_mesh,
            kind="Right overlay",
            source_type="overlay",
            source_format=scene.source_format,
        )

    if scene.metric_source is not None:
        p_low, p_high = _validate_percentiles(
            float(params.auto_percentiles[0]), float(params.auto_percentiles[1])
        )
        vmin, vmax = compute_intensity_bounds(
            cast(Any, scene.metric_source),
            selected_index=int(params.index),
            intensity_mode=str(params.intensity_mode),
            p_low=p_low,
            p_high=p_high,
            max_samples=int(params.auto_max_samples),
            max_total_samples=int(params.auto_max_total_samples),
            vmin_arg=params.vmin,
            vmax_arg=params.vmax,
        )
    elif params.vmin is not None or params.vmax is not None:
        vmin = params.vmin
        vmax = params.vmax
    else:
        vmin = None
        vmax = None

    time_s: float | None = None
    if bool(params.time_annotate):
        if scene.metric_source is None:
            raise ValueError("--time-annotate requires metric input")
        if params.tr is None:
            raise ValueError("--tr is required when --time-annotate is true")
        time_s = (float(params.t0_trs) + float(params.index)) * float(params.tr)

    overlay = None
    if scene.overlay is not None:
        overlay = SurfaceOverlay(
            left_plot=scene.overlay.left_plot,
            right_plot=scene.overlay.right_plot,
            cmap=scene.overlay.cmap,
            labels=scene.overlay.labels,
            labels_left=scene.overlay.labels_left,
            labels_right=scene.overlay.labels_right,
            output_tag=scene.overlay.output_tag,
        )

    options = SurfaceRenderOptions(
        views=[str(view) for view in params.surf_views],
        figure_size=(
            int(params.size[0]) / int(params.dpi),
            int(params.size[1]) / int(params.dpi),
        ),
        dpi=int(params.dpi),
        ncols=int(params.ncols) if params.ncols is not None else None,
        cmap=str(params.cmap),
        vmin=vmin,
        vmax=vmax,
        mesh_alpha=float(params.mesh_alpha),
        surf_zoom=float(params.surf_zoom),
        black_bg=bool(params.black_bg),
        colorbar=bool(params.colorbar),
        colorbar_side=str(params.colorbar_side),
        title_template=params.title,
        source_format=scene.source_format,
        atlas_ignore_zero=bool(params.atlas_ignore_zero),
        atlas_legend=bool(params.atlas_legend),
        atlas_legend_max_items=int(params.atlas_legend_max_items),
        roi_legend=bool(params.roi_legend),
        exclude_medial_wall=bool(params.exclude_medial_wall),
        time_annotate=bool(params.time_annotate),
        time_s=time_s,
        threshold=params.threshold,
        left_camera=SurfaceCameraConfig(
            elev=params.surf_elev_left, azim=params.surf_azim_left
        ),
        right_camera=SurfaceCameraConfig(
            elev=params.surf_elev_right, azim=params.surf_azim_right
        ),
    )

    return SurfaceSnapshotPlotter(
        surf_left_mesh=surf_left_mesh,
        surf_right_mesh=surf_right_mesh,
        surf_left_sulc=surf_left_sulc,
        surf_right_sulc=surf_right_sulc,
        options=options,
        stat_map_left=selected_left,
        stat_map_right=selected_right,
        overlay=overlay,
        index=int(params.index),
    )


@dataclasses.dataclass(frozen=True)
class SurfaceSnapshotPlotter:
    """Container for loaded snapshot data and rendering settings."""

    surf_left_mesh: Any
    surf_right_mesh: Any
    surf_left_sulc: np.ndarray
    surf_right_sulc: np.ndarray
    options: SurfaceRenderOptions
    stat_map_left: np.ndarray | None = None
    stat_map_right: np.ndarray | None = None
    overlay: SurfaceOverlay | None = None
    index: int = 0

    def render_into_axes(
        self,
        *,
        axes: list[Any],
        colorbar_axis: Any | None = None,
        legend_axis: Any | None = None,
        legend_kwargs: dict[str, Any] | None = None,
        colorbar_kwargs: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Render the snapshot into an existing axes layout."""
        return render_surface_into_axes(
            axes=axes,
            colorbar_axis=colorbar_axis,
            legend_axis=legend_axis,
            legend_kwargs=legend_kwargs,
            colorbar_kwargs=colorbar_kwargs,
            surf_left_mesh=self.surf_left_mesh,
            surf_right_mesh=self.surf_right_mesh,
            surf_left_sulc=self.surf_left_sulc,
            surf_right_sulc=self.surf_right_sulc,
            stat_map_left=self.stat_map_left,
            stat_map_right=self.stat_map_right,
            overlay=self.overlay,
            options=self.options,
            index=self.index,
        )

    def render_figure(
        self,
        *,
        colorbar_kwargs: dict[str, Any] | None = None,
    ) -> RenderedSurfaceFigure:
        """Render the snapshot into a new Matplotlib figure."""
        return render_surface_figure(
            surf_left_mesh=self.surf_left_mesh,
            surf_right_mesh=self.surf_right_mesh,
            surf_left_sulc=self.surf_left_sulc,
            surf_right_sulc=self.surf_right_sulc,
            stat_map_left=self.stat_map_left,
            stat_map_right=self.stat_map_right,
            overlay=self.overlay,
            options=self.options,
            index=self.index,
            colorbar_kwargs=colorbar_kwargs,
        )


__all__ = [
    "SurfaceSnapshotPlotter",
    "build_snapshot_plotter",
    "load_snapshot_scene",
    "select_snapshot_output_path",
]

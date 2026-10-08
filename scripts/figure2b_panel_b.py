"""Render a Figure 2B-style SCAN versus effector FC summary figure.

Inputs:
- Four paired lh/rh FC maps in GIFTI metric format for inter-effector, foot,
  hand, and mouth.
- A hard-coded CIFTI atlas selection (`yeo` or `gordon`) used to aggregate
    network-level values and define the cingulo-opercular family.

The surface panel displays the conservative contrast:

    inter-effector - max(foot, hand, mouth)

The bar panel computes the same conservative difference per atlas network using
the atlas label indices.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Colormap, ListedColormap, to_rgba
from nibabel.cifti2.cifti2 import Cifti2Image
from nibabel.gifti.gifti import GiftiImage
from nibabel.loadsave import load as nib_load
from nilearn import surface

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scan.plots import (
    SurfaceOverlay,
    SurfaceRenderOptions,
    render_surface_figure,
    validate_map_against_mesh,
)

LH_MEDIAL_WALL_MASK = "template/fsLR_hemi-L_den-32k_desc-nomedialwall_dparc.label.gii"
RH_MEDIAL_WALL_MASK = "template/fsLR_hemi-R_den-32k_desc-nomedialwall_dparc.label.gii"


@dataclass(frozen=True)
class PairData:
    left: np.ndarray
    right: np.ndarray


@dataclass(frozen=True)
class NetworkSummary:
    index: int
    name: str
    color: tuple[float, float, float, float]
    value: float


@dataclass(frozen=True)
class AtlasLabelInfo:
    parcel_name: str
    family_name: str
    color: tuple[float, float, float, float]


@dataclass(frozen=True)
class AtlasSelection:
    atlas: PairData
    label_lookup: dict[int, AtlasLabelInfo]
    medial_wall_mask: PairData
    con_family_name: str


def _default_surface_path(filename: str) -> Path:
    return ROOT / "template" / filename


def _strip_suffixes(name: str, suffixes: tuple[str, ...]) -> str:
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _normalize_rgba(
    rgba4: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    mx = max(rgba4)
    if mx > 1.0 and mx <= 255.0:
        return tuple(float(x) / 255.0 for x in rgba4)  # type: ignore[return-value]
    return rgba4


def _cifti_family_name(parcel_name: str) -> str:
    parts = str(parcel_name).split("_")
    if len(parts) >= 3 and parts[0] == "7Networks":
        return str(parts[2])
    if len(parts) >= 2:
        return str(parts[1])
    return str(parcel_name)


def _is_ignored_atlas_label_name(name: str) -> bool:
    normalized = str(name).strip().lower().replace("_", "").replace("-", "")
    ignore_labels = {"???", "none", "medialwall", "unknown", "unlabeled", "background"}
    return any(label in normalized for label in ignore_labels)


def _normalize_network_name(name: str) -> str:
    return str(name).strip()


def _excluded_label_ids(
    label_lookup: dict[int, AtlasLabelInfo], excluded_networks: set[str]
) -> set[int]:
    if not excluded_networks:
        return set()
    return {
        label_id
        for label_id, info in label_lookup.items()
        if _normalize_network_name(info.family_name) in excluded_networks
    }


def _mask_atlas_background(values: PairData, mask: PairData) -> PairData:
    left = np.asarray(values.left, dtype=float).copy()
    right = np.asarray(values.right, dtype=float).copy()
    left[np.asarray(mask.left, dtype=float) <= 0] = np.nan
    right[np.asarray(mask.right, dtype=float) <= 0] = np.nan
    return PairData(left=left, right=right)


def _load_medial_wall_mask() -> PairData:
    left_mask = _load_gifti_image(
        ROOT / LH_MEDIAL_WALL_MASK, kind="left medial wall mask"
    )
    right_mask = _load_gifti_image(
        ROOT / RH_MEDIAL_WALL_MASK, kind="right medial wall mask"
    )
    if len(left_mask.darrays) == 0 or len(right_mask.darrays) == 0:
        raise ValueError("Medial wall mask GIFTI files must contain at least one frame")
    return PairData(
        left=np.asarray(left_mask.darrays[0].data, dtype=float).ravel(),
        right=np.asarray(right_mask.darrays[0].data, dtype=float).ravel(),
    )


def _copy_cmap_with_transparent_bad(cmap: Colormap | str) -> Colormap | str:
    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap
    try:
        cmap_obj = cast(Any, cmap_obj).copy()
    except (AttributeError, TypeError):
        pass
    try:
        cast(Any, cmap_obj).set_bad((0.0, 0.0, 0.0, 0.0))
    except (AttributeError, TypeError, ValueError):
        pass
    return cmap_obj


def _load_gifti_image(path: Path, *, kind: str) -> GiftiImage:
    if not path.exists():
        raise FileNotFoundError(str(path))
    loaded = nib_load(str(path))
    if not isinstance(loaded, GiftiImage):
        raise TypeError(f"Expected {kind} GIFTI image, got {type(loaded)}")
    return loaded


def _extract_cifti_label_legend(
    atlas_img: Any,
) -> list[tuple[int, str, tuple[float, float, float, float]]]:
    try:
        from nibabel.cifti2 import cifti2_axes

        ax0 = cifti2_axes.from_index_mapping(atlas_img.header.get_index_map(0))
        ax1 = cifti2_axes.from_index_mapping(atlas_img.header.get_index_map(1))
        label_axis = None
        if isinstance(ax0, cifti2_axes.LabelAxis):
            label_axis = ax0
        elif isinstance(ax1, cifti2_axes.LabelAxis):
            label_axis = ax1
        if label_axis is None:
            return []

        labels_any = cast(Any, getattr(label_axis, "label", None))
        if labels_any is None or len(labels_any) == 0:
            return []

        entries: list[tuple[int, str, tuple[float, float, float, float]]] = []
        for key, value in cast(Any, labels_any[0]).items():
            try:
                name, rgba = value
                rgba4 = tuple(float(x) for x in rgba)
            except (TypeError, ValueError):
                continue
            if len(rgba4) != 4:
                continue
            if _is_ignored_atlas_label_name(str(name)):
                continue
            entries.append((int(key), str(name), _normalize_rgba(cast(Any, rgba4))))
        entries.sort(key=lambda item: item[0])
        return entries
    except (AttributeError, IndexError, TypeError, ValueError):
        return []


def _infer_cifti_axes(atlas_img: Any) -> tuple[Any, Any, bool]:
    from nibabel.cifti2 import cifti2_axes

    ax0 = cifti2_axes.from_index_mapping(atlas_img.header.get_index_map(0))
    ax1 = cifti2_axes.from_index_mapping(atlas_img.header.get_index_map(1))
    if isinstance(ax0, cifti2_axes.BrainModelAxis) and not isinstance(
        ax1, cifti2_axes.BrainModelAxis
    ):
        return ax1, ax0, False
    if isinstance(ax1, cifti2_axes.BrainModelAxis) and not isinstance(
        ax0, cifti2_axes.BrainModelAxis
    ):
        return ax0, ax1, True
    raise ValueError(
        "Expected one BrainModelAxis and one frame axis (SeriesAxis/ScalarAxis)."
    )


def _get_cifti_frame_vector(
    atlas_img: Any, *, frame_spec_first: bool, frame_index: int
) -> np.ndarray:
    dataobj = atlas_img.dataobj
    if frame_spec_first:
        vec = np.asanyarray(dataobj[frame_index, :])
    else:
        vec = np.asanyarray(dataobj[:, frame_index])
    return np.asarray(vec, dtype=float).ravel()


def _extract_cortex_structures(brain_axis: Any) -> dict[str, tuple[slice, Any]]:
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


def _load_cifti_atlas(
    path: Path,
) -> tuple[PairData, dict[int, AtlasLabelInfo]]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    loaded = nib_load(str(path))
    if not isinstance(loaded, Cifti2Image):
        raise TypeError(f"Expected CIFTI-2 atlas image, got {type(loaded)}")

    _frame_axis, brain_axis, frame_first = _infer_cifti_axes(loaded)
    structures = _extract_cortex_structures(brain_axis)
    frame_vec = _get_cifti_frame_vector(
        loaded, frame_spec_first=frame_first, frame_index=0
    )
    left = _brain_to_hemi_vertices(
        frame_vec=frame_vec,
        structures=structures,
        structure_name="CIFTI_STRUCTURE_CORTEX_LEFT",
    )
    right = _brain_to_hemi_vertices(
        frame_vec=frame_vec,
        structures=structures,
        structure_name="CIFTI_STRUCTURE_CORTEX_RIGHT",
    )
    legend = {
        key: AtlasLabelInfo(
            parcel_name=str(name),
            family_name=_cifti_family_name(str(name)),
            color=rgba,
        )
        for key, name, rgba in _extract_cifti_label_legend(loaded)
    }
    return PairData(left=left, right=right), legend


def _load_yeo_atlas() -> AtlasSelection:
    atlas, label_lookup = _load_cifti_atlas(
        ROOT / "template" / "Yeo2011_7Networks.split_components.dlabel.nii"
    )
    return AtlasSelection(
        atlas=atlas,
        label_lookup=label_lookup,
        medial_wall_mask=_load_medial_wall_mask(),
        con_family_name="Cont",
    )


def _load_gordon_atlas() -> AtlasSelection:
    atlas, label_lookup = _load_cifti_atlas(
        ROOT / "template" / "Gordon333_FreesurferSubcortical.32k_fs_LR.dlabel.nii"
    )
    return AtlasSelection(
        atlas=atlas,
        label_lookup=label_lookup,
        medial_wall_mask=_load_medial_wall_mask(),
        con_family_name="CinguloOperc",
    )


def _get_gifti_frame(img: GiftiImage, index: int) -> np.ndarray:
    if index < 0 or index >= len(img.darrays):
        raise ValueError(f"frame index out of range: {index}")
    return np.asarray(img.darrays[index].data, dtype=float).ravel()


def _load_metric_pair(
    *, left_path: Path, right_path: Path, index: int, label: str
) -> PairData:
    left_img = _load_gifti_image(left_path, kind=f"{label} metric")
    right_img = _load_gifti_image(right_path, kind=f"{label} metric")
    if len(left_img.darrays) != len(right_img.darrays):
        raise ValueError(
            f"{label} left/right GIFTI files must have the same number of frames"
        )
    if len(left_img.darrays) == 0:
        raise ValueError(f"{label} GIFTI files contain no data arrays")
    left = _get_gifti_frame(left_img, index)
    right = _get_gifti_frame(right_img, index)
    return PairData(left=left, right=right)


def _compute_difference(
    scan_fc: PairData, foot_fc: PairData, hand_fc: PairData, mouth_fc: PairData
) -> PairData:
    left = scan_fc.left - np.maximum.reduce([foot_fc.left, hand_fc.left, mouth_fc.left])
    right = scan_fc.right - np.maximum.reduce(
        [foot_fc.right, hand_fc.right, mouth_fc.right]
    )
    return PairData(left=left, right=right)


def _build_network_summaries(
    *,
    atlas: PairData,
    label_lookup: dict[int, AtlasLabelInfo],
    medial_wall_mask: PairData,
    excluded_networks: set[str],
    scan_fc: PairData,
    foot_fc: PairData,
    hand_fc: PairData,
    mouth_fc: PairData,
    con_family_name: str,
    network_order: list[int] | None,
) -> list[NetworkSummary]:
    keep_left = np.asarray(medial_wall_mask.left, dtype=float) > 0
    keep_right = np.asarray(medial_wall_mask.right, dtype=float) > 0
    family_to_labels: dict[str, list[int]] = {}
    family_order: list[str] = []
    for label_index in sorted(label_lookup):
        info = label_lookup[label_index]
        if _normalize_network_name(info.family_name) in excluded_networks:
            continue
        if info.family_name not in family_to_labels:
            family_to_labels[info.family_name] = []
            family_order.append(info.family_name)
        family_to_labels[info.family_name].append(label_index)

    if network_order is not None and len(network_order) > 0:
        ordered_families: list[str] = []
        for label_index in network_order:
            info = label_lookup.get(int(label_index))
            if info is None:
                continue
            if _normalize_network_name(info.family_name) in excluded_networks:
                continue
            if info.family_name not in ordered_families:
                ordered_families.append(info.family_name)
        ordered_families.extend(
            family for family in family_order if family not in ordered_families
        )
        family_order = ordered_families

    summaries: list[NetworkSummary] = []
    fallback_colors = plt.get_cmap("tab20")

    for idx, family_name in enumerate(family_order):
        label_ids = family_to_labels[family_name]
        left_mask = (
            np.isin(atlas.left.astype(np.int64, copy=False), label_ids) & keep_left
        )
        right_mask = (
            np.isin(atlas.right.astype(np.int64, copy=False), label_ids) & keep_right
        )
        if not np.any(left_mask) and not np.any(right_mask):
            continue

        scan_vals = np.concatenate(
            (scan_fc.left[left_mask], scan_fc.right[right_mask]), axis=0
        )
        foot_vals = np.concatenate(
            (foot_fc.left[left_mask], foot_fc.right[right_mask]), axis=0
        )
        hand_vals = np.concatenate(
            (hand_fc.left[left_mask], hand_fc.right[right_mask]), axis=0
        )
        mouth_vals = np.concatenate(
            (mouth_fc.left[left_mask], mouth_fc.right[right_mask]), axis=0
        )
        value = float(np.mean(scan_vals)) - max(
            float(np.mean(foot_vals)),
            float(np.mean(hand_vals)),
            float(np.mean(mouth_vals)),
        )

        family_color = (
            label_lookup[label_ids[0]].color
            if label_ids
            else _normalize_rgba(cast(Any, fallback_colors(idx % fallback_colors.N)))
        )
        summaries.append(
            NetworkSummary(
                index=label_ids[0],
                name=family_name,
                color=family_color,
                value=value,
            )
        )
    if con_family_name not in {summary.name for summary in summaries}:
        raise ValueError(
            f"CON family {con_family_name} is not present in the atlas labels"
        )
    return summaries


def _mask_excluded_networks(
    values: PairData, atlas: PairData, excluded_label_ids: set[int]
) -> PairData:
    if not excluded_label_ids:
        return values
    left = np.asarray(values.left, dtype=float).copy()
    right = np.asarray(values.right, dtype=float).copy()
    excluded_ids = np.asarray(sorted(excluded_label_ids), dtype=np.int64)
    left[np.isin(atlas.left.astype(np.int64, copy=False), excluded_ids)] = np.nan
    right[np.isin(atlas.right.astype(np.int64, copy=False), excluded_ids)] = np.nan
    return PairData(left=left, right=right)


def _build_con_overlay(
    *,
    atlas: PairData,
    con_family_name: str,
    label_lookup: dict[int, AtlasLabelInfo],
    medial_wall_mask: PairData,
    con_color: str,
) -> SurfaceOverlay:
    keep_left = np.asarray(medial_wall_mask.left, dtype=float) > 0
    keep_right = np.asarray(medial_wall_mask.right, dtype=float) > 0
    con_label_ids = [
        label_id
        for label_id, info in label_lookup.items()
        if info.family_name == con_family_name
    ]
    if not con_label_ids:
        raise ValueError(
            f"CON family {con_family_name} does not contain any non-background atlas labels"
        )
    left = np.where(
        np.isin(atlas.left.astype(np.int64, copy=False), con_label_ids) & keep_left,
        1.0,
        np.nan,
    )
    right = np.where(
        np.isin(atlas.right.astype(np.int64, copy=False), con_label_ids) & keep_right,
        1.0,
        np.nan,
    )
    cmap = ListedColormap([con_color], name="con_contour")
    cmap = cast(Colormap, _copy_cmap_with_transparent_bad(cmap))
    return SurfaceOverlay(
        left_plot=[left],
        right_plot=[right],
        output_tag="roi",
        cmap=cmap,
        labels_left=[
            (
                1,
                "CON",
                _normalize_rgba(cast(Any, to_rgba(con_color))),
            )
        ],
        labels_right=[
            (
                1,
                "CON",
                _normalize_rgba(cast(Any, to_rgba(con_color))),
            )
        ],
    )


def _compute_vrange(
    diff_fc: PairData, *, percentiles: tuple[float, float], symmetric: bool
) -> tuple[float, float]:
    values = np.concatenate((diff_fc.left, diff_fc.right), axis=0)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("FC difference map contains no finite values")
    low, high = np.percentile(values, list(percentiles))
    if symmetric:
        bound = float(max(abs(low), abs(high)))
        return -bound, bound
    return float(low), float(high)


def _render_surface_image(
    *,
    diff_fc: PairData,
    overlay: SurfaceOverlay,
    surf_left_mesh: Any,
    surf_right_mesh: Any,
    surf_left_sulc: np.ndarray,
    surf_right_sulc: np.ndarray,
    figure_size: tuple[float, float],
    dpi: int,
    cmap: str,
    vmin: float,
    vmax: float,
    threshold: float,
    sulc_file_reverse_sign: bool = True,
) -> np.ndarray:
    if sulc_file_reverse_sign:
        surf_left_sulc = -surf_left_sulc
        surf_right_sulc = -surf_right_sulc

    rendered = render_surface_figure(
        surf_left_mesh=surf_left_mesh,
        surf_right_mesh=surf_right_mesh,
        surf_left_sulc=surf_left_sulc,
        surf_right_sulc=surf_right_sulc,
        stat_map_left=diff_fc.left,
        stat_map_right=diff_fc.right,
        overlay=overlay,
        options=SurfaceRenderOptions(
            views=["lateral", "medial"],
            figure_size=figure_size,
            dpi=dpi,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar=True,
            colorbar_side="right",
            mesh_alpha=1.0,
            surf_zoom=1.6,
            title_template=None,
            source_format="gifti",
            threshold=threshold,
            exclude_medial_wall=False,
        ),
        index=0,
    )
    canvas = FigureCanvasAgg(rendered.figure)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba()).copy()
    plt.close(rendered.figure)
    return rgba


def _plot_summary_bars(
    *,
    ax: Axes,
    summaries: list[NetworkSummary],
    con_family_name: str,
    bar_width: float,
) -> None:
    positions = np.arange(len(summaries), dtype=float)
    colors = [summary.color for summary in summaries]
    edgecolors = [
        "black" if summary.name == con_family_name else "none" for summary in summaries
    ]
    linewidths = [
        1.5 if summary.name == con_family_name else 0.0 for summary in summaries
    ]
    values = [summary.value for summary in summaries]

    ax.bar(
        positions,
        values,
        width=bar_width,
        color=colors,
        edgecolor=edgecolors,
        linewidth=linewidths,
    )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels([summary.name for summary in summaries], rotation=50, ha="right")
    ax.set_ylabel("Δ Functional connectivity Z(r)")
    ax.set_title("Inter-effector versus effector-specific\nfunctional connectivity")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.02)


def _derive_output_path(scan_left_path: Path) -> Path:
    stem = _strip_suffixes(scan_left_path.name, (".func.gii", ".gii"))
    return scan_left_path.with_name(f"{stem}_figure2b_panel_b.png")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a Figure 2B-style FC difference figure from GIFTI inputs."
    )
    parser.add_argument("--scan-left", type=Path, required=True)
    parser.add_argument("--scan-right", type=Path, required=True)
    parser.add_argument("--foot-left", type=Path, required=True)
    parser.add_argument("--foot-right", type=Path, required=True)
    parser.add_argument("--hand-left", type=Path, required=True)
    parser.add_argument("--hand-right", type=Path, required=True)
    parser.add_argument("--mouth-left", type=Path, required=True)
    parser.add_argument("--mouth-right", type=Path, required=True)
    parser.add_argument(
        "--atlas",
        choices=["yeo", "gordon"],
        required=True,
        help="Select the hard-coded atlas loader to use.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--network-order",
        nargs="+",
        type=int,
        default=None,
        help="Optional atlas label order for the summary bars.",
    )
    parser.add_argument(
        "--exclude-network",
        nargs="+",
        type=str,
        default=None,
        help="One or more atlas network family names to exclude from the surface map and bar plot.",
    )
    parser.add_argument(
        "--surf-left",
        type=Path,
        default=_default_surface_path("fsaverage.L.inflated.32k_fs_LR.surf.gii"),
    )
    parser.add_argument(
        "--surf-right",
        type=Path,
        default=_default_surface_path("fsaverage.R.inflated.32k_fs_LR.surf.gii"),
    )
    parser.add_argument(
        "--sulc-left",
        type=Path,
        default=_default_surface_path("fsaverage.L.sulc.32k_fs_LR.surf.gii"),
    )
    parser.add_argument(
        "--sulc-right",
        type=Path,
        default=_default_surface_path("fsaverage.R.sulc.32k_fs_LR.surf.gii"),
    )
    parser.add_argument("--cmap", type=str, default="turbo")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument(
        "--auto-percentiles",
        nargs=2,
        type=float,
        metavar=("PLOW", "PHIGH"),
        default=(1.0, 99.0),
    )
    parser.add_argument(
        "--symmetric-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a symmetric color scale around zero.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Symmetric threshold for the FC difference map (default: 0.0). Values below the threshold will be set to NaN.",
    )
    parser.add_argument(
        "--exclude-medial-wall",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude medial wall vertices using the FS_LR no-medial-wall masks (default: true).",
    )
    parser.add_argument("--con-color", type=str, default="#5A189A")
    parser.add_argument("--surface-width", type=float, default=6.8)
    parser.add_argument("--surface-height", type=float, default=2.8)
    parser.add_argument("--figure-width", type=float, default=13.5)
    parser.add_argument("--figure-height", type=float, default=4.5)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--bar-width", type=float, default=0.72)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    scan_fc = _load_metric_pair(
        left_path=args.scan_left,
        right_path=args.scan_right,
        index=int(args.index),
        label="scan",
    )
    foot_fc = _load_metric_pair(
        left_path=args.foot_left,
        right_path=args.foot_right,
        index=int(args.index),
        label="foot",
    )
    hand_fc = _load_metric_pair(
        left_path=args.hand_left,
        right_path=args.hand_right,
        index=int(args.index),
        label="hand",
    )
    mouth_fc = _load_metric_pair(
        left_path=args.mouth_left,
        right_path=args.mouth_right,
        index=int(args.index),
        label="mouth",
    )
    if str(args.atlas) == "yeo":
        atlas_selection = _load_yeo_atlas()
    elif str(args.atlas) == "gordon":
        atlas_selection = _load_gordon_atlas()
    else:
        raise ValueError(f"Unsupported atlas selection: {args.atlas}")

    excluded_networks = {
        _normalize_network_name(name)
        for name in (cast(list[str] | None, args.exclude_network) or [])
        if _normalize_network_name(name)
    }
    excluded_label_ids = _excluded_label_ids(
        atlas_selection.label_lookup, excluded_networks
    )
    if atlas_selection.con_family_name in excluded_networks:
        raise ValueError(
            f"CON family {atlas_selection.con_family_name} cannot be excluded"
        )

    surf_left_mesh = surface.load_surf_mesh(str(args.surf_left))
    surf_right_mesh = surface.load_surf_mesh(str(args.surf_right))
    surf_left_sulc = np.asarray(
        surface.load_surf_data(str(args.sulc_left)), dtype=float
    )
    surf_right_sulc = np.asarray(
        surface.load_surf_data(str(args.sulc_right)), dtype=float
    )

    for label, pair in {
        "scan": scan_fc,
        "foot": foot_fc,
        "hand": hand_fc,
        "mouth": mouth_fc,
        "atlas": atlas_selection.atlas,
    }.items():
        validate_map_against_mesh(
            pair.left,
            surf_left_mesh,
            kind=f"{label} left",
            source_type="metric" if label != "atlas" else "overlay",
            source_format="gifti" if label != "atlas" else "cifti",
        )
        validate_map_against_mesh(
            pair.right,
            surf_right_mesh,
            kind=f"{label} right",
            source_type="metric" if label != "atlas" else "overlay",
            source_format="gifti" if label != "atlas" else "cifti",
        )

    diff_fc = _compute_difference(scan_fc, foot_fc, hand_fc, mouth_fc)
    diff_fc = _mask_excluded_networks(
        diff_fc, atlas_selection.atlas, excluded_label_ids
    )
    summaries = _build_network_summaries(
        atlas=atlas_selection.atlas,
        label_lookup=atlas_selection.label_lookup,
        medial_wall_mask=atlas_selection.medial_wall_mask,
        excluded_networks=excluded_networks,
        scan_fc=scan_fc,
        foot_fc=foot_fc,
        hand_fc=hand_fc,
        mouth_fc=mouth_fc,
        con_family_name=atlas_selection.con_family_name,
        network_order=cast(list[int] | None, args.network_order),
    )
    con_overlay = _build_con_overlay(
        atlas=atlas_selection.atlas,
        con_family_name=atlas_selection.con_family_name,
        label_lookup=atlas_selection.label_lookup,
        medial_wall_mask=atlas_selection.medial_wall_mask,
        con_color=str(args.con_color),
    )
    diff_fc_plot = (
        _mask_atlas_background(diff_fc, atlas_selection.medial_wall_mask)
        if bool(args.exclude_medial_wall)
        else diff_fc
    )

    if args.vmin is None or args.vmax is None:
        auto_vmin, auto_vmax = _compute_vrange(
            diff_fc_plot,
            percentiles=(
                float(args.auto_percentiles[0]),
                float(args.auto_percentiles[1]),
            ),
            symmetric=bool(args.symmetric_scale),
        )
    else:
        auto_vmin, auto_vmax = float(args.vmin), float(args.vmax)
    vmin = float(args.vmin) if args.vmin is not None else auto_vmin
    vmax = float(args.vmax) if args.vmax is not None else auto_vmax

    surface_image = _render_surface_image(
        diff_fc=diff_fc_plot,
        overlay=con_overlay,
        surf_left_mesh=surf_left_mesh,
        surf_right_mesh=surf_right_mesh,
        surf_left_sulc=surf_left_sulc,
        surf_right_sulc=surf_right_sulc,
        figure_size=(float(args.surface_width), float(args.surface_height)),
        dpi=int(args.dpi),
        cmap=str(args.cmap),
        vmin=vmin,
        vmax=vmax,
        threshold=float(args.threshold),
    )

    fig = plt.figure(
        figsize=(float(args.figure_width), float(args.figure_height)), dpi=int(args.dpi)
    )
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.5, 0.7], wspace=0.01)
    ax_surface = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    ax_surface.imshow(surface_image)
    ax_surface.set_axis_off()
    _plot_summary_bars(
        ax=ax_bar,
        summaries=summaries,
        con_family_name=atlas_selection.con_family_name,
        bar_width=float(args.bar_width),
    )

    output = (
        args.output if args.output is not None else _derive_output_path(args.scan_left)
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

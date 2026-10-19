"""
Plotting utilities for violin plot of vertices values by network label, and for plotting the network summary figure.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, Literal

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from nibabel.gifti.gifti import GiftiImage
from nibabel.cifti2.cifti2 import Cifti2Image
from nibabel.loadsave import load as nib_load


@dataclass(frozen=True)
class PairData:
    left: np.ndarray
    right: np.ndarray


@dataclass(frozen=True)
class NetworkSummary:
    index: int
    name: str
    color: tuple[float, float, float, float]
    values: np.ndarray


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


def _load_medial_wall_mask_pair() -> PairData:
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
    return PairData(
        left=np.asarray(left_img.darrays[0].data, dtype=float).ravel(),
        right=np.asarray(right_img.darrays[0].data, dtype=float).ravel(),
    )


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


def _load_yeo_atlas() -> AtlasSelection:
    root = Path(__file__).resolve().parent.parent.parent
    atlas, label_lookup = _load_cifti_atlas(
        root / "template" / "Yeo2011_7Networks.split_components.dlabel.nii"
    )
    return AtlasSelection(
        atlas=atlas,
        label_lookup=label_lookup,
        medial_wall_mask=_load_medial_wall_mask_pair(),
        con_family_name="Cont",
    )


def _load_gordon_atlas() -> AtlasSelection:
    root = Path(__file__).resolve().parent.parent.parent
    atlas, label_lookup = _load_cifti_atlas(
        root / "template" / "Gordon333_FreesurferSubcortical.32k_fs_LR.dlabel.nii"
    )
    return AtlasSelection(
        atlas=atlas,
        label_lookup=label_lookup,
        medial_wall_mask=_load_medial_wall_mask_pair(),
        con_family_name="CinguloOperc",
    )


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
            except Exception:
                continue
            if len(rgba4) != 4:
                continue
            if _is_ignored_atlas_label_name(str(name)):
                continue
            entries.append((int(key), str(name), _normalize_rgba(cast(Any, rgba4))))
        entries.sort(key=lambda item: item[0])
        return entries
    except Exception:
        return []


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


def _build_single_map_summaries(
    *,
    atlas: PairData,
    label_lookup: dict[int, AtlasLabelInfo],
    medial_wall_mask: PairData,
    input_fc: PairData,
    con_family_name: str,
    network_order: list[int] | None,
    excluded_networks: set[str],
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

        values = np.concatenate(
            (input_fc.left[left_mask], input_fc.right[right_mask]), axis=0
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
                values=values,
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


def _plot_summary_bars(
    *,
    ax: Axes,
    summaries: list[NetworkSummary],
    con_family_name: str,
    max_width: float,
    tick_label_fontsize: float = 10.0,
) -> None:
    positions = np.arange(len(summaries), dtype=float)
    colors = [summary.color for summary in summaries]
    values = [summary.values for summary in summaries]

    violin_parts = ax.violinplot(
        values,
        positions=positions,
        widths=max_width,
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    violin_parts["cmedians"].set_color("black")
    violin_parts["cmedians"].set_linestyle("--")
    violin_parts["cmedians"].set_alpha(0.6)

    violin_parts["cbars"].set_color("black")
    violin_parts["cbars"].set_alpha(0.6)
    violin_parts["cmins"].set_color("black")
    violin_parts["cmins"].set_alpha(0.6)
    violin_parts["cmaxes"].set_color("black")
    violin_parts["cmaxes"].set_alpha(0.6)

    for i, pc in enumerate(violin_parts["bodies"]):  # type: ignore[union-attr]
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [summary.name for summary in summaries],
        rotation=50,
        ha="right",
        fontsize=tick_label_fontsize,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.02)


def plot_network_summary(
    ax: Axes,
    input_left: Path,
    input_right: Path,
    atlas: Literal["gordon", "yeo"],
    atlas_label_order: list[int] | None = None,
    exclude_networks: list[str] | None = None,
    frame_index: int = 0,
    max_width: float = 0.5,
    tick_label_fontsize: float = 10.0,
):
    """
    Build a bar plot summarizing the mean values of the input metric for each network in the specified atlas.

    Parameters
    ----------
    ax: Axes
        The matplotlib Axes object where the summary plot will be drawn.
    input_left : Path
        Path to the left hemisphere GIFTI file containing the metric data.
    input_right : Path
        Path to the right hemisphere GIFTI file containing the metric data.
    atlas : Literal['gordon', 'yeo']
        The atlas to use for network labeling. Must be either 'gordon' or 'yeo'.
    atlas_label_order : list[int] | None, optional
        A list specifying the order of network labels to display in the plot. If None, the default order is used.
    exclude_networks : list[str] | None, optional
        A list of network family names to exclude from the plot. If None, no networks are excluded.
    frame_index : int, optional
        The index of the frame to extract from the GIFTI files. Default is 0.
    max_width : float, optional
        The maximum width of the violins in the plot. Default is 0.5.
    """

    input_pair = _load_metric_pair(
        left_path=input_left,
        right_path=input_right,
        index=frame_index,
        label="input",
    )

    if atlas == "yeo":
        atlas_selection = _load_yeo_atlas()
    elif atlas == "gordon":
        atlas_selection = _load_gordon_atlas()
    else:
        raise ValueError(f"Unsupported atlas selection: {atlas}")

    excluded_networks = {
        _normalize_network_name(name)
        for name in (exclude_networks or [])
        if _normalize_network_name(name)
    }
    excluded_label_ids = _excluded_label_ids(
        atlas_selection.label_lookup, excluded_networks
    )
    if atlas_selection.con_family_name in excluded_networks:
        raise ValueError(
            f"CON family {atlas_selection.con_family_name} cannot be excluded"
        )

    input_pair = _mask_excluded_networks(
        input_pair, atlas_selection.atlas, excluded_label_ids
    )
    summaries = _build_single_map_summaries(
        atlas=atlas_selection.atlas,
        label_lookup=atlas_selection.label_lookup,
        medial_wall_mask=atlas_selection.medial_wall_mask,
        input_fc=input_pair,
        con_family_name=atlas_selection.con_family_name,
        network_order=atlas_label_order,
        excluded_networks=excluded_networks,
    )

    _plot_summary_bars(
        ax=ax,
        summaries=summaries,
        con_family_name=atlas_selection.con_family_name,
        max_width=max_width,
        tick_label_fontsize=tick_label_fontsize,
    )

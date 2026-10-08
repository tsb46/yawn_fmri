"""
Module for writing analysis results to func.gii.
"""

import os
from typing import Literal

import nibabel as nb
import numpy as np
from nibabel.gifti.gifti import (
    GiftiDataArray,
    GiftiImage,
    GiftiLabel,
    GiftiLabelTable,
)

from scan.io.load import Gifti


def write_func_gii(data: np.ndarray, gii_params: Gifti, fp_out: str) -> None:
    """
    Write out functional data in 2D format (# of time points, # of vertices)
    to a func.gii file. Use parameters in Gifti class (gii_params) to write
    out in consistent format.
    """
    # split data into left and right hemispheres
    data_lh, data_rh = _separate_gii_hemispheres(data, gii_params.split_indx)
    # Create new GiftiDataArrays for the left and right hemispheres
    gii_lh = GiftiImage()
    gii_rh = GiftiImage()
    for row_i in range(data_lh.shape[0]):
        gii_data_array_lh = GiftiDataArray(data=data_lh[row_i, :], datatype=16)
        gii_data_array_rh = GiftiDataArray(data=data_rh[row_i, :], datatype=16)
        gii_lh.add_gifti_data_array(gii_data_array_lh)
        gii_rh.add_gifti_data_array(gii_data_array_rh)

    # Save the new GIFTI files
    nb.save(gii_lh, f"{fp_out}_lh.func.gii")  # type: ignore
    nb.save(gii_rh, f"{fp_out}_rh.func.gii")  # type: ignore

    # set structure as left or right cortex to view in connectome workbench
    _set_structure(fp_out, "func")


def write_label_gii(data: np.ndarray, gii_params: Gifti, fp_out: str) -> None:
    """
    Write out label data in 1D format (# of vertices) to a label.gii file.
    Use parameters in Gifti class (gii_params) to write out in consistent
    format.
    """
    # split data into left and right hemispheres
    data_lh, data_rh = _separate_gii_hemispheres(data, gii_params.split_indx)

    # squeeze data to 1D array
    data_lh = data_lh.squeeze()
    data_rh = data_rh.squeeze()

    # get unique labels
    unique_labels = np.unique(data)

    # generate unique RGB colors for each label
    unique_colors = _generate_unique_rgb_colors(len(unique_labels))

    # create label table
    label_table = GiftiLabelTable()
    for label_i, label in enumerate(unique_labels):
        # associate with unique r, g, b values
        r, g, b = unique_colors[label_i]
        gifti_label = GiftiLabel(key=label, red=r, green=g, blue=b)
        gifti_label.label = str(label)  # type: ignore
        label_table.labels.append(gifti_label)

    # Create new GiftiDataArrays for the left and right hemispheres
    gii_lh = GiftiImage(labeltable=label_table)
    gii_rh = GiftiImage(labeltable=label_table)
    gii_lh.add_gifti_data_array(
        GiftiDataArray(data=data_lh, datatype=16, intent="NIFTI_INTENT_LABEL")
    )
    gii_rh.add_gifti_data_array(
        GiftiDataArray(data=data_rh, datatype=16, intent="NIFTI_INTENT_LABEL")
    )

    # Save the new GIFTI files
    nb.save(gii_lh, f"{fp_out}_lh.label.gii")  # type: ignore
    nb.save(gii_rh, f"{fp_out}_rh.label.gii")  # type: ignore

    # set structure as left or right cortex to view in connectome workbench
    _set_structure(fp_out, "label")


def _separate_gii_hemispheres(
    data: np.ndarray,
    split_indx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Separate data into left and right hemispheres. Return as separate arrays
    with left hemisphere first and right hemisphere second.
    """
    data_lh = data[:, :split_indx]
    data_rh = data[:, split_indx:]
    return data_lh, data_rh


def _generate_unique_rgb_colors(n_colors: int) -> np.ndarray:
    """
    Generate unique RGB colors for labels.

    Parameters
    ----------
    n_colors : int
        Number of unique colors to generate

    Returns
    -------
    np.ndarray
        Array of shape (n_colors, 3) with RGB values between 0 and 1
    """
    if n_colors <= 0:
        return np.array([])

    # For small number of colors, use predefined distinct colors
    if n_colors <= 12:
        # Use distinct colors that are visually distinguishable
        distinct_colors = np.array(
            [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 0.0],  # Yellow
                [1.0, 0.0, 1.0],  # Magenta
                [0.0, 1.0, 1.0],  # Cyan
                [1.0, 0.5, 0.0],  # Orange
                [0.5, 0.0, 1.0],  # Purple
                [0.0, 0.5, 0.0],  # Dark Green
                [0.5, 0.5, 0.0],  # Olive
                [0.5, 0.0, 0.5],  # Dark Magenta
                [0.0, 0.5, 0.5],  # Teal
            ]
        )
        return distinct_colors[:n_colors]

    # For larger numbers, generate colors using golden ratio method
    # This ensures good distribution in color space
    colors = np.zeros((n_colors, 3))
    golden_ratio = 0.618033988749895

    for i in range(n_colors):
        hue = (i * golden_ratio) % 1.0
        # Convert HSV to RGB (simplified version)
        h = hue * 6
        c = 1.0
        x = c * (1 - abs(h % 2 - 1))
        m = 0.3  # Minimum brightness

        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        colors[i] = [r + m, g + m, b + m]

    return colors


def _set_structure(fp_out: str, type: Literal["func", "label"]) -> None:
    """
    Set structure as left or right cortex to view in connectome workbench
    """
    os.system(f"""
        wb_command -set-structure {fp_out}_lh.{type}.gii CORTEX_LEFT
    """)
    os.system(f"""
        wb_command -set-structure {fp_out}_rh.{type}.gii CORTEX_RIGHT
    """)

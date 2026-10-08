"""
Custom preprocessing functions
"""

import nibabel as nb
import numpy as np
from nibabel.cifti2 import Cifti2Header  # type: ignore
from nibabel.cifti2.cifti2_axes import SeriesAxis


def framewise_displacement(
    motion_params: dict[str, np.ndarray], radius: float = 50.0
) -> np.ndarray:
    """
    Calculate framewise displacement from motion parameters

    Parameters
    ----------
    motion_params : dict[str, np.ndarray]
        Dictionary of motion parameters:
            {
                'pitch': np.ndarray,
                'roll': np.ndarray,
                'yaw': np.ndarray,
                'trans_x': np.ndarray,
                'trans_y': np.ndarray,
                'trans_z': np.ndarray
            }

    Returns
    -------
    framewise_displacement : np.ndarray
        Framewise displacement (length = timepoints)
    """

    # place motion parameters into ndarray
    mpars = np.zeros((len(motion_params["trans_x"]), 6))
    mpars[:, 0] = motion_params["trans_x"]
    mpars[:, 1] = motion_params["trans_y"]
    mpars[:, 2] = motion_params["trans_z"]
    mpars[:, 3] = motion_params["pitch"]
    mpars[:, 4] = motion_params["roll"]
    mpars[:, 5] = motion_params["yaw"]

    # Calculate framewise displacement. Prepend a leading zero so the
    # returned series keeps the same length as the input motion trace.
    diff = mpars[:-1, :6] - mpars[1:, :6]
    diff[:, 3:6] *= radius
    fd = np.abs(diff).sum(axis=1)
    fd = np.concatenate(([0.0], fd))
    return fd


def trim_cifti(fp: str, fp_out: str, n_trim: int) -> None:
    """
    Trim the first N timepoints of a CIFTI-2 file. The CIFTI-2 file
    is assumed to have a SeriesAxis as the first axis, and a BrainModelAxis
    as the second axis.

    Parameters
    ----------
    fp : str
        Path to input CIFTI file
    fp_out : str
        Path to output CIFTI file
    n_trim : int
        Number of timepoints to trim from the beginning
    """
    # Load the CIFTI file
    img = nb.load(fp)  # type: ignore

    # Get data and header information
    data = img.get_fdata()  # type: ignore

    # Trim the first n_trim timepoints, assume time is the first dimension
    trimmed_data = data[n_trim:, ...]

    # Create a new image with the trimmed data
    # We need to update the header to reflect the reduced number of timepoints
    header = img.header.copy()

    # Update the axes information, assume time is the first dimension
    old_size = header.get_axis(0).size  # type: ignore
    new_size = old_size - n_trim

    # Create a new Series axis with updated size
    old_series_axis = header.get_axis(0)  # type: ignore
    new_series_axis = SeriesAxis(
        start=old_series_axis.start,
        step=old_series_axis.step,
        size=new_size,
        unit=old_series_axis.unit,
    )

    # Create a new header with the updated axis
    new_header = Cifti2Header.from_axes(
        axes=[new_series_axis, header.get_axis(1)]  # type: ignore
    )

    # Create the new image with updated header
    new_img = nb.Cifti2Image(trimmed_data, new_header, img.nifti_header)  # type: ignore

    # Save the new image
    new_img.to_filename(fp_out)

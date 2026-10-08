"""FSL (V6.0.5) command line utilities"""

import os
from typing import List

import nibabel as nb
import numpy as np

from nipype.interfaces import fsl

from scan import utils

# Ensure output is .nii.gz
fsl.FSLCommand.set_default_output_type("NIFTI_GZ")


def bet(fp: str, fp_out: str, frac: float | None = None) -> None:
    """
    BET - Skullstrip anatomical Image.

    Parameters
    ----------
        frac : float
            fractional intensity threshold
    """
    fslbet = fsl.BET(frac=frac, robust=True, mask=True)
    fslbet.inputs.in_file = fp
    fslbet.inputs.out_file = fp_out
    fslbet.run()


def load_fsl_motion_params(fp: str) -> dict[str, np.ndarray]:
    """
    Load motion parameters from .par motion file output by FSL
    McFlirt motion correction.

    Parameters
    ----------
        fp: filepath to motion file

    Returns
    -------
        dict of np.ndarray motion parameters:
            {
                'pitch': np.ndarray,
                'roll': np.ndarray,
                'yaw': np.ndarray,
                'trans_x': np.ndarray,
                'trans_y': np.ndarray,
                'trans_z': np.ndarray
            }

    FSL saves motion parameters in the following order:
        rx  Pitch  (rad)
        ry  Roll   (rad)
        rz  Yaw    (rad)
        x, y, z    (mm)

    Values are the transform that realigns each volume to the reference
    (the inverse of head motion). Verified with simulated head movements
    on LAS-stored data (voxel axes point Left, Anterior, Superior); positive
    values correspond to this head motion:
        pitch   nose up (chin up)
        roll    top of head tilts to the left
        yaw     nose turns to the right
        trans_x head moves right
        trans_y head moves posterior
        trans_z head moves inferior
    Signs may differ for other image orientations.

    """
    motion = np.loadtxt(fp)
    motion_params = {
        "pitch": motion[:, 0],
        "roll": motion[:, 1],
        "yaw": motion[:, 2],
        "trans_x": motion[:, 3],
        "trans_y": motion[:, 4],
        "trans_z": motion[:, 5],
    }
    return motion_params


def mcflirt(fp: str, fp_out: str) -> None:
    """
    McFLIRT Motion Correction

    Parameters
    ----------
        fp: str
            filepath to functional volume
        fp_out: str
            output file path for functional mean volume

    """
    fslmcflirt = fsl.MCFLIRT(mean_vol=True, save_plots=True)
    # remove extension for proper output file renaming
    fp_out_base = utils.get_fp_base(fp_out)
    fslmcflirt.inputs.in_file = fp
    fslmcflirt.inputs.out_file = fp_out_base
    fslmcflirt.inputs.save_mats = True
    fslmcflirt.inputs.save_plots = True
    # Mcflirt raises FileNotFoundError after successful completion, ignore
    try:
        fslmcflirt.run()
    except FileNotFoundError:
        pass


def mcflirt_multiecho(
    fps: List[str], fps_out: List[str], fp_meanvol: str, fp_mat: str
) -> None:
    """
    McFLIRT Motion Correction for multiple echos. Apply motion
    correction to first echo and apply registration to subsequent
    echos

    Parameters
    ----------
        fps : List[str]
            lists of filepaths to each echo in order
        fp_mean_vol: str
            output file path for functional mean volume
        fp_mat: str
            output file path for mcflirt transformation matrix

    """
    # apply MCFLIRT motion correction to first echo
    mcflirt(fps[0], fps_out[0])
    # rename mats file

    # apply mcflirt transform to rest of echos
    for fp, fp_out in zip(fps[1:], fps_out[1:]):
        # applyxfm4d is not available in nipype, run from terminal
        os.system(f"applyxfm4D {fp} {fp_meanvol} {fp_out} {fp_mat} -fourdigit")


def slicetime(fp: str, fp_out: str, slice_order: str, tr: float) -> None:
    """
    Slice time correction with SliceTimer

    Parameters
    ----------
        slice_order: str
            filepath to slicetime ordering .txt file.
            File should contain the order of each slice
            in descending order (index starts at 1).
        tr: float
            repetition time

    """
    slicetimer = fsl.SliceTimer(custom_order=slice_order, time_repetition=tr)
    slicetimer.inputs.in_file = fp
    slicetimer.inputs.out_file = fp_out
    slicetimer.run()


def slicetime_multiecho(
    fps: List[str], fps_out: str, slice_order: str, tr: float
) -> None:
    """
    Slice time correction with SliceTimer for multiecho datasets

    Parameters
    ----------
        fps : List[str]
            lists of filepaths to each echo in order
        slice_order: str
            filepath to slicetime ordering .txt file.
            File should contain the order of each slice
            in descending order (index starts at 1).
        tr: float
            repetition time
    """
    for fp, fp_out in zip(fps, fps_out):
        slicetime(fp, fp_out, slice_order, tr)


def trim_vol(fp: str, fp_out: str, n_trim: int) -> None:
    """
    Trim first (+) or last (-) N volumes with ExtractROI

    Parameters
    ----------
        n_trim : int
            number of volumes to trim. If the integer, n,
            provided is positive, trim off the first n volumes.
            If negative, trim off the last n volumes.
    """
    if n_trim >= 0:
        trim = fsl.ExtractROI(t_min=n_trim, t_size=-1)
    else:
        n_end = nb.load(fp).shape[-1]  # type: ignore
        n_end -= abs(n_trim)
        trim = fsl.ExtractROI(t_min=0, t_size=n_end)
    trim.inputs.in_file = fp
    trim.inputs.roi_file = fp_out
    trim.run()


def trim_vol_multiecho(fps: List[str], fps_out: List[str], n_trim: int) -> None:
    """
    Trim first (+) or last (-) N volumes with ExtractROI for mulitecho
    datasets

    Parameters
    ----------
        fps : List[str]
            lists of filepaths to each echo in order
        n_trim : int
            number of volumes to trim. If the integer, n,
            provided is positive, trim off the first n volumes.
            If negative, trim off the last n volumes.
    """
    # loop through echos
    for fp, fp_out in zip(fps, fps_out):
        trim_vol(fp, fp_out, n_trim)

"""
utils for loading and writing data
"""

from typing import List, Literal

import numpy as np
from nilearn.signal import butterworth
from scipy.stats import zscore


def check_roi_masks(lh_rois: List[np.ndarray], rh_rois: List[np.ndarray]) -> None:
    """
    Check that left and right hemisphere ROI masks are valid.
    """
    # ensure only 2 unique values
    check_two_unique = lambda x: np.unique(x).size == 2
    # ensure only 1s and 0s in roi masks
    check_one_zero = lambda x: (np.unique(x) == [0, 1]).all()
    if not all(check_two_unique(roi_mask) for roi_mask in lh_rois):
        raise ValueError("lh roi_masks must only contain 1s and 0s")
    if not all(check_two_unique(roi_mask) for roi_mask in rh_rois):
        raise ValueError("rh roi_masks must only contain 1s and 0s")
    if not all(check_one_zero(roi_mask) for roi_mask in lh_rois):
        raise ValueError("lh roi_masks must only contain 1s and 0s")
    if not all(check_one_zero(roi_mask) for roi_mask in rh_rois):
        raise ValueError("rh roi_masks must only contain 1s and 0s")


def filter(
    signals: np.ndarray, low_pass: bool, high_pass: bool, tr: float
) -> np.ndarray:
    """
    perform low- (< 0.15), high- (>0.01) or band-pass filtering
    of time courses with nilearn.signal.butterworth (5th order butterworth
    filter with padding of 100 samples). If no filtering
    is specified, return original signal

    Parameters
    ----------
    signals: np.ndarray
        time courses to filter
    low_pass: bool
        whether to perform low-pass filtering
    high_pass: bool
        whether to perform high-pass filtering
    tr: float
        repetition time of time courses

    Returns
    -------
    signals: np.ndarray
        filtered time courses
    """
    if low_pass or high_pass:
        # specify low- and high-pass cutoffs
        if high_pass:
            highpass = 0.01
        else:
            highpass = None

        if low_pass:
            lowpass = 0.15
        else:
            lowpass = None

        # get sampling frequency
        sf = 1 / tr
        # perform signal filtering
        signals = butterworth(
            signals, sampling_rate=sf, low_pass=lowpass, high_pass=highpass, padlen=100
        )

    return signals


def norm(
    signals: np.ndarray, norm: Literal["zscore", "demean", "robust_z"] | None = "zscore"
) -> np.ndarray:
    """
    normalize signals, either with zscoring, mean centering, or robust z-scoring. If no
    normalization is specified, return original signal. 'Robust-z' normalization is performed
    by subtracting the median and dividing by the median absolute deviation (MAD).
    """
    if norm == "zscore":
        # zscore along temporal dimension
        signals = zscore(signals, axis=0)
    elif norm == "demean":
        signals -= np.mean(signals, axis=0)
    elif norm == "robust_z":
        abs_dev = np.abs(signals - np.median(signals, axis=0))
        mad = np.median(abs_dev, axis=0)
        signals = (signals - np.median(signals, axis=0)) / mad
    return signals

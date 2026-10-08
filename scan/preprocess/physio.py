"""
Utilities for extracting features from raw physio signals
"""

from typing import Tuple

import neurokit2 as nk
import numpy as np
import scipy
from neurokit2.rsp.rsp_rvt import _rsp_rvt_find_min
from scipy.ndimage import gaussian_filter1d

from scan.preprocess.custom import framewise_displacement


def extract_eog_features(ts: np.ndarray, sf: int) -> dict[str, np.ndarray]:
    """
    Extract EOG features from raw EOG signal

    Parameters
    ----------
    ts: np.ndarray
        time series of raw electrooculography (eog) signal
    sf: int
        sampling frequency

    Returns
    -------
    eog_features: dict[str, np.ndarray]
        extracted EOG features
    """
    # bandpass filters
    ts_hf = nk.signal.signal_filter(ts, sampling_rate=sf, lowcut=20, highcut=100)

    # Hilbert amplitude
    ts_hf_amp = np.abs(scipy.signal.hilbert(ts_hf))

    # smoothing
    ts_hf_amp = _smooth(ts_hf_amp, sf, sigma_sec=0.05)

    # normalization
    ts_hf_amp = _robust_z(ts_hf_amp)

    return {"eog_hf_amp": ts_hf_amp}


def extract_emg_features(ts: np.ndarray, sf: int) -> dict[str, np.ndarray]:
    """
    Extract EMG features from raw EMG signal

    Parameters
    ----------
    ts: np.ndarray
        time series of raw electromyography (emg) signal
    sf: int
        sampling frequency

    Returns
    -------
    emg_features: dict[str, np.ndarray]
        extracted EMG features - amplitude, slope, and peak
    """
    ts_filt = nk.signal.signal_filter(ts, sampling_rate=sf, lowcut=20, highcut=100)

    ts_amp = np.abs(scipy.signal.hilbert(ts_filt))

    # smoothing
    ts_amp = _smooth(ts_amp, sf, sigma_sec=0.05)

    # normalization
    ts_amp = _robust_z(ts_amp)

    return {"emg_hf_amp": ts_amp}


def extract_motion_features(
    motion_params: dict[str, np.ndarray], sf: int | None = None
) -> dict[str, np.ndarray]:
    """
    Extract motion parameters from motion parameters dictionary

    Parameters
    ----------
    motion_params: dict[str, np.ndarray]
        motion parameters dictionary
    sf: float
        sampling frequency (unused, included for consistency with other physio functions)

    Returns
    -------
    motion_params_extract: dict[str, np.ndarray]
        motion parameters
    """

    # detrend motion parameters
    for key, signal in motion_params.items():
        motion_params[key] = nk.signal.signal_detrend(signal, order=3)

    fd = framewise_displacement(motion_params)

    return {
        "fd": fd,
        "pitch": np.rad2deg(motion_params["pitch"]),
        "roll": np.rad2deg(motion_params["roll"]),
        "yaw": np.rad2deg(motion_params["yaw"]),
        "trans_x": motion_params["trans_x"],
        "trans_z": motion_params["trans_z"],
        "trans_y": motion_params["trans_y"],
    }


def extract_resp_features(ts: np.ndarray, sf: int) -> dict[str, np.ndarray]:
    """
    Extract respiratory features via the method of Harrison et al. (2021)
    https://doi.org/10.1016/j.neuroimage.2021.117787.

    Parameters
    ----------
    ts: np.ndarray
        time series of raw respiratory signal
    sf: float
        sampling frequency

    Returns
    -------
    resp_features: dict[str, np.ndarray]
        respiratory amplitude, rate, instantaneous frequency signals
    """
    ts_clean = nk.rsp.rsp_clean(ts, sampling_rate=sf)

    rsp_amp, _, rsp_if = rsp_rvt_harrison(np.asarray(ts_clean), sf)

    # normalize features
    rsp_amp = _robust_z(rsp_amp)

    return {
        "resp_filt": np.asarray(ts_clean),
        "resp_amp": rsp_amp,
        "resp_if": rsp_if,
    }


def extract_sample_weight(ts: np.ndarray, sf: int) -> dict[str, np.ndarray]:
    """
    Return sample weights for weighting of individual time points in later
    regression analyses. This is needed due to known drop-out issues in some
    recordings (e.g. vanderbilt respiratory recordings). This function
    performs no transformation on the signal, but simply returns the input signal as a dictionary
    for consistency with the API.

    Parameters
    ----------
    ts: np.ndarray
        time series of sample weights
    sf: float
        sampling frequency

    Returns
    -------
    ts: dict[str, np.ndarray]
        sample weight signal
    """
    return {"weight": ts}


def rsp_rvt_harrison(
    rsp_signal: np.ndarray,
    sf: int,
    boundaries: Tuple[float, float] = (2.0, 1 / 30),
    iterations: int = 10,
    silent: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slight modification of the NeuroKit2 (v0.2.11) RVT function to return
    the amplitude and phase of the respiratory signal.

    Note: This function is not part of the NeuroKit2 API and is not
    guaranteed to be stable.
    https://github.com/neuropsychology/NeuroKit/blob/v0.2.11/neurokit2/rsp/rsp_rvt.py#L235

    Parameters
    ----------
    rsp_signal: np.ndarray
        respiratory signal
    sf: int
        sampling frequency
    boundaries: Tuple[float, float]
        boundaries for breathing rate
    iterations: int
        number of iterations

    Returns
    -------
    rvt: np.ndarray
        respiratory volume per time
    phase: np.ndarray
        respiratory phase
    ifreq: np.ndarray
        instantaneous frequency
    """
    # low-pass filter at not too far above breathing-rate to remove high-frequency noise
    n_pad = int(np.ceil(10 * sf))

    d = scipy.signal.iirfilter(
        N=10, Wn=0.75, btype="lowpass", analog=False, output="sos", fs=sf
    )
    fr_lp = scipy.signal.sosfiltfilt(d, np.pad(rsp_signal, n_pad, "symmetric"))
    fr_lp = fr_lp[n_pad : (len(fr_lp) - n_pad)]

    # derive Hilbert-transform
    fr_filt = fr_lp
    fr_mag = abs(scipy.signal.hilbert(fr_filt))

    # initialize phase as the angle of the analytic signal
    fr_phase = np.unwrap(np.angle(scipy.signal.hilbert(fr_filt)))

    for _ in range(iterations):
        # analytic signal to phase
        fr_phase = np.unwrap(np.angle(scipy.signal.hilbert(fr_filt)))
        # Remove any phase decreases that may occur
        # Find places where the gradient changes sign
        # maybe can be changed with signal.signal_zerocrossings
        fr_phase_diff = np.diff(np.sign(np.gradient(fr_phase)))
        decrease_inds = np.argwhere(fr_phase_diff < 0)
        increase_inds = np.append(np.argwhere(fr_phase_diff > 0), [len(fr_phase) - 1])
        for n_max in decrease_inds:
            # Find value of `fr_phase` at max and min:
            fr_max = fr_phase[n_max].squeeze()
            n_min, fr_min = _rsp_rvt_find_min(increase_inds, fr_phase, n_max, silent)

            if n_min is None:
                # There is no finishing point to the interpolation at the very end
                continue
            # Find where `fr_phase` passes `fr_min` for the first time
            n_start = np.argwhere(fr_phase > fr_min)
            if len(n_start) == 0:
                n_start = n_max
            else:
                n_start = n_start[0].squeeze()
            # Find where `fr_phase` exceeds `fr_max` for the first time
            n_end = np.argwhere(fr_phase < fr_max)
            if len(n_end) == 0:
                n_end = n_min
            else:
                n_end = n_end[-1].squeeze()

            # Linearly interpolate from n_start to n_end
            fr_phase[n_start:n_end] = np.linspace(
                fr_min,  # type: ignore
                fr_max,
                num=n_end - n_start,  # type: ignore
            ).squeeze()
        # Filter out any high frequencies from phase-only signal
        fr_filt = scipy.signal.sosfiltfilt(
            d, np.pad(np.cos(fr_phase), n_pad, "symmetric")
        )
        fr_filt = fr_filt[n_pad : (len(fr_filt) - n_pad)]
    # Keep phase only signal as reference
    fr_filt = np.cos(fr_phase)

    # Make RVT

    # Low-pass filter to remove within_cycle changes
    # Note factor of two is for compatability with the common definition of RV
    # as the difference between max and min inhalation (i.e. twice the amplitude)
    d = scipy.signal.iirfilter(
        N=10,
        Wn=0.2,
        btype="lowpass",
        analog=False,
        output="sos",
        fs=sf,
    )
    fr_rv = 2 * scipy.signal.sosfiltfilt(d, np.pad(fr_mag, n_pad, "symmetric"))
    fr_rv = fr_rv[n_pad : (len(fr_rv) - n_pad)]
    fr_rv[fr_rv < 0] = 0

    # Breathing rate is instantaneous frequency
    fr_if = sf * np.gradient(fr_phase) / (2 * np.pi)
    fr_if = scipy.signal.sosfiltfilt(d, np.pad(fr_if, n_pad, "symmetric"))
    fr_if = fr_if[n_pad : (len(fr_if) - n_pad)]
    # remove in-human patterns, since both limits are in Hertz, the upper_limit is lower
    fr_if = np.clip(fr_if, boundaries[1], boundaries[0])

    return fr_rv, fr_phase, fr_if


def _robust_z(x: np.ndarray) -> np.ndarray:
    """
    Compute robust z-score of a signal using median and median absolute deviation.
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-8
    return (x - med) / mad


def _smooth(x: np.ndarray, sf: int, sigma_sec: float) -> np.ndarray:
    """
    Smooth a signal using a Gaussian filter.
    """
    sigma = sigma_sec * sf
    return gaussian_filter1d(x, sigma=sigma)

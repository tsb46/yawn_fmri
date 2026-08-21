"""
Utilities for extracting features from raw physio signals
"""

from typing import Tuple

import mne
import neurokit2 as nk
import numpy as np
import scipy

from neurokit2.rsp.rsp_rvt import _rsp_rvt_find_min

from scan.preprocess.custom import framewise_displacement


def extract_eog_blink(ts: np.ndarray, sf: int) -> dict[str, np.ndarray]:
    """
    Extract blink rate from eog signals

    Parameters
    ----------
    ts: np.ndarray
        time series of raw electrooculography (eog) signal
    sf: int
        sampling frequency

    Returns
    -------
    eog_blink: dict[str, np.ndarray]
        eog blink rate
    """
    # resp amplitude label
    eog_blink = "EOG_Rate"
    # extract respiration amplitude and frequency
    eog_signals, _ = np.asarray(nk.eog_process(ts, sampling_rate=sf))
    return {"eog_blink": eog_signals[eog_blink].values}


def extract_emg_amplitude(ts: np.ndarray, sf: int) -> dict[str, np.ndarray]:
    """
    Extract electromyography or electrocorticography amplitude signals

    Parameters
    ----------
    ts: np.ndarray
        time series of raw emg signal
    sf: int
        sampling frequency

    Returns
    -------
    emg_amp: dict[str, np.ndarray]
        emg amplitude signal
    """
    # extract emg amplitude
    ts_filt = nk.signal_filter(ts, sampling_rate=sf, lowcut=10, highcut=30)
    ts_complex = scipy.signal.hilbert(ts_filt)
    ts_amp = np.abs(ts_complex)

    return {"emg_amp": ts_amp}


def extract_motion(
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
    # extract framewise displacement
    fd = framewise_displacement(motion_params)
    # extract motion parameters most relevant to breathing behaviors
    motion_params_extract = {
        "fd": fd,
        "pitch": np.rad2deg(motion_params["pitch"]),
        "trans_z": motion_params["trans_z"],
        "trans_y": motion_params["trans_y"],
    }
    return motion_params_extract


def extract_resp_rvt(ts: np.ndarray, sf: int) -> dict[str, np.ndarray]:
    """
    Extract respiratory amplitude and rate by method of Harrison et al. (2021)
    https://doi.org/10.1016/j.neuroimage.2021.117787

    Parameters
    ----------
    ts: np.ndarray
        time series of raw respiratory signal
    sf: float
        sampling frequency

    Returns
    -------
    resp_amp: dict[str, np.ndarray]
        respiratory amplitude and rate signals
    """
    # Clean raw respiratory signal
    ts_clean = nk.rsp_clean(
        ts,
        sampling_rate=sf,
    )
    # extract respiration amplitude and frequency
    rvt, phase = rsp_rvt_harrison(np.asarray(ts_clean), sf)
    return {
        "resp_amp": rvt,
        "resp_rate": phase,
    }


def extract_sample_weight(ts: np.ndarray, sf: int) -> dict[str, np.ndarray]:
    """
    Return sample weights for weighting of individual time points in later
    regression analyses. This is needed due to known drop-out issues in some
    recordings (e.g. vanderbilt respiratory recordings). This function
    performs no transformation on the signal (i.e. an identity transform)
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
        respiratory amplitude signal
    """
    return {"weight": ts}


def extract_eeg_vigilance(
    eeg_data: np.ndarray, sf_eeg: int, window_sec: float = 2
) -> dict[str, np.ndarray]:
    """
    Window-based computation of vigilance index from eeg data. Computed as
    the ratio of the power in the Alpha (8-12 Hz) band to the power in the
    Theta (4-7 Hz) band. Returns alpha and theta power with vigilance signal.

    Parameters
    ----------
        eeg_data: np.ndarray
            eeg data (time x channel)
        sf_eeg: int
            sampling frequency of eeg data
        window_sec: float
            window size in seconds

    Returns
    -------
    vigilance: dict[str, np.ndarray]
        vigilance index (ratio of alpha and theta), alpha and theta power
    """
    # define alpha and theta bands
    alpha_band = (8, 12)
    theta_band = (4, 7)
    # compute power in alpha and theta bands
    alpha_power = _wavelet_power(
        eeg_data,
        sf_eeg,
        alpha_band,
    )
    theta_power = _wavelet_power(
        eeg_data,
        sf_eeg,
        theta_band,
    )
    # average across channels
    alpha_power_avg = np.mean(alpha_power, axis=0)
    theta_power_avg = np.mean(theta_power, axis=0)
    # compute vigilance index
    vigilance = alpha_power_avg / theta_power_avg
    return {
        "alpha_power": alpha_power_avg,
        "theta_power": theta_power_avg,
        "eeg_vigilance": vigilance,
    }


def _wavelet_power(
    data: np.ndarray, sf: int, frequency_band: Tuple[float, float], precision: int = 20
) -> np.ndarray:
    """
    Compute wavelet power of data in a given frequency band.

    Parameters
    ----------
        data: np.ndarray
            data (time x channel)
        sf: int
            sampling frequency
        frequency_band: Tuple[float, float]
            frequency band
        precision: int
            precision of frequency resolution

    Returns
    -------
    power: np.ndarray
        power of data in given frequency band (channel x time x frequency)
    """
    power = mne.time_frequency.tfr_array_morlet(
        # transpose to channel x time and add singleton dimension for epochs
        data.T[np.newaxis, ...],
        sfreq=sf,
        freqs=np.linspace(frequency_band[0], frequency_band[1], precision),
        output="power",
    )
    power = np.squeeze(power)
    # average across frequencies
    power = np.mean(power, axis=1)
    return power


def rsp_rvt_harrison(
    rsp_signal: np.ndarray,
    sf: int,
    boundaries: Tuple[float, float] = (2.0, 1 / 30),
    iterations: int = 10,
    silent: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
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
                fr_min, fr_max, num=n_end - n_start
            ).squeeze()  # type: ignore
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

    return fr_rv, fr_phase

"""
Module for estimating the relationship between functional MRI signals
and physio signals at successive temporal lags of the physio signal
"""

from __future__ import annotations

import os
import pickle
from typing import List, Literal, Tuple

import numpy as np
from patsy import dmatrix  # type: ignore
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

from scan.io.load import Gifti
from scan.io.write import write_func_gii


class HRFBasis:
    """
    Spline basis for modeling stimulus-evoked hemodynamic response functions.

    The basis is defined over post-stimulus time in seconds and is applied
    by convolving event regressors with spline basis functions.

    Parameters
    ----------
    duration_sec : float
        Duration of the HRF response window.

    knots_per_sec : float
        Approximate density of spline basis functions. This is used to determine the number of knots for the spline basis functions.

    knot_spacing : {"uniform", "geometric"}
        Strategy for distributing knots over the response window before any
        basis-type-specific adjustment.

    geometric_alpha : float
        Controls concentration of knots near stimulus onset.
        Larger values place more knots early in the HRF.

    Usage:
        >>> hrf_basis = HRFBasis(duration_sec=30.0, knot_spacing="geometric")
        >>> hrf_basis.create()
        >>> design_matrix = hrf_basis.project(task_regressor, tr=2.0)
    """

    def __init__(
        self,
        duration_sec: float,
        knots_per_sec: float = 0.2,
        knot_spacing: Literal["uniform", "geometric"] = "uniform",
        geometric_alpha: float = 2.0,
    ):

        if knot_spacing not in ("uniform", "geometric"):
            raise ValueError("knot_spacing must be 'uniform' or 'geometric'")

        self.duration_sec = duration_sec
        self.knots_per_sec = knots_per_sec
        self.knot_spacing = knot_spacing
        self.geometric_alpha = geometric_alpha

    def _create_knots(self):

        # Get number of basis functions based on duration and knots per second.
        n_basis = max(3, int(np.ceil(self.duration_sec * self.knots_per_sec)))

        # Create knots for spline basis functions.
        if self.knot_spacing == "uniform":
            knots = np.linspace(0, self.duration_sec, n_basis)

        else:
            # Create knots with geometric spacing to concentrate basis functions near stimulus onset.
            x = np.linspace(0, 1, n_basis)
            knots = (
                self.duration_sec
                * (np.exp(self.geometric_alpha * x) - 1)
                / (np.exp(self.geometric_alpha) - 1)
            )

        # cr() expects inner knots only
        if len(knots) > 2:
            knots = knots[1:-1]

        self.knots = knots

        return knots

    def create(self, dt: float = 0.1, extrapolation: str = "extend"):
        """
        Create the spline basis functions over the specified duration.

        Parameters
        ----------
        dt : float, default=0.1
            Time step for sampling the basis functions in seconds.
        extrapolation : str, default="extend"
            Extrapolation method for the spline basis functions, options
            specified in the formulaic documentation. Default is "extend" to
            allow evaluation of the basis functions at the minimum and maximum bounds of the response window.
        """

        self.dt = dt
        self.extrapolation = extrapolation

        self.times = np.arange(0, self.duration_sec + dt, dt)

        self.knots = self._create_knots()

        self.basis = np.asarray(
            dmatrix(
                "cr(x, knots=self.knots) - 1",
                {"x": self.times},
            ),
        )

        self._n_basis = self.basis.shape[1]

        return self

    def project(self, X, tr: float, fill_value: float = 0) -> np.ndarray:
        """
        Project a stimulus time course onto the spline basis functions.

        Parameters
        ----------
        X : np.ndarray
            One-dimensional stimulus time course with shape (n_timepoints,).
        tr : float
            Repetition time of the fMRI acquisition in seconds.
        fill_value : float, default=0
            Value used to fill samples before the start of the time series.
        """

        return self.project_to_times(X, tr=tr, fill_value=fill_value)

    def project_to_times(
        self,
        X,
        tr: float,
        fill_value: float = 0,
        sample_times: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Project a stimulus time course onto the spline basis and optionally
        resample the projected result at specific times.

        Parameters
        ----------
        X : np.ndarray
            One-dimensional stimulus time course with shape (n_timepoints,).
        tr : float
            Sampling interval of ``X`` in seconds.
        fill_value : float, default=0
            Value used to fill samples before the start of the time series.
        sample_times : np.ndarray | None, default=None
            Optional output times in seconds. If provided, the projected time
            course is interpolated to these times.
        """

        if not hasattr(self, "basis"):
            raise RuntimeError("Call create() before project().")

        # HRF lag grid (seconds)
        hrf_times = self.times

        # interpolate basis onto TR grid
        interp = interp1d(
            hrf_times,
            self.basis,
            axis=0,
            bounds_error=False,
            fill_value=0,
        )

        basis_tr = interp(np.arange(0, self.duration_sec + tr, tr))

        # build lag matrix using TR-based lags
        lags = np.arange(basis_tr.shape[0])

        lag_matrix = _lag_mat(X, lags, fill_val=fill_value)
        projected = lag_matrix @ basis_tr

        if sample_times is None:
            return projected

        stim_times = np.arange(projected.shape[0]) * tr
        interp = interp1d(
            stim_times,
            projected,
            axis=0,
            bounds_error=False,
            fill_value=0,
        )
        return interp(sample_times)


class GLMSplineResults:
    """
    Class for storing predictions of GLMSpline. Provides
    utilities for writing predicted time courses to func.gii files.

    Attributes
    ----------
    pred_func: np.ndarray
        predicted time courses from GLMSpline model represented as an ndarray
        with predicted time points in the rows and vertices in columns.

    glm_spline_params: dict
        the parameters used to fit the GLMSpline model

    Methods
    -------
    write(out_fp, out_dir=None):
        write predicted time courses to func.gii and GLMSpline params to pickle

    """

    def __init__(self, pred_func: np.ndarray, glm_spline_params: dict):
        self.pred_func = pred_func
        self.glm_spline_params = glm_spline_params

    def write(
        self,
        gii_params: Gifti,
        file_prefix: str = "glm_pred_out",
        out_dir: str | None = None,
    ) -> None:
        """
        Write out prediction results from GLMSpline model to func.gii and pickle
        file. The func.gii displayed the predicted fMRI values over the
        predicted time span, and the pickle contains params passed to the
        GLMSpline class.

        Parameters
        ----------
        gii_params: Gifti
            Gifti class that contains a loaded func.gii file. Used for
            writing out func.gii in the same format as the input func.gii.
            If running group-level analysis, this is returned in the
            Dataset.load() method.
        file_prefix: str
            Optional - file path prefix for pickle and func.gii file
        out_dir: str
            Optional - output directory for writing files. If None (default),
            write out to current working directory.
        """
        # set output prefix for file paths
        if out_dir is None:
            out_dir = os.getcwd()

        out_prefix = f"{out_dir}/{file_prefix}"
        # write out GLMSpline pred params
        with open(f"{out_prefix}.pkl", "wb") as f:
            pickle.dump(self.glm_spline_params, f)

        # write predicted time courses to func.gii
        write_func_gii(self.pred_func, gii_params, out_prefix)


class GLMSpline:
    """
    General linear model with natural cubic spline basis for modeling the
    event related response of fMRI signals to physiological signal events.

    tr: float
        repetition time of the fMRI acquisition in seconds
    duration_sec: int
        duration of the event-related response in seconds
    knots_per_sec: float
        approximate density of spline basis functions. This is used to determine the number of knots for the
        spline basis functions.
    knot_spacing: Literal["uniform", "geometric"]
        strategy for distributing knots over the response window before any basis-type-specific adjustment.
    geometric_alpha: float
        controls concentration of knots near stimulus onset. Larger values place more knots early in the HRF.
    slicetime_ref: float
        reference time for slice timing correction. This is used to adjust the event onsets for slice timing correction.
        The value should be between 0 and 1, where 0 corresponds to the first slice and 1 corresponds to the last slice.

    Methods
    -------
    fit(X,y):
        regress lags of physio signal onto voxel-wise functional time courses.

    predict()

    """

    def __init__(
        self,
        tr: float,
        duration_sec: float = 30.0,
        knots_per_sec: float = 0.2,
        knot_spacing: Literal["uniform", "geometric"] = "uniform",
        geometric_alpha: float = 2.0,
        slicetime_ref: float = 0.5,
    ):
        self.duration_sec = duration_sec
        self.knots_per_sec = knots_per_sec
        self.knot_spacing = knot_spacing
        self.tr = tr
        self.geometric_alpha = geometric_alpha
        self.slicetime_ref = slicetime_ref

    def _internal_dt(self) -> float:
        """Return the fine internal sampling interval used for stimulus timing."""

        return self.tr / 10.0

    def fit(
        self,
        events: list[list[float]],
        fmri: list[np.ndarray],
    ):
        """
        fit regression model of event-related response of fMRI signals to physiological signal events.

        Parameters
        ----------
        events: List[List[int]]
            The list of event onsets for each trial. Each sublist contains the onset times for a single trial.
        fmri: List[np.ndarray]
            functional MRI time courses represented as a list of ndarrays, each with time
            points along the rows and vertices in the columns (# of time
            points, # of vertices).
        """
        # create B-spline basis across lags of physio signal
        self.basis = HRFBasis(
            duration_sec=self.duration_sec,
            knots_per_sec=self.knots_per_sec,
            knot_spacing=self.knot_spacing,  # type: ignore
            geometric_alpha=self.geometric_alpha,  # type: ignore
        )
        fmri_list = []
        event_list = []
        stim_dt = self._internal_dt()
        self.basis.create(dt=stim_dt)
        for fmri_data, fmri_events in zip(fmri, events):
            # if no events, skip this fmri_data
            if len(fmri_events) == 0:
                continue
            n_frames = fmri_data.shape[0]
            # get time samples of functional scan based on slicetime reference
            frametimes = self.slicetime_ref + np.arange(n_frames) * self.tr
            fine_frametimes = np.arange(
                0, frametimes[-1] + self.duration_sec + stim_dt, stim_dt
            )
            # rasterize event onsets on the fine internal grid before projection
            event_regressor = _rasterize_events(
                np.asarray(fmri_events, dtype=float),
                fine_frametimes,
                event_duration_sec=0.1,
            )
            convolved = self.basis.project_to_times(
                event_regressor,
                tr=stim_dt,
                fill_value=0.0,
                sample_times=frametimes,
            )
            fmri_list.append(fmri_data)
            event_list.append(convolved)

        # concatenate fmri data and convolved event regressors across runs
        Y = np.vstack(fmri_list)
        X = np.vstack(event_list)
        # fit Ridge regression model
        self.glm = LinearRegression()
        self.glm.fit(
            X,
            Y,
        )
        return self

    def evaluate(
        self,
        duration_max: float | None = None,
        n_eval: int = 30,
        pred_val: float = 1.0,
    ) -> GLMSplineResults:
        """
        Evaluate the model for a

        Parameters
        ----------
        duration_max: float
            The maximum duration of the event to predict functional time
            courses for. If None, set to duration_max specified in initialization. (default: None)
        n_eval: int
            The number of evaluation points to use for predicting functional time
            courses. (default: 30)
        pred_val: float
            The predicted physio signal value used to predict functional time
            courses (default: 1.0).

        Returns
        -------
        glm_pred: GLMSplineResults
            Container object for GLM spline model prediction results
        """
        # if lag_max is None, set nlags
        if duration_max is None:
            duration_max = self.duration_sec

        if duration_max > self.duration_sec:
            raise ValueError(
                f"duration_max ({duration_max}) cannot be greater than duration_sec ({self.duration_sec})"
            )

        # specify lags for prediction (number of samples set by n_eval )
        pred_lags = np.linspace(0, duration_max, n_eval)
        stim_dt = self._internal_dt()
        n_stim = int(np.ceil(self.duration_sec / stim_dt)) + 1
        impulse = np.zeros(n_stim, dtype=float)
        impulse[0] = pred_val
        pred_func = self.basis.project_to_times(
            impulse,
            tr=stim_dt,
            fill_value=0.0,
            sample_times=pred_lags,
        )
        pred_func = self.glm.predict(pred_func)
        return GLMSplineResults(
            pred_func=pred_func,
            glm_spline_params={
                "duration_max": duration_max,
                "n_eval": n_eval,
                "pred_lags": pred_lags,
                "pred_val": pred_val,
            },
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict functional MRI time courses from event time courses.

        Parameters
        ----------
        X: np.ndarray
            The event time course represented in an ndarray with time points
            along the rows and a single column (# of time points, 1).
        """
        X = np.asarray(X)
        if X.ndim != 1:
            raise ValueError("X must be a one-dimensional array.")

        # project physio signal lags on B-spline basis
        x_basis = self.basis.project(X, tr=self.tr, fill_value=0.0)
        # get predictions from model
        pred_func = self.glm.predict(x_basis)
        return pred_func


def _lag_mat(
    X: np.ndarray,
    lags: np.ndarray,
    fill_val: float = 0.0,
) -> np.ndarray:
    """
    Create a positive-lagged design matrix from a stimulus time course.

    Parameters
    ----------
    X : np.ndarray
        One-dimensional stimulus time course with shape (n_timepoints,).

    lags : np.ndarray
        Positive integer lags (in TRs). Lag 0 corresponds to the original
        time course.

    fill_val : float, default=0.0
        Value used to fill samples before the start of the time series.

    Returns
    -------
    lagged : np.ndarray
        Lag matrix with shape (n_timepoints, n_lags).

        Column j contains X shifted by lags[j] TRs:

            lagged[t, j] = X[t - lags[j]]

    Notes
    -----
    This function is intended for distributed lag / HRF basis modeling,
    where each column represents the stimulus history at a different
    post-stimulus delay.
    """

    X = np.asarray(X)

    if X.ndim != 1:
        raise ValueError("X must be a one-dimensional array.")

    lags = np.asarray(lags)

    if np.any(lags < 0):
        raise ValueError("Only positive lags are allowed.")

    if not np.all(lags.astype(int) == lags):
        raise ValueError("lags must contain integer TR offsets.")

    lags = lags.astype(int)

    n_time = X.shape[0]

    lagged = np.full(
        (n_time, len(lags)),
        fill_val,
        dtype=float,
    )

    for i, lag in enumerate(lags):
        if lag == 0:
            lagged[:, i] = X

        else:
            lagged[lag:, i] = X[:-lag]

    return lagged


def _rasterize_events(
    onsets: np.ndarray,
    sample_times: np.ndarray,
    event_duration_sec: float = 0.1,
) -> np.ndarray:
    """
    Rasterize event onsets onto a uniformly sampled stimulus grid.

    Parameters
    ----------
    onsets : np.ndarray
        Event onset times in seconds.
    sample_times : np.ndarray
        Uniformly spaced time points defining the stimulus grid.
    event_duration_sec : float, default=0.1
        Duration of each event boxcar in seconds.
    """

    onsets = np.asarray(onsets, dtype=float)
    sample_times = np.asarray(sample_times, dtype=float)

    if sample_times.ndim != 1:
        raise ValueError("sample_times must be a one-dimensional array.")

    if sample_times.size == 0:
        return np.zeros(0, dtype=float)

    if sample_times.size == 1:
        dt = event_duration_sec
    else:
        deltas = np.diff(sample_times)
        dt = float(np.median(deltas))
        if not np.allclose(deltas, dt):
            raise ValueError("sample_times must be evenly spaced.")

    stimulus = np.zeros(sample_times.shape[0], dtype=float)
    if onsets.size == 0:
        return stimulus

    start_time = sample_times[0]
    for onset in onsets:
        start_idx = int(np.floor((onset - start_time) / dt))
        stop_idx = int(np.ceil((onset + event_duration_sec - start_time) / dt))
        start_idx = max(start_idx, 0)
        stop_idx = min(stop_idx, stimulus.shape[0])
        if stop_idx > start_idx:
            stimulus[start_idx:stop_idx] = 1.0

    return stimulus

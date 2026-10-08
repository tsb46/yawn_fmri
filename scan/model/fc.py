"""
Module for estimating seed-based functional connectivity modulation
by a continuous signal interaction.
"""

import os
import pickle

import numpy as np
from sklearn.linear_model import LinearRegression

from scan.io.load import Gifti
from scan.io.write import write_func_gii


class FCMapResults:
    """
    Class for storing results of simple seed-based functional connectivity.

    Attributes
    ----------
    fc_map: np.ndarray
        Seed-based functional connectivity map.
    model_params: dict
        Parameters used to fit the model.
    """

    def __init__(self, fc_map: np.ndarray, model_params: dict):
        self.fc_map = fc_map
        self.model_params = model_params

    def write(
        self,
        gii_params: Gifti,
        file_prefix: str | None = None,
        out_dir: str | None = None,
    ) -> None:
        """
        Write out a simple seed-based functional connectivity map to func.gii.
        """
        if out_dir is None:
            out_dir = os.getcwd()

        out_prefix = f"{out_dir}/{file_prefix}"

        with open(f"{out_prefix}.pkl", "wb") as f:
            pickle.dump(self.model_params, f)

        write_func_gii(self.fc_map[np.newaxis, :], gii_params, out_prefix)


class FCMapModel:
    """
    Class for estimating simple seed-based functional connectivity.

    Attributes
    ----------
    model : LinearRegression
        The fitted linear regression model.
    """

    def fit(self, seed_ts: np.ndarray, func_data: np.ndarray) -> FCMapResults:
        """
        Fit a simple seed-based FC model.

        Parameters
        ----------
        seed_ts: np.ndarray
            1d time series data for a seed region of interest (ROI).
        func_data : np.ndarray
            Functional MRI data: a 2D array where rows are time points and columns
            are vertices or regions.

        Returns
        -------
        FCMapResults
            Object containing the seed-based FC map.
        """
        seed_ts = np.asarray(seed_ts)
        func_data = np.asarray(func_data)

        if func_data.ndim != 2:
            raise ValueError("func_data must be a 2d array.")

        if seed_ts.ndim == 1:
            seed_ts = seed_ts[:, np.newaxis]
        elif seed_ts.ndim == 2 and seed_ts.shape[1] == 1:
            pass
        else:
            raise ValueError("seed_ts must be a 1d array or a single-column 2d array.")

        if seed_ts.shape[0] != func_data.shape[0]:
            raise ValueError(
                "seed_ts and func_data must have the same number of time points."
            )

        self.model = LinearRegression()
        self.model.fit(seed_ts, func_data)

        fc_map = np.asarray(self.model.coef_).squeeze()

        return FCMapResults(
            fc_map=fc_map,
            model_params={
                "model": "linear_regression",
                "n_timepoints": int(func_data.shape[0]),
                "n_vertices": int(func_data.shape[1]),
            },
        )

"""
Module for loading and concatenating functional (func.gii),
eeg and physio.
"""

import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import neurokit2 as nk
import nibabel as nb
import numpy as np
from sklearn.linear_model import LinearRegression

from scan.io import utils
from scan.io.file import Participant

LH_MEDIAL_WALL_MASK = "template/fsLR_hemi-L_den-32k_desc-nomedialwall_dparc.label.gii"
RH_MEDIAL_WALL_MASK = "template/fsLR_hemi-R_den-32k_desc-nomedialwall_dparc.label.gii"


# Dataset output
@dataclass
class DatasetOutput:
    func: List[np.ndarray]
    physio: Dict[str, List[np.ndarray]]
    sigh_events: List[List[float]]
    yawn_events: List[List[float]]

    def concatenate(self) -> "DatasetOutputConcat":
        """
        Concatenate the lists of arrays in the DatasetOutput into single arrays.
        """
        concatenated_output = DatasetOutputConcat(
            func=np.concatenate(self.func, axis=0),
            physio={
                key: np.concatenate(value, axis=0) for key, value in self.physio.items()
            },
            sigh_events=[item for sublist in self.sigh_events for item in sublist],
            yawn_events=[item for sublist in self.yawn_events for item in sublist],
        )
        return concatenated_output


@dataclass
class DatasetOutputConcat:
    func: np.ndarray
    physio: Dict[str, np.ndarray]
    sigh_events: List[float]
    yawn_events: List[float]


class Gifti:
    """
    Class for loading left and right hemisphere
    func.gii files for analysis

    Attributes
    ----------
    fp_gii_lh: str
        filepath to left hemisphere func.gii
    fp_gii_rh: str
        filepath to right hemisphere func.gii

     Methods
    -------
    load():
        load left and right hemisphere func.gii and concatenate into array
    split()
        split concatenated array into left and right hemisphere arrays (in
        that order)

    """

    def __init__(self, fp_gii_lh: str, fp_gii_rh: str, no_medial_wall: bool = False):
        # Load the GIFTI files for both hemispheres
        self.gii_lh = nb.load(fp_gii_lh)  # type: ignore
        self.gii_rh = nb.load(fp_gii_rh)  # type: ignore

        # create medial wall mask if specified
        if no_medial_wall:
            self.mask_lh = nb.load(LH_MEDIAL_WALL_MASK).darrays[0].data.astype(bool)  # type: ignore
            self.mask_rh = nb.load(RH_MEDIAL_WALL_MASK).darrays[0].data.astype(bool)  # type: ignore
        else:
            self.mask_lh = np.ones(self.gii_lh.darrays[0].data.shape[0], dtype=bool)  # type: ignore
            self.mask_rh = np.ones(self.gii_rh.darrays[0].data.shape[0], dtype=bool)  # type: ignore

        _n_samples_lh = len(self.gii_lh.darrays)  # type: ignore
        _n_samples_rh = len(self.gii_rh.darrays)  # type: ignore
        if _n_samples_lh != _n_samples_rh:
            raise ValueError(
                "left and right hemispheres should have same number of samples"
            )
        self.n_samples = _n_samples_lh

        self.split_indx = self.gii_lh.darrays[0].data[self.mask_lh].shape[0]  # type: ignore
        # get # of vertices per hemisphere
        self.lh_nvert = self.gii_lh.darrays[0].data[self.mask_lh].shape[0]  # type: ignore
        self.rh_nvert = self.gii_rh.darrays[0].data[self.mask_rh].shape[0]  # type: ignore
        if self.lh_nvert != self.rh_nvert:
            raise ValueError(
                "left and right hemispheres should have same number of vertices"
            )

    def load(self) -> np.ndarray:
        """
        load left and right hemisphere func.gii and concatenate into array
        """
        # loop through samples and concatenate into one array
        combined_data = []
        for lh_d, rh_d in zip(self.gii_lh.darrays, self.gii_rh.darrays):  # type: ignore
            # Access the data arrays in the GIFTI files
            data_left = lh_d.data[self.mask_lh]
            data_right = rh_d.data[self.mask_rh]
            combined_data.append(np.hstack((data_left, data_right)))

        return np.vstack(combined_data)

    def load_separate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        load left and right hemisphere func.gii and return as separate arrays
        """
        # loop through arrays and return as separate arrays
        lh_data = []
        rh_data = []
        for lh_d, rh_d in zip(self.gii_lh.darrays, self.gii_rh.darrays):  # type: ignore
            lh_data.append(lh_d.data[self.mask_lh])
            rh_data.append(rh_d.data[self.mask_rh])
        return np.array(lh_data), np.array(rh_data)

    def merge(self, lh_data: np.ndarray, rh_data: np.ndarray) -> np.ndarray:
        """
        Concatenate left and right hemisphere arrays into single array
        """
        if lh_data.shape[1] != self.lh_nvert:
            raise ValueError(
                "the # of vertices in the input left hemisphere data does"
                "not match the number of expected left hemisphere vertices"
            )
        if rh_data.shape[1] != self.rh_nvert:
            raise ValueError(
                "the # of vertices in the input right hemisphere data does"
                "not match the number of expected right hemisphere vertices"
            )
        return np.hstack((lh_data, rh_data))

    def split(self, combined_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        split concatenated array into left and right hemisphere arrays (in
        that order). The array is expected to have # of samples in rows
        and # of vertices in the columns
        """
        if combined_data.shape[1] != (self.lh_nvert + self.rh_nvert):
            raise ValueError(
                "the # of vertices in combined_data does not match the number "
                "left and right hemisphere vertices"
            )
        data_left = combined_data[:, : self.split_indx]
        data_right = combined_data[:, self.split_indx :]
        return data_left, data_right


class DatasetLoad:
    """
    Class for loading scan data, functional MRI (.gii) or physio, and
    concatenation across scans (optional).

    Attributes
    ----------
    dataset: Literal['vanderbilt']
        dataset label
    subj_ses_select: List[Tuple[str, str]] | None
        list of subject and session label pairs to load. If None, load all
        subjects and sessions in the dataset (default: None). For subjects with
        only one session, pass as a tuple with the session as None (e.g. ('01', None))
    physio_dir:
        physio directory name for preprocessing output. Should be
        'proc1_physio' (default: 'proc1_physio')
    func_dir
        functional directory name for preprocessing output. If you
        want to analyze in native space, specify 'proc5_surface_smooth'.
        Otherwise, data is loaded from the last preprocessing step where
        functional data is in fsLR space: 'proc6_surfacelr'.
        (default: 'proc6_surfacelr')

    Methods
    -------
    load(concat = True):
        Iterate through scan data and concatenate (optional)
    load_scan(data, subj, ses)
        Load data for individual scan
    """

    def __init__(
        self,
        dataset: Literal["vanderbilt"],
        subj_ses_select: List[Tuple[str, str]] | None = None,
        physio_dir: str = "proc1_physio",  # last output of physio pipeline
        func_dir: str = "proc6_surfacelr",  # last output of func pipeline
    ):
        self.dataset = dataset
        self.subj_ses_select = subj_ses_select
        # get dataset parameters
        # get data formatting
        with open("scan/meta/params.json", "rb") as f:
            self.params = json.load(f)[dataset]
        # define output directories to search for files
        self.func_dir = f"{self.params['directory']['func']}/{func_dir}"
        self.physio_dir = f"{self.params['directory']['physio']}/{physio_dir}"
        # get scan iterator
        self.iter = Participant(dataset, subj_ses_select=subj_ses_select)
        # check if multiple sessions per subject
        self.session_flag = "session" in self.iter.fields

    def load(
        self,
        data_type: Tuple[Literal["func", "physio"], ...] | Literal["func", "physio"] = (
            "func",
            "physio",
        ),
        verbose: bool = True,
        norm: Literal["zscore", "demean", "robust_z"] | None = "zscore",
        resample_physio: bool = True,
        func_low_pass: bool = False,
        func_high_pass: bool = False,
        physio_low_pass: bool = False,
        physio_high_pass: bool = False,
        regress_global_signal: bool = False,
        input_mask: bool = False,
        lh_roi_masks: List[str] | None = None,
        rh_roi_masks: List[str] | None = None,
        roi_avg_left_right: bool = False,
    ) -> Tuple[DatasetOutput, Gifti | None]:
        """
        Iteratively load scan data and concatenate for group
        analysis (optional). Data can be functional (gii) or physio, or both.
        If concatenation is set to false, return the data for individual
        subjects in a list. Two outputs are returned, inluding the data in a
        dictionary with the key as the data modality (e.g. 'physio'), and a
        Gifti class from the last scan (this is needed for writing out
        outputs to func.gii after analysis). This assumes that the func.gii
        are consistent in shape (# of vertices in the left and right hemispheres
        are consistent across scans). This should be the case when performing
        group analysis.

        Parameters
        ----------
        data_type: Tuple[str, ...] | str = ('func', 'physio')
            data modality. Can be functional or physio, or both. If both
            pass as a tuple (default: ('func', 'physio'))
        norm: Literal['zscore', 'demean', 'robust_z', None]:
            The type of normalization to perform on the time courses. This
            is important if performing concatenation to remove differences
            in baseline signal between scans. Using zscore, differences in signal
            variability between scan are also removed (default: zscore). Using robust_z,
            the median and median absolute deviation are used for normalization,
            which is less sensitive to outliers (default: zscore).
        resample_physio: bool
            whether to resample physio data to match the functional TR (default: True). If False,
            the physio data is returned at the original sampling frequency (10Hz).
        func_high_pass: bool
            whether to perform a high-pass (>0.01Hz) 5th order butterworth filter
            to both physio and func time courses with nilearn.signal.butterworth.
            This is performed at the individual scan level before concatenation
            (default: False)
        func_low_pass: bool
            whether to perform a low-pass (<0.15Hz) 5th order butterworth filter
            to both physio and func time courses with nilearn.signal.butterworth.
            This is performed at the individual scan level before concatenation
            (default: False)
        physio_high_pass: bool
            high-pass filtering on physio time courses (default: False)
        physio_low_pass: bool
            low-pass filtering on physio time courses (default: False)
        regress_global_signal: bool
            whether to regress out the global signal from func.gii data. Note,
             global signal regression should not be applied to ROI time courses (default: False)
        input_mask: bool
            whether to apply roi masks to func.gii data (default: False).
            Masks should have have a value of 1 for vertices within the mask,
            and 0 for vertices outside the mask. Time courses of vertices within the mask
            are averaged together, and returned instead of the vertex time courses.
            BOTH left and right hemisphere roi masks should be passed as a list, with
            matching ROIs in the left and right hemisphers in the same order.
        lh_roi_masks: List[str]
            list of left hemisphere roi mask file paths to apply to
            func.gii data. See input_mask for more details. If input_mask is False,
            this parameter is ignored.
        rh_roi_masks: List[str]
            list of right hemisphere roi mask file paths to apply to
            func.gii data. See input_mask for more details. If input_mask is False,
            this parameter is ignored.
        roi_avg_left_right: bool
            whether to average left and right hemisphere ROI time courses together (default: False).
            Note, this is only applicable if the positions of the left and right hemisphere ROI masks
            are in the same order in the lists passed to lh_roi_masks and rh_roi_masks. If this is not the case,
            the left and right hemisphere ROI time courses will be averaged together incorrectly.
        verbose: bool
            print progress (default: True)

        Returns
        -------
        output: DatasetOutput
            group data in a list or one concatenated array packaged
            in a dictionary where the key is the data modality.
        gii: Gifti
            Gifti class for storing gifti parameters, this is needed for
            writing out outputs to func.gii after analysis. Gifti class
            is returned only if 'func' is passed to data parameter. Otherwise,
            returns None
        """
        # if data_type is not passed, set to all data types
        if data_type is None:
            data_type = ("func", "physio")
        # if data_type is passed as str, convert to list
        if isinstance(data_type, str):
            data_type = (data_type,)
        # check if data_type is valid
        if not all(d in ["func", "physio"] for d in data_type):
            raise ValueError(f"data type {data_type} is not available")

        # if roi_masks are passed, load them
        if input_mask:
            if lh_roi_masks is None or rh_roi_masks is None:
                raise ValueError(
                    "roi_lh_masks and roi_rh_masks must be provided if input_mask is True"
                )
            if roi_avg_left_right and (len(lh_roi_masks) != len(rh_roi_masks)):
                raise ValueError(
                    "The number of left and right hemisphere ROI masks must be the same"
                    " if roi_avg_left_right is True"
                )
            lh_roi, rh_roi = self._load_masks(lh_roi_masks, rh_roi_masks)
        else:
            if lh_roi_masks is not None or rh_roi_masks is not None:
                warnings.warn(
                    "roi masks are passed, but input_mask is False. ROI masks will be ignored."
                )
            lh_roi, rh_roi = None, None

        # initalize output dictionary
        _output = {
            "func": [],
            "physio": {
                p_out: []
                for p in self.params["physio"]["out"]
                for p_out in self.params["physio"]["out"][p]
            },
            "sigh_events": [],
            "yawn_events": [],
        }
        # set func_gii as None (returns None if 'physio' is set as data)
        func_gii = None
        for subj_ses in self.iter:
            # if multiple sessions per participant, split into subj and session
            if self.session_flag:
                subj = subj_ses[0]
                ses = subj_ses[1]
            else:
                subj = subj_ses[0]
                ses = None
            # print progress
            if verbose:
                print(f"loading scan: subj: {subj} ses: {ses}")
            # loop through data modalities, load data and append to list
            data_out, func_gii = self.load_scan(
                subj=subj,
                ses=str(ses),
                data=data_type,
                norm=norm,
                resample_physio=resample_physio,
                func_low_pass=func_low_pass,
                func_high_pass=func_high_pass,
                physio_low_pass=physio_low_pass,
                physio_high_pass=physio_high_pass,
                regress_global_signal=regress_global_signal,
                roi_lh_masks=lh_roi,
                roi_rh_masks=rh_roi,
                input_mask=input_mask,
                roi_avg_left_right=roi_avg_left_right,
            )
            _output["func"].append(data_out["func"])
            _output["sigh_events"].append(data_out["sigh_events"])
            _output["yawn_events"].append(data_out["yawn_events"])
            # loop through physio signals and append to list
            for p in self.params["physio"]["out"]:
                for p_out in self.params["physio"]["out"][p]:
                    _output["physio"][p_out].append(data_out["physio"][p_out])

        dataset_output = DatasetOutput(
            func=_output["func"],
            physio=_output["physio"],
            sigh_events=_output["sigh_events"],
            yawn_events=_output["yawn_events"],
        )

        return dataset_output, func_gii

    def load_scan(
        self,
        subj: str,
        ses: str,
        data: Tuple[Literal["func", "physio"], ...] = ("func", "physio"),
        norm: Literal["zscore", "demean", "robust_z"] | None = "zscore",
        resample_physio: bool = True,
        func_low_pass: bool = False,
        func_high_pass: bool = False,
        physio_low_pass: bool = False,
        physio_high_pass: bool = False,
        regress_global_signal: bool = False,
        roi_lh_masks: Dict[str, np.ndarray] | None = None,
        roi_rh_masks: Dict[str, np.ndarray] | None = None,
        input_mask: bool = False,
        roi_avg_left_right: bool = False,
    ) -> Tuple[dict, Gifti | None]:
        """
        given subject and session label, load func or physio data. Data is
        returned in a dictionary with 'func' and 'physio' as separate keys (
        if both data modalities chosen). Functional time courses are
        returned as a 2D np.ndarray (# of timepoints, # of vertices),
        physio data is returned as a dictionary with keys as the physio signal
        label and values as the physio signal in a 2D np.ndarray (# of
        timepoints, 1). In addition, Gifti class for storing gifti parameters
        are returned; this is needed for writing out outputs to func.gii after
        analysis. Gifti class is returned only if 'func' is included in the
        list passed to data parameter. Otherwise, returns None.

        Parameters
        ----------
        data: Tuple[Literal['func', 'physio']]
            list of data modalities - can only be 'func' and/or physio
        subj: str
            subject label
        ses: str
            subject
        norm: Literal['zscore', 'demean', 'robust_z', None]
            whether to normalize the data (default: zscore).
        resample_physio: bool
            whether to resample physio data to match func.gii TR (default: True). If False,
            physio data is returned at the original sampling frequency (10Hz). Note, the
            sigh and yawn event times are always returned in seconds. Also note,
            the raw sampling frequency of head motion matches the func.gii TR.
        func_low_pass: bool
            whether to perform low-pass filtering on func.gii data
        func_high_pass: bool
            whether to perform high-pass filtering on func.gii data
        physio_low_pass: bool
            whether to perform low-pass filtering on physio data
        physio_high_pass: bool
            whether to perform high-pass filtering on physio data
        regress_global_signal: bool
            whether to regress out the global signal from func.gii data. Note,
            global signal regression should not be applied to ROI time courses.
            Default is False.
        roi_lh_masks: Dict[str, np.ndarray]
            left hemisphere roi mask with keys as the roi name
        roi_rh_masks: Dict[str, np.ndarray]
            right hemisphere roi mask with keys as the roi name
        input_mask: bool
            whether to apply roi masks to func.gii data
        roi_avg_left_right: bool
            whether to average left and right hemisphere ROI time courses together (default: False)
        eog_emg_average: bool
            whether to average EOG and EMG physiological signals across channels (default: False)

        Returns
        -------
        output: dict
            func and/or physio data returned as a dictionary. Top-level
            keys are 'func' and 'physio'. Within 'physio', different physio
            signals are returned as a dictionary with physio labels as keys (
            e.g. 'eog1').
        """
        # check data modality labels
        for d in data:
            if d not in ["func", "physio"]:
                raise ValueError(f"data modality {data} is not available")

        # set func_gii as None (returns None if 'physio' is set as data)
        func_gii = None
        # initialize output dictionary
        output = {
            "func": [],
            "physio": {
                p_out: []
                for p in self.params["physio"]["out"]
                for p_out in self.params["physio"]["out"][p]
            },
        }

        # get data from left and right hemispheres
        fp_lh = self.iter.to_file(
            data="func",
            subject=subj,
            session=ses,
            basedir=self.func_dir,
            file_ext="lh.func.gii",
        )
        fp_rh = self.iter.to_file(
            data="func",
            subject=subj,
            session=ses,
            basedir=self.func_dir,
            file_ext="rh.func.gii",
        )
        # initialize Gifti class
        func_gii = Gifti(fp_lh, fp_rh)

        for d in data:
            if d == "func":
                # load gifti data
                if input_mask:
                    if roi_lh_masks is None or roi_rh_masks is None:
                        raise ValueError(
                            "roi_lh_masks and roi_rh_masks must be provided if input_mask is True"
                        )
                    func_data = self._extract_roi(
                        func_gii,
                        roi_lh_masks,
                        roi_rh_masks,
                        roi_avg_left_right=roi_avg_left_right,
                    )
                else:
                    func_data = func_gii.load()

                # perform global signal regression
                if regress_global_signal:
                    if input_mask:
                        warnings.warn(
                            "Global signal regression is being applied to ROI time courses. "
                            "Recommend setting regress_global_signal to False when using ROI masks."
                        )
                    global_signal = func_data.mean(axis=1, keepdims=True)
                    reg = LinearRegression().fit(global_signal, func_data)
                    func_data = func_data - reg.predict(global_signal)

                # signal filtering, if specified in init
                func_data_proc = utils.filter(
                    func_data,
                    low_pass=func_low_pass,
                    high_pass=func_high_pass,
                    tr=self.params["func"]["tr"],
                )
                # normalize data, if specified in init
                func_data_proc = utils.norm(func_data_proc, norm=norm)
                output[d] = func_data_proc

            if d == "physio":
                # loop through physio signals of dataset
                output[d] = {}
                for p in self.params["physio"]["out"]:
                    for p_out in self.params["physio"]["out"][p]:
                        # all physio preprocessing outputs are .txt files
                        physio_fp = self.iter.to_file(
                            data=d,
                            subject=subj,
                            session=ses,
                            basedir=self.physio_dir,
                            physio=p_out,
                            file_ext="txt",
                            physio_type="out",
                        )
                        physio = np.loadtxt(physio_fp, ndmin=2)
                        # motion data is already sampled at functional TR
                        if p == "motion":
                            physio_tr = self.params["func"]["tr"]
                        # non-motion physio is 10Hz sampling frequency, resample to match func.gii TR
                        elif resample_physio:
                            physio = resample_physio_to_func(
                                physio,
                                func_tr=self.params["func"]["tr"],
                                sf=10,  # physio sampling frequency
                                func_len=func_gii.n_samples,
                                interp_method="cubic"
                                if p_out != "weight"
                                else "nearest",
                            )
                            physio_tr = self.params["func"]["tr"]
                        else:
                            physio_tr = 0.1  # physio sampling frequency

                        # signal filtering, if specified in init
                        # do not filter or normalize if physio is sample weights
                        if p_out != "weight":
                            physio = utils.filter(
                                physio,
                                low_pass=physio_low_pass,
                                high_pass=physio_high_pass,
                                tr=physio_tr,
                            )
                            # normalize data, if specified in init
                            physio = utils.norm(physio, norm=norm)
                        output[d][p_out] = physio

        # load yawns and sigh events
        event_fp = f"{self.params['event_dir']}/subject-{subj}_session-{ses}.json"
        with open(event_fp, "r") as f:
            events = json.load(f)
            sigh_events = [
                marker["time_seconds"]
                for marker in events["markers"]
                if marker["label"] == "Sigh"
            ]
            yawn_events = [
                marker["time_seconds"]
                for marker in events["markers"]
                if marker["label"] == "Yawn"
            ]
        output["sigh_events"] = sigh_events
        output["yawn_events"] = yawn_events

        return output, func_gii

    def _extract_roi(
        self,
        gifti: Gifti,
        left_roi_masks: Dict[str, np.ndarray],
        right_roi_masks: Dict[str, np.ndarray],
        roi_avg_left_right: bool = False,
    ) -> np.ndarray:
        """
        extract roi time courses from func.gii data

        Parameters
        ----------
        gifti: Gifti
            Gifti class for storing gifti parameters
        left_roi_masks: Dict[str,np.ndarray]
            left hemisphere roi mask with keys as the roi name
        right_roi_masks: Dict[str, np.ndarray]
            right hemisphere roi mask with keys as the roi name
        roi_avg_left_right: bool
            whether to average left and right hemisphere ROI time courses together (default: False)

        Returns
        -------
        roi_data: np.ndarray
            roi time courses arranged in column-order (left hemisphere time courses
            followed by right hemisphere time courses). If roi_avg_left_right is True,
            left and right hemisphere time courses are averaged together.
        """
        lh_func, rh_func = gifti.load_separate()
        # loop through roi masks and extract roi time courses
        roi_data = []
        roi_names = []
        for lh_roi_name, lh_roi in left_roi_masks.items():
            roi_data.append(lh_func[:, lh_roi].mean(axis=1)[:, np.newaxis])
            roi_names.append(lh_roi_name)
        for rh_roi_name, rh_roi in right_roi_masks.items():
            roi_data.append(rh_func[:, rh_roi].mean(axis=1)[:, np.newaxis])
            roi_names.append(rh_roi_name)
        roi_data = np.hstack(roi_data)
        if roi_avg_left_right:
            # average left and right hemisphere time courses together
            half = roi_data.shape[1] // 2
            roi_data = (roi_data[:, :half] + roi_data[:, half:]) / 2
            self.roi_names = [
                f"{lh_roi_name}_{rh_roi_name}"
                for lh_roi_name, rh_roi_name in zip(
                    left_roi_masks.keys(), right_roi_masks.keys()
                )
            ]
        else:
            # set roi names for future reference
            self.roi_names = roi_names
        return roi_data

    def _load_masks(
        self, lh_roi_mask_fps: List[str], rh_roi_mask_fps: List[str]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        check if roi masks are valid and load them
        """
        # check if roi_masks are passed as a list
        if not isinstance(lh_roi_mask_fps, list):
            raise TypeError("lh_roi_mask_fps must be passed as a list")
        # check if roi_masks are passed as a list
        if not isinstance(rh_roi_mask_fps, list):
            raise TypeError("rh_roi_mask_fps must be passed as a list")
        # check if roi masks are valid
        if not all(os.path.exists(roi_mask) for roi_mask in lh_roi_mask_fps):
            raise ValueError("lh_roi_mask_fps must be valid file paths")
        if not all(os.path.exists(roi_mask) for roi_mask in rh_roi_mask_fps):
            raise ValueError("rh_roi_mask_fps must be valid file paths")
        # load roi masks
        lh_roi = [nb.load(roi_mask).darrays[0].data for roi_mask in lh_roi_mask_fps]  # type: ignore
        rh_roi = [nb.load(roi_mask).darrays[0].data for roi_mask in rh_roi_mask_fps]  # type: ignore
        # check if roi masks are valid
        utils.check_roi_masks(lh_roi, rh_roi)
        # convert to boolean mask
        lh_roi_mask = {
            roi_name: roi_mask == 1
            for roi_name, roi_mask in zip(lh_roi_mask_fps, lh_roi)
        }
        rh_roi_mask = {
            roi_name: roi_mask == 1
            for roi_name, roi_mask in zip(rh_roi_mask_fps, rh_roi)
        }
        return lh_roi_mask, rh_roi_mask


def resample_physio_to_func(
    signal: np.ndarray,
    func_tr: float,
    func_len: int,
    sf: float,
    interp_method: Literal["cubic", "nearest"] = "cubic",
) -> np.ndarray:
    """
    Resample preprocessed physio signals to functional scan volumes using
    cubic or nearest neighbor interpolation to the functional times
    attribute. For physio recordings, signals should be first low-pass
    filtered (< 0.2 Hz).

    Parameters
    ----------
    signal: np.ndarray
        physio signals
    func_tr: float
        functional scan repetition time (TR)
    func_len: int
        number of functional scan volumes
    sf: float
        sampling frequency of physio signal
    interp_method
        interpolation method for interpolating physio data to functional
        volume samples - use 'nearest' for weights resampling. Otherwise, 'cubic'.

    Returns
    -------
    signal_resamp: np.ndarray
        physio signal resampled to functional volumes
    """
    # get functional time points
    func_t = _calc_func_frame_times(func_tr, func_len)
    # dont filter sample weights
    if interp_method not in ["cubic", "nearest"]:
        raise ValueError("interp method must be cubic or nearest")
    # get signal time points
    signal_t = np.arange(len(signal)) * (1 / sf)
    signal_resamp = nk.signal.signal_interpolate(
        x_values=signal_t,
        y_values=np.squeeze(signal),
        x_new=func_t,
        method=interp_method,
    )
    return signal_resamp


def _calc_func_frame_times(func_tr: float, func_len: int) -> np.ndarray:
    """
    Calculate interpolation time points from physio to functional samples.
    FSL slicetimer aligns all functional slices to the middle of the
    TR (0.5 * TR), so time points should be selected with this in mind.

    Parameters
    ----------
    func_tr: float
        functional scan repetition time (TR)
    func_len: int
        number of functional scan volumes

    Returns
    -------
    frame_times: np.ndarray
        time points (from the start of the functional scan) of preprocessed
        functional volumes to interpolate physio signals to

    """
    # calculate interpolation time points
    frame_times = func_tr * (np.arange(func_len) + 0.5)
    # round to two decimal points
    frame_times = np.round(frame_times, 2)
    return frame_times

"""
Pipeline module for preprocessing anatomical, functional, EEG and physio
datasets.
"""

import json
import os

from enum import Enum
from multiprocessing import Pool
from typing import List, Literal, Tuple

import matplotlib.pyplot as plt
import neurokit2 as nk
import nibabel as nb
import numpy as np
import pandas as pd
from tedana.workflows import tedana_workflow

from scan.io.file import Participant
from scan.preprocess import dataset as ds
from scan.preprocess import fsl
from scan.preprocess import freesurfer as fs
from scan.preprocess import physio
from scan.preprocess import workbench as wb
from scan.preprocess.custom import trim_cifti
from scan.preprocess.fsl import load_fsl_motion_params
from scan import utils


def physio_func_map(
    signal_labels: list[str],
) -> dict:
    """
    Create a dictionary of physio function mappings.

    Parameters
    ----------
    signal_labels: list[str]
        list of signal labels

    Returns
    -------
    physio_func_dict: dict
        dictionary of physio function mappings and their output label mappings
    """
    physio_func_dict = {}
    for p in signal_labels:
        if p.startswith(("eog", "emg")):
            func = physio.extract_emg_amplitude
            # fstring for potentially multiple channels
            output_map = {"emg_amp": f"{p}_amp"}
        elif p.startswith("resp"):
            func = physio.extract_resp_rvt
            output_map = {
                "resp_amp": "resp_amp",
                "resp_rate": "resp_rate",
            }
        elif p.startswith("weight"):
            func = physio.extract_sample_weight
            output_map = {"weight": "weight"}
        elif p.startswith("eeg"):
            func = physio.extract_eeg_vigilance
            output_map = {
                "eeg_vigilance": "eeg_vigilance",
                "theta_power": "theta_power",
                "alpha_power": "alpha_power",
            }
        elif p.startswith("motion"):
            func = physio.extract_motion
            output_map = {
                "fd": "framewise_displacement",
                "pitch": "pitch",
                "trans_z": "trans_z",
                "trans_y": "trans_y",
            }
        else:
            raise ValueError(f"Invalid signal type: {p}")

        physio_func_dict[p] = {"func": func, "output_map": output_map}

    return physio_func_dict


class AnatPipeOutMisc(Enum):
    """
    Enum for miscellaneous anatomical pipeline outputs
    """

    FS_BRAINMASK = "fs_brainmask"
    FS_MID = "fs_mid"
    FS_SPHERE = "fs_sphere"


class AnatPipeOut(Enum):
    """
    Enum for anatomical pipeline steps
    """

    RAW = "raw"
    FREESURFER = "freesurfer"
    FREESURFER_SDIR = "freesurfer_sdir"
    MISC = "misc"
    FSLR = "fslr"


class FuncPipeOut(Enum):
    """
    Enum for full functional pipeline steps - from raw
    nifti data to surface-based, smoothed GIFTI data
    """

    RAW = "raw"
    TRIM = "trim"
    MOTION_CORRECT = "motion"
    BBREGISTER = "coregister"
    SLICE_TIMING = "slicetime"
    TEDANA = "tedana"
    VOL2SURF = "vol2surf"  # vol2surf performs surface smoothing
    FSLR_RESAMPLE = "surface_lr"
    MISC = "misc"


class FuncPipeOutMisc(Enum):
    """
    Enum for miscellaneous functional pipeline outputs
    """

    MEAN_VOL = "mean_vol"
    MCFLIRT_MAT = "mcflirt_mat"
    SURFREGISTER_MAT = "surfregister_mat"
    SURFREGISTER_FSL_MAT = "surfregister_fslmat"
    SURFREGISTER_REG = "surfregister_reg"
    FS_FUNC_MASK = "fs_func_mask"
    TEDANA_DENOISED = "tedana_denoised"


class FuncPipeOutCiftiPartial(Enum):
    """
    Enum for partial cifti functional pipeline steps - from fMRIPrep
    CIFTI-2 data to surface-based CIFTI-2 data
    """

    RAW = "raw"
    TRIM = "trim"
    SURFACE_SMOOTH = "surface_smooth"


class PhysioPipeOut(Enum):
    """
    Enum for physio pipeline steps
    """

    RAW = "raw"
    PROC = "proc"
    SIGNAL = "signal"


class FileMapper:
    """
    Utility class for mapping input and output files for
    anatomical and functional and physio

    Attributes
    ----------
    dataset: str
        dataset label
    params: dict
        dataset params

    Methods
    -------
    map():
        map all input and output files for each scan

    """

    # get participant iterator
    def __init__(self, dataset: Literal["vanderbilt", "newcastle"], params: dict):
        self.dataset = dataset
        self.params = params
        # initialize participant iterator
        self.iter = Participant(self.dataset)

    def map(self) -> dict:
        """
        define output paths for all preprocessing outputs and create directories for storing them

        Returns
        -------
        file_map: str
            dictionary containing file paths to all preprocessing output
        """
        # create output directories
        dir_map = self._create_directories()
        # store directory paths
        file_map = {"anat": {}, "func": {}, "physio": {}}

        # loop through subjects then sessions
        for subj, ses_list in self.iter.subject_by_session():
            # initialize file maps
            file_map["anat"][subj] = {}
            file_map["func"][subj] = {}
            file_map["physio"][subj] = {}

            if "anat" in self.params["data"]:
                self._map_anat(subj, dir_map["anat"], file_map["anat"][subj])

            # loop through sessions and create directories
            for ses in ses_list:
                file_map["func"][subj][ses] = {}
                file_map["physio"][subj][ses] = {}
                if self.params["func"]["pipeline"] == "full":
                    self._map_func_full(
                        subj, ses, dir_map["func"], file_map["func"][subj][ses]
                    )
                elif self.params["func"]["pipeline"] == "cifti-partial":
                    self._map_func_cifti_partial(
                        subj, ses, dir_map["func"], file_map["func"][subj][ses]
                    )
                if "physio" in self.params["data"]:
                    self._map_physio(
                        subj,
                        ses,
                        dir_map["physio"],
                        dir_map["func"],
                        file_map["physio"][subj][ses],
                    )

        return file_map

    def _map_anat(
        self, subj: str, dir_anat: dict[str, str], file_map_subj: dict
    ) -> None:
        """
        create output files for anatomical pipeline. Modify in-place
        """
        # raw anatomical
        file_map_subj[AnatPipeOut.RAW.value] = self._filepath(
            data="anat", subject=subj, basedir=dir_anat[AnatPipeOut.RAW.value]
        )
        # freesurfer out dir
        file_map_subj[AnatPipeOut.FREESURFER.value] = os.path.abspath(
            f"{dir_anat[AnatPipeOut.FREESURFER.value]}"
        )
        # freesurfer subject out dir
        file_map_subj[AnatPipeOut.FREESURFER_SDIR.value] = os.path.abspath(
            f"{dir_anat[AnatPipeOut.FREESURFER.value]}/{subj}"
        )

        # miscellanous anat outputs
        file_map_subj[AnatPipeOut.MISC.value] = {}
        ## freesurfer brain mask
        file_map_subj[AnatPipeOut.MISC.value][AnatPipeOutMisc.FS_BRAINMASK.value] = (
            os.path.abspath(
                f"{file_map_subj[AnatPipeOut.FREESURFER_SDIR.value]}/mri/brainmask.mgz"
            )
        )
        ## freesurfer midthickness
        fslr_fp = utils.get_fp_base(
            self._filepath(
                data="anat", subject=subj, basedir=dir_anat[AnatPipeOut.FSLR.value]
            )
        )
        ### create files for each hemisphere (32k res)
        for hemi, hemi_lr in zip(["lh", "rh"], ["L", "R"]):
            file_map_subj[AnatPipeOut.MISC.value][f"fs_mid_{hemi}"] = (
                f"{fslr_fp}.{hemi}.midthickness.surf.gii"
            )
            file_map_subj[AnatPipeOut.MISC.value][f"fslr_mid_{hemi}"] = (
                f"{fslr_fp}.{hemi_lr}.midthickness.32k_fs_LR.surf.gii"
            )
            file_map_subj[AnatPipeOut.MISC.value][f"fs_sphere_{hemi}"] = (
                f"{fslr_fp}.{hemi}.sphere.reg.surf.gii"
            )

    def _map_func_cifti_partial(
        self, subj: str, ses: str, dir_func: dict[str, str], file_map_subj: dict
    ) -> None:
        """
        create output files for partial cifti functional pipeline. Modify in-place
        """
        # map raw file
        file_map_subj[FuncPipeOutCiftiPartial.RAW.value] = self._filepath(
            data="func",
            subject=subj,
            session=ses,
            basedir=dir_func[FuncPipeOutCiftiPartial.RAW.value],
            file_ext="dtseries.nii",
        )
        # map trimmed file
        file_map_subj[FuncPipeOutCiftiPartial.TRIM.value] = self._filepath(
            data="func",
            subject=subj,
            session=ses,
            basedir=dir_func[FuncPipeOutCiftiPartial.TRIM.value],
            file_ext="dtseries.nii",
        )
        # map surface smooth file
        file_map_subj[FuncPipeOutCiftiPartial.SURFACE_SMOOTH.value] = self._filepath(
            data="func",
            subject=subj,
            session=ses,
            basedir=dir_func[FuncPipeOutCiftiPartial.SURFACE_SMOOTH.value],
            file_ext="dtseries.nii",
        )

    def _map_func_full(
        self, subj: str, ses: str, dir_func: dict[str, str], file_map_subj: dict
    ) -> None:
        """
        create output files for full functional pipeline. Modify in-place
        """
        # filepaths for tasks with gifti output
        # map files differently for multi- vs. single-echo datasets
        tasks_start = [
            FuncPipeOut.RAW.value,
            FuncPipeOut.TRIM.value,
            FuncPipeOut.SLICE_TIMING.value,
            FuncPipeOut.MOTION_CORRECT.value,
        ]
        for task in tasks_start:
            file_map_subj[task] = self._filepath(
                data="func",
                subject=subj,
                session=ses,
                basedir=dir_func[task],
                file_ext="nii.gz",
                return_echos=True,
            )
        # coregistration file handling is same for multi- vs. single-echo
        file_map_subj[FuncPipeOut.BBREGISTER.value] = self._filepath(
            data="func",
            subject=subj,
            session=ses,
            basedir=dir_func[FuncPipeOut.BBREGISTER.value],
            file_ext="nii.gz",
        )
        # separate .gii outputs per hemisphere
        tasks_surface = [FuncPipeOut.VOL2SURF.value, FuncPipeOut.FSLR_RESAMPLE.value]
        for task in tasks_surface:
            file_map_subj[task] = {}
            for hemi in ["lh", "rh"]:
                file_map_subj[task][hemi] = self._filepath(
                    data="func",
                    subject=subj,
                    session=ses,
                    basedir=dir_func[task],
                    file_ext=f"{hemi}.func.gii",  # type: ignore
                )

        # tedana out directory
        file_map_subj[FuncPipeOut.TEDANA.value] = utils.get_fp_base(
            self._filepath(
                data="func",
                subject=subj,
                session=ses,
                basedir=dir_func[FuncPipeOut.TEDANA.value],
                file_ext="nii.gz",
            )
        )
        # miscellanous functional outputs
        file_map_subj[FuncPipeOut.MISC.value] = {}
        ## mcflirt mean volume
        motion_fp = file_map_subj[FuncPipeOut.MOTION_CORRECT.value]
        if self.params["multiecho"]:
            motion_fp_base = utils.get_fp_base(motion_fp[0])
        else:
            motion_fp_base = utils.get_fp_base(motion_fp)
        file_map_subj[FuncPipeOut.MISC.value][FuncPipeOutMisc.MEAN_VOL.value] = (
            os.path.abspath(f"{motion_fp_base}_mean_reg.nii.gz")
        )
        ## mcflirt transform matrix
        file_map_subj[FuncPipeOut.MISC.value][FuncPipeOutMisc.MCFLIRT_MAT.value] = (
            os.path.abspath(f"{motion_fp_base}.mat")
        )

        ## coregistration transform matrix
        coreg_fp = utils.get_fp_base(file_map_subj[FuncPipeOut.BBREGISTER.value])
        file_map_subj[FuncPipeOut.MISC.value][
            FuncPipeOutMisc.SURFREGISTER_MAT.value
        ] = os.path.abspath(f"{coreg_fp}.lta")
        ## coregistration inital FLIRT transform matrix
        file_map_subj[FuncPipeOut.MISC.value][
            FuncPipeOutMisc.SURFREGISTER_FSL_MAT.value
        ] = os.path.abspath(f"{coreg_fp}.mat")
        ## coregistration reg file
        file_map_subj[FuncPipeOut.MISC.value][
            FuncPipeOutMisc.SURFREGISTER_REG.value
        ] = os.path.abspath(f"{coreg_fp}_mean_reg.nii.gz")
        ## mask file for tedana preprocessing
        file_map_subj[FuncPipeOut.MISC.value][FuncPipeOutMisc.FS_FUNC_MASK.value] = (
            os.path.abspath(f"{coreg_fp}_brainmask.nii.gz")
        )
        ## tedana denoised file
        tedana_out = file_map_subj[FuncPipeOut.TEDANA.value]
        tedana_out_fp = os.path.basename(file_map_subj[FuncPipeOut.TEDANA.value])
        file_map_subj[FuncPipeOut.MISC.value][FuncPipeOutMisc.TEDANA_DENOISED.value] = (
            os.path.abspath(f"{tedana_out}/{tedana_out_fp}_desc-denoised_bold.nii.gz")
        )

    def _map_physio(
        self,
        subj: str,
        ses: str,
        dir_physio: dict[str, str],
        dir_func: dict[str, str],
        file_map_subj: dict,
    ) -> None:
        """
        create output files for physio pipeline. Modify in-place
        """
        # map raw files
        file_map_subj[PhysioPipeOut.RAW.value] = {}
        for p in self.params["physio"]["raw"]:
            if p == "motion":
                file_map_subj[PhysioPipeOut.RAW.value][p] = self._filepath(
                    data="physio",
                    subject=subj,
                    session=ses,
                    basedir=dir_func[FuncPipeOut.MOTION_CORRECT.value],
                    physio_type="raw",
                    physio_str=p,
                    file_ext=self.params["physio_ext"][p],
                )
            else:
                file_map_subj[PhysioPipeOut.RAW.value][p] = self._filepath(
                    data="physio",
                    subject=subj,
                    session=ses,
                    basedir=dir_physio[PhysioPipeOut.RAW.value],
                    physio_type="raw",
                    physio_str=p,
                    file_ext=self.params["physio_ext"][p],
                )

        # map physio signals extracted from raw physio files
        file_map_subj[PhysioPipeOut.SIGNAL.value] = {}
        for p in self.params["physio"]["signals"]:
            file_map_subj[PhysioPipeOut.SIGNAL.value][p] = self._filepath(
                data="physio",
                subject=subj,
                session=ses,
                basedir=dir_physio[PhysioPipeOut.PROC.value],
                physio_type="out",
                physio_str=p,
                file_ext="txt",
            )
        # map physio signal outputs
        file_map_subj[PhysioPipeOut.PROC.value] = {}
        # loop over physio signals
        for p in self.params["physio"]["out"]:
            # loop over physio signal extraction outputs
            for p_out in self.params["physio"]["out"][p]:
                file_map_subj[PhysioPipeOut.PROC.value][p_out] = self._filepath(
                    data="physio",
                    subject=subj,
                    session=ses,
                    basedir=dir_physio[PhysioPipeOut.PROC.value],
                    physio_type="out",
                    physio_str=p_out,
                    file_ext="txt",
                )

    def _create_directories(self) -> dict:
        """
        create output directories and store
        paths in dictionary

        Returns
        -------
        directory_map: dict
            dictionary of filepaths to output directories
        """
        # initialize directories
        directory_map = {}
        # anatomical directory
        if "anat" in self.params["data"]:
            anat_dir = self.params["directory"]["anat"]
            directory_map["anat"] = {
                AnatPipeOut.RAW.value: f"{anat_dir}/raw",
                AnatPipeOut.FREESURFER.value: f"{anat_dir}/freesurfer",
                AnatPipeOut.FSLR.value: f"{anat_dir}/proc1_fslr",
            }
            # create directories
            for d in directory_map["anat"]:
                os.makedirs(directory_map["anat"][d], exist_ok=True)

        # physio directory
        if "physio" in self.params["data"]:
            physio_dir = self.params["directory"]["physio"]
            directory_map["physio"] = {
                PhysioPipeOut.RAW.value: f"{physio_dir}/raw",
                PhysioPipeOut.PROC.value: f"{physio_dir}/proc1_physio",
            }
            # create directories
            for d in directory_map["physio"]:
                os.makedirs(directory_map["physio"][d], exist_ok=True)

        # functional directory
        func_dir = self.params["directory"]["func"]

        if self.params["func"]["pipeline"] == "full":
            directory_map["func"] = {
                FuncPipeOut.RAW.value: f"{func_dir}/raw",
                FuncPipeOut.TRIM.value: f"{func_dir}/proc1_trim",
                FuncPipeOut.SLICE_TIMING.value: f"{func_dir}/proc2_slicetime",
                FuncPipeOut.MOTION_CORRECT.value: f"{func_dir}/proc3_motion",
                FuncPipeOut.BBREGISTER.value: f"{func_dir}/proc4_coregister",
                FuncPipeOut.VOL2SURF.value: f"{func_dir}/proc5_vol2surf",
                FuncPipeOut.FSLR_RESAMPLE.value: f"{func_dir}/proc6_surfacelr",
            }

            # include tedana directory, if multiecho dataset
            if self.params["multiecho"]:
                directory_map["func"][FuncPipeOut.TEDANA.value] = (
                    f"{func_dir}/proc_tedana"
                )

        elif self.params["func"]["pipeline"] == "cifti-partial":
            directory_map["func"] = {
                FuncPipeOutCiftiPartial.RAW.value: f"{func_dir}/raw",
                FuncPipeOutCiftiPartial.TRIM.value: f"{func_dir}/proc1_trim",
                FuncPipeOutCiftiPartial.SURFACE_SMOOTH.value: f"{func_dir}/proc2_surfacesmooth",
            }

        # create directories, if doesn't exist
        for d in directory_map["func"]:
            os.makedirs(directory_map["func"][d], exist_ok=True)

        return directory_map

    def _filepath(
        self,
        data: Literal["func", "anat", "physio"],
        basedir: str,
        subject: str,
        file_ext: utils.FileExtParams | None = None,
        return_echos: bool = False,
        physio_str: str | None = None,
        physio_type: Literal["raw", "out"] | None = None,
        session: str | None = None,
    ) -> str:
        """
        take parameters and return file path

        Parameters
        ----------
        data: Literal['func', 'anat', 'physio']
            Choice of data modality
        basedir: str
            prepend filepath to directory
        subject: str
            subject label
        file_ext: utils.FileExtParams,
            the file extension for the file path (optional)
        return_echos: bool
            whether to return filepaths for individual echos (default: False)
        physio_str: str
            physio label (optional)
        physio_type: Literal['raw', 'out']
            type of physio file to return (optional)
        session: str
            session label (optional)

        Returns
        -------
        fp: str
            file path

        """
        if return_echos:
            fp = [
                self.iter.to_file(
                    data=data,
                    subject=subject,
                    session=session,
                    basedir=basedir,
                    echo=e + 1,
                    file_ext="nii.gz",
                )
                for e in range(len(self.params["func"]["echos"]))
            ]
            # get full path
            fp = [os.path.abspath(f) for f in fp]
            if len(fp) < 2:
                fp = fp[0]
        elif data == "physio":
            if physio_type == "raw":
                fp = self.iter.to_file(
                    data=data,
                    subject=subject,
                    session=session,
                    basedir=basedir,
                    physio=physio_str,
                    file_ext=file_ext,
                    physio_type="raw",
                )
            elif physio_type == "out":
                fp = self.iter.to_file(
                    data=data,
                    subject=subject,
                    session=session,
                    basedir=basedir,
                    physio=physio_str,
                    file_ext="txt",
                    physio_type="out",
                )
            # get full path
            fp = os.path.abspath(fp)
        else:
            fp = self.iter.to_file(
                data=data,
                subject=subject,
                session=session,
                basedir=basedir,
                file_ext=file_ext,
            )
            # get full path
            fp = os.path.abspath(fp)

        return fp  # type: ignore


class Pipeline:
    """
    Master pipeline for orchestrating preprocessing of functional,
    anatomical (T1w), physiological and EEG data.
    Dataset specific parameters are set in scan/meta/params.json.

    Attributes
    ----------
    dataset: str
        chosen dataset
    anat_skip: bool
        whether to skip anatomical preprocessing, if already run
        (default = False)
    reconall_skip: bool
        whether to skip reconall preprocessing, if already run
        (default: False)
    func_skip: bool
        whether to skip functional preprocessing, if already run
        (default = False)
    physio_skip: bool
        whether to skip physio preprocessing, if already run
        (default = False)
    Methods
    -------

    run():
        Iterate through scans sequentially and execute preprocessing

    run_parallel(n_cores: int):
        Iterate through scans in parallel and execute preprocessing
    """

    def __init__(
        self,
        dataset: Literal["vanderbilt"],
        anat_skip: bool = False,
        func_skip: bool = False,
        reconall_skip: bool = False,
        physio_skip: bool = False,
    ):
        self.dataset = dataset
        # dataset specific parameters
        with open("scan/meta/params.json", "rb") as f:
            self.params = json.load(f)[dataset]
        self.multiecho = self.params["multiecho"]
        # if multiecho, get echos
        if self.multiecho:
            self.echos = self.params["func"]["echos"]
        else:
            self.echos = [None]
        # get mapping to output files
        self.file_iter = FileMapper(dataset, self.params)
        self.file_map = self.file_iter.map()

        # set skip flags
        # check if anat is available
        if "anat" in self.params["data"]:
            self.anat_skip = anat_skip
        else:
            print(f"{self.dataset} does not have anatomical data, skipping")
            self.anat_skip = True
        # check if func is available
        if "func" in self.params["data"]:
            self.func_skip = func_skip
        else:
            print(f"{self.dataset} does not have functional data, skipping")
            self.func_skip = True
        # check if physio is available
        if "physio" in self.params["data"]:
            self.physio_skip = physio_skip
        else:
            print(f"{self.dataset} does not have physiological data, skipping")
            self.physio_skip = True

        # set reconall skip flag
        if self.anat_skip:
            self.reconall_skip = True
        else:
            self.reconall_skip = reconall_skip

        # set func type
        self.func_pipe_type = self.params["func"]["pipeline"]

    def run(self) -> None:
        """
        Execute full preprocessing pipeline sequentially over all subjects
        """
        # loop through subjects then sessions
        for subj, ses_list in self.file_iter.iter.subject_by_session():
            if not self.anat_skip:
                # initialized and execute anatomical pipeline
                anat_proc = AnatomicalPipeline(
                    subj=subj,
                    params=self.params,
                    fmap=self.file_map,
                    reconall_skip=self.reconall_skip,
                )
                anat_proc.run()
            # loop through sessions and preprocess functionals and physio
            for ses in ses_list:
                print(f"Preprocessing subject {subj}, session {ses}")
                if not self.func_skip:
                    if self.func_pipe_type == "full":
                        # preprocess functional
                        func_proc = FunctionalPipelineFull(
                            subj=subj, ses=ses, params=self.params, fmap=self.file_map
                        )
                    elif self.func_pipe_type == "cifti-partial":
                        func_proc = FunctionalPipelineCiftiPartial(
                            subj=subj, ses=ses, params=self.params, fmap=self.file_map
                        )
                    else:
                        raise ValueError(
                            f"{self.func_pipe_type} is not a valid function pipeline type"
                        )
                    # preprocess functional
                    func_proc.run()

                # preprocess physio
                if not self.physio_skip:
                    physio_proc = PhysioPipeline(
                        dataset=self.dataset,
                        subj=subj,
                        ses=ses,
                        params=self.params,
                        fmap=self.file_map,
                    )
                    physio_proc.run()

    def run_parallel(self, n_cores: int = 4) -> None:
        """
        Execute full preprocessing pipeline in parallel over all subjects

        Parameters
        ----------
        n_cores: ints
            number of cores to run in parallel
        """
        # loop through subjects then sessions
        anat_proc_params = []
        func_proc_params = []
        physio_proc_params = []
        for subj, ses_list in self.file_iter.iter.subject_by_session():
            # initialized and execute anatomical pipeline
            anat_p = {
                "subj": subj,
                "params": self.params,
                "fmap": self.file_map,
                "reconall_skip": self.reconall_skip,
            }
            anat_proc_params.append(anat_p)
            # loop through sessions and preprocess functionals and physio
            for ses in ses_list:
                # functional parameters
                func_p = {
                    "subj": subj,
                    "ses": ses,
                    "params": self.params,
                    "fmap": self.file_map,
                }
                func_proc_params.append(func_p)
                # physio parameters
                physio_p = {
                    "dataset": self.dataset,
                    "subj": subj,
                    "ses": ses,
                    "params": self.params,
                    "fmap": self.file_map,
                }
                physio_proc_params.append(physio_p)

        # parallel execution
        # define preproc execution function
        # execute anatomical pipeline
        if not self.anat_skip:
            with Pool(n_cores) as pool:
                pool.map(_par_execute_anat, anat_proc_params)

        # execute functional pipeline
        if not self.func_skip:
            with Pool(n_cores) as pool:
                # Create a list of tuples with (params, func_pipe_type) for each process
                func_args = [
                    (params, self.func_pipe_type) for params in func_proc_params
                ]
                pool.starmap(_par_execute_func, func_args)

        # execute physio pipeline
        if not self.physio_skip:
            with Pool(n_cores) as pool:
                pool.map(_par_execute_physio, physio_proc_params)


class AnatomicalPipeline:
    """
    Freesurfer recon-all T1w preprocessing

    Attributes
    ----------
    subj: str
        subject label
    params: dict
        dictionary containing dataset specific
        preprocessing parameters from scan/meta/params.json
    fmap: dict
        dictionary containing output file paths
    reconall_skip: bool
        whether to skip reconall preprocessing, if already run
        (default: False)

    Methods
    -------
    run():
        Execute anatomical preprocessing pipeline

    """

    def __init__(
        self, subj: str, params: dict, fmap: dict, reconall_skip: bool = False
    ):
        self.subj = subj
        self.fmap = fmap
        self.reconall_skip = reconall_skip
        # get anatomical preprocessing params
        self.anat_params = params["anat"]
        # get anat directory
        self.anat_dir = params["directory"]["anat"]

    def run(self) -> None:
        """
        Execute anatomical preprocessing pipeline
        """
        # freesurfer recon-all
        if not self.reconall_skip:
            fs.recon_all(
                t1=self.fmap["anat"][self.subj][AnatPipeOut.RAW.value],
                subj_label=self.subj,
                dir_out=self.fmap["anat"][self.subj][AnatPipeOut.FREESURFER.value],
            )
        # create midthickness files for left and right hemispheres
        for hemi in ["lh", "rh"]:
            wb.create_midthickness(
                hemi=hemi,  # type: ignore
                fs_mid=self.fmap["anat"][self.subj][AnatPipeOut.MISC.value][
                    f"fs_mid_{hemi}"
                ],
                lr_mid=self.fmap["anat"][self.subj][AnatPipeOut.MISC.value][
                    f"fslr_mid_{hemi}"
                ],
                sphere_out=self.fmap["anat"][self.subj][AnatPipeOut.MISC.value][
                    f"fs_sphere_{hemi}"
                ],
                fs_subj_dir=self.fmap["anat"][self.subj][
                    AnatPipeOut.FREESURFER_SDIR.value
                ],
            )


class FunctionalPipelineFull:
    """
    Full functional preprocessing pipeline for a single scan. Process raw nifti
    file to surface-smoothed, fsLR-resampled file.

    Steps:
        1. Trimming
        2. Slice-timing correction
        3. Head motion correction (mcflirt)
        4. Coregistration of mean functional volume to T1w (bbregister)
        5. If multiecho, tedana preprocessing
        6. Volume-to-surface sampling and smoothing (mri_vol2surf)
        7. Resampling to fsLR space (wb_command -metric-resample)

    Attributes
    ----------
    subj: str
        subject label
    ses: str
        session label
    params: dict
        dictionary containing dataset specific
        preprocessing parameters from scan/meta/params.json
    fmap: dict
        dictionary containing output file paths

    Methods
    -------
    run():
        Execute functional preprocessing pipeline

    """

    def __init__(
        self,
        subj: str,
        ses: str,
        params: dict,
        fmap: dict,
    ):
        self.subj = subj
        self.ses = ses
        self.fmap = fmap
        # get directory paths
        self.func_dir = params["directory"]["func"]
        # get functional preprocessing params
        self.func_params = params["func"]
        # check whether multiecho dataset
        self.multiecho = params["multiecho"]

    def run(self) -> None:
        """
        Execute functional preprocessing pipeline
        """
        # trim first N volumes
        if self.multiecho:
            fsl.trim_vol_multiecho(
                fps=self.fmap["func"][self.subj][self.ses][FuncPipeOut.RAW.value],
                fps_out=self.fmap["func"][self.subj][self.ses][FuncPipeOut.TRIM.value],
                n_trim=self.func_params["trim"],
            )
        else:
            fsl.trim_vol(
                fp=self.fmap["func"][self.subj][self.ses][FuncPipeOut.RAW.value],
                fp_out=self.fmap["func"][self.subj][self.ses][FuncPipeOut.TRIM.value],
                n_trim=self.func_params["trim"],
            )

        # slicetiming correction w/ FSL slicetimer
        if self.multiecho:
            fsl.slicetime_multiecho(
                fps=self.fmap["func"][self.subj][self.ses][FuncPipeOut.TRIM.value],
                fps_out=self.fmap["func"][self.subj][self.ses][
                    FuncPipeOut.SLICE_TIMING.value
                ],
                slice_order=self.func_params["sliceorder"],
                tr=self.func_params["tr"],
            )
        else:
            fsl.slicetime(
                fp=self.fmap["func"][self.subj][self.ses][FuncPipeOut.TRIM.value],
                fp_out=self.fmap["func"][self.subj][self.ses][
                    FuncPipeOut.SLICE_TIMING.value
                ],
                slice_order=self.func_params["sliceorder"],
                tr=self.func_params["tr"],
            )

        # motion correction with FSL MCLFIRT
        if self.multiecho:
            fsl.mcflirt_multiecho(
                fps=self.fmap["func"][self.subj][self.ses][
                    FuncPipeOut.SLICE_TIMING.value
                ],
                fps_out=self.fmap["func"][self.subj][self.ses][
                    FuncPipeOut.MOTION_CORRECT.value
                ],
                fp_meanvol=self.fmap["func"][self.subj][self.ses][
                    FuncPipeOut.MISC.value
                ][FuncPipeOutMisc.MEAN_VOL.value],
                fp_mat=self.fmap["func"][self.subj][self.ses][FuncPipeOut.MISC.value][
                    FuncPipeOutMisc.MCFLIRT_MAT.value
                ],
            )
        else:
            fsl.mcflirt(
                fp=self.fmap["func"][self.subj][self.ses][
                    FuncPipeOut.SLICE_TIMING.value
                ],
                fp_out=self.fmap["func"][self.subj][self.ses][
                    FuncPipeOut.MOTION_CORRECT.value
                ],
            )

        # coregistration of mean functional to T1w with Freesurfer BBRegister
        fs.surf_register(
            fp_reg_out=self.fmap["func"][self.subj][self.ses][FuncPipeOut.MISC.value][
                FuncPipeOutMisc.SURFREGISTER_REG.value
            ],
            fp_mat_out=self.fmap["func"][self.subj][self.ses][FuncPipeOut.MISC.value][
                FuncPipeOutMisc.SURFREGISTER_MAT.value
            ],
            fp_fslmat_out=self.fmap["func"][self.subj][self.ses][
                FuncPipeOut.MISC.value
            ][FuncPipeOutMisc.SURFREGISTER_FSL_MAT.value],
            func_mean=self.fmap["func"][self.subj][self.ses][FuncPipeOut.MISC.value][
                FuncPipeOutMisc.MEAN_VOL.value
            ],
            subj=self.subj,
            subjects_dir=self.fmap["anat"][self.subj][AnatPipeOut.FREESURFER.value],
        )

        # if multiecho, run tedana pipeline
        if self.multiecho:
            # create binary brain mask from Freesurfer in functional space
            fs.mask_to_func(
                mask=self.fmap["anat"][self.subj][AnatPipeOut.MISC.value][
                    AnatPipeOutMisc.FS_BRAINMASK.value
                ],
                func_mean=self.fmap["func"][self.subj][self.ses][
                    FuncPipeOut.MISC.value
                ][FuncPipeOutMisc.MEAN_VOL.value],
                mask_out=self.fmap["func"][self.subj][self.ses][FuncPipeOut.MISC.value][
                    FuncPipeOutMisc.FS_FUNC_MASK.value
                ],
                mat_lta=self.fmap["func"][self.subj][self.ses][FuncPipeOut.MISC.value][
                    FuncPipeOutMisc.SURFREGISTER_MAT.value
                ],
            )
            # run tedana workflow
            tedana_prefix = os.path.basename(
                self.fmap["func"][self.subj][self.ses][FuncPipeOut.TEDANA.value]
            )
            tedana_denoise(
                fps_in=self.fmap["func"][self.subj][self.ses][
                    FuncPipeOut.MOTION_CORRECT.value
                ],
                echo_times=self.func_params["echos"],
                mask=self.fmap["func"][self.subj][self.ses][FuncPipeOut.MISC.value][
                    FuncPipeOutMisc.FS_FUNC_MASK.value
                ],
                out_dir=self.fmap["func"][self.subj][self.ses][
                    FuncPipeOut.TEDANA.value
                ],
                out_prefix=tedana_prefix,
            )
            fp_vol2surf = self.fmap["func"][self.subj][self.ses][
                FuncPipeOut.MISC.value
            ][FuncPipeOutMisc.TEDANA_DENOISED.value]
        else:
            fp_vol2surf = self.fmap["func"][self.subj][self.ses][
                FuncPipeOut.MOTION_CORRECT.value
            ]

        # from here on, we process separate hemispheres (lh, rh)
        for hemi in ["lh", "rh"]:
            # from volume to native (subject) surface with Freesurfer
            # mri_vol2surf, additionaly performs surface-smoothing
            fs.vol2surf(
                hemi=hemi,  # type: ignore
                smooth_fwhm=self.func_params["smooth_fwhm"],
                fp_in=fp_vol2surf,
                fp_out=self.fmap["func"][self.subj][self.ses][
                    FuncPipeOut.VOL2SURF.value
                ][hemi],
                subj=self.subj,
                subjects_dir=self.fmap["anat"][self.subj][AnatPipeOut.FREESURFER.value],
                fp_reg_mat=self.fmap["func"][self.subj][self.ses][
                    FuncPipeOut.MISC.value
                ][FuncPipeOutMisc.SURFREGISTER_MAT.value],
            )
            # freesurfer native surface to fs_LR surface with
            # workbench metric-resample
            wb.fs2fslr(
                hemi=hemi,
                fp_in=self.fmap["func"][self.subj][self.ses][
                    FuncPipeOut.VOL2SURF.value
                ][hemi],
                fp_out=self.fmap["func"][self.subj][self.ses][
                    FuncPipeOut.FSLR_RESAMPLE.value
                ][hemi],
                fs_mid=self.fmap["anat"][self.subj][AnatPipeOut.MISC.value][
                    f"fs_mid_{hemi}"
                ],
                lr_mid=self.fmap["anat"][self.subj][AnatPipeOut.MISC.value][
                    f"fslr_mid_{hemi}"
                ],
                fs_sphere=self.fmap["anat"][self.subj][AnatPipeOut.MISC.value][
                    f"fs_sphere_{hemi}"
                ],
                fs_subj_dir=self.fmap["anat"][self.subj][
                    AnatPipeOut.FREESURFER_SDIR.value
                ],
            )


class FunctionalPipelineCiftiPartial:
    """
    Partial functional preprocessing pipeline for an fMRIPrep preprocessed CIFTI scan.
    Process CIFTI image with trimming and surfaces-smoothing.

    Steps:
        1. Trimming
        2. Surface smoothing (wb_command -cifti-smoothing)

    Attributes
    ----------
    subj: str
        subject label
    ses: str
        session label
    params: dict
        dictionary containing dataset specific
        preprocessing parameters from scan/meta/params.json
    fmap: dict
        dictionary containing output file paths

    Methods
    -------
    run():
        Execute cifti-partial functional preprocessing pipeline

    """

    def __init__(
        self,
        subj: str,
        ses: str,
        params: dict,
        fmap: dict,
    ):
        self.subj = subj
        self.ses = ses
        self.fmap = fmap
        # get directory paths
        self.func_dir = params["directory"]["func"]
        # get functional preprocessing params
        self.func_params = params["func"]

    def run(self) -> None:
        """
        Execute cifti-partial functional preprocessing pipeline
        """
        # trim first N volumes
        trim_cifti(
            fp=self.fmap["func"][self.subj][self.ses][
                FuncPipeOutCiftiPartial.RAW.value
            ],
            fp_out=self.fmap["func"][self.subj][self.ses][
                FuncPipeOutCiftiPartial.TRIM.value
            ],
            n_trim=self.func_params["trim"],
        )
        # surface smoothing
        wb.cifti_smooth(
            fp_in=self.fmap["func"][self.subj][self.ses][
                FuncPipeOutCiftiPartial.TRIM.value
            ],
            fp_out=self.fmap["func"][self.subj][self.ses][
                FuncPipeOutCiftiPartial.SURFACE_SMOOTH.value
            ],
            fwhm=self.func_params["smooth_fwhm"],
        )


class PhysioPipeline:
    """
    Physio (including EEG) preprocessing pipeline for a single scan.

    Steps of the preprocessing depend on the dataset and the physio
    signals recorded, but in general:

    1. Load physio signals (and EEG) - dataset-specific
    2. Extract physio signal features - e.g. respiratory amplitude
    3. Trim physio signals to match trimming of functional volumes
    4. Detrending (polynomial order 2 - quadratic) - except for eeg_vigilance
    5. Low-pass filter (<0.20 Hz) - except for weight
    6. Interpolation to functional MRI volumes

    Attributes
    ----------
    dataset: str
        dataset label. Some aspects of physio preprocessing are
        dataset specific.
    subj: str
        subject label
    ses: str
        session label
    params: dict
        dictionary containing dataset specific
        preprocessing parameters from scan/meta/params.json
    fmap: dict
        dictionary containing output file paths
    plot_physio: bool
        whether to save plot of raw physio signals for QC (before
        preprocessing)

    Methods
    -------
    run():
        Execute physio preprocessing pipeline
    """

    STEPS = ["trim", "detrend", "lowpass", "resample"]

    def __init__(
        self,
        dataset: str,
        subj: str,
        ses: str,
        params: dict,
        fmap: dict,
        plot_physio: bool = True,
    ):
        self.dataset = dataset
        self.subj = subj
        self.ses = ses
        self.params = params
        self.fmap = fmap
        self.plot_physio = plot_physio
        # calculate # of secs to trim off physio to match functional mri trim
        self.trim = params["func"]["tr"] * params["func"]["trim"]
        # calculate interpolation time points to sample physio to functional
        self.func_t = self._calc_frame_times()
        # create physio function mapping
        self.physio_func_map = physio_func_map(self.params["physio"]["signals"])

    def run(self) -> None:
        """
        Execute physio preprocessing pipeline
        """
        # load physio
        signals, sf = self._load_physio()
        # perform physio preprocessing
        for p in self.params["physio"]["signals"]:
            # extract physio signal features from raw physio signals
            signals_extract = self._physio_extract(signals[p], sf[p], p)
            # account for trimming of functional volumes
            trim_n = int(sf[p] * self.trim)
            # loop over extracted physio signal features
            for physio_out, signal in signals_extract.items():
                pipeline_steps = self._set_pipeline_steps(physio_out)
                # trim physio to functional length (account for removal of dummy scans)
                if pipeline_steps["trim"]:
                    signal_proc = signal[trim_n:]
                else:
                    signal_proc = signal

                # polynomial (2nd order) detrending
                if pipeline_steps["detrend"]:
                    signal_proc = nk.signal_detrend(
                        signal_proc, method="polynomial", order=2
                    )
                # low-pass filtering
                if pipeline_steps["lowpass"]:
                    signal_proc = nk.signal_filter(
                        signal_proc, sampling_rate=sf[p], highcut=0.20, order=5
                    )
                # resampling via cubic interpolation to functional scan volumes
                # for signals other than weight, signal should be low-passed
                if pipeline_steps["resample"]:
                    # set interpolation method to 'nearest' for sample weight
                    if physio_out == "weight":
                        # if sample weights, use nearest-neighbour interpolation
                        interp_method = "nearest"
                    else:
                        interp_method = "cubic"
                    # resample physio
                    signal_proc = self._resample_physio(
                        signal_proc, sf[p], interp_method=interp_method
                    )
                # save out physio in .txt format
                np.savetxt(
                    self.fmap["physio"][self.subj][self.ses]["proc"][physio_out],
                    signal_proc,
                )

    def _calc_frame_times(self) -> np.ndarray:
        """
        Calculate interpolation time points from physio to functional samples.
        FSL slicetimer aligns all functional slices to the middle of the
        TR (0.5 * TR), so time points should be selected with this in mind.

        Returns
        -------
        frame_times: np.ndarray
            time points (from the start of the functional scan) of preprocessed
            functional volumes to interpolate physio signals to

        """
        # get functional TR
        func_tr = self.params["func"]["tr"]
        # load preprocessed functional scan to get number of volumes
        if self.params["func"]["pipeline"] == "full":
            gii_lh = nb.load(  # type: ignore
                self.fmap["func"][self.subj][self.ses]["surface_lr"]["lh"]
            )
            func_len = len(gii_lh.darrays)  # type: ignore
        elif self.params["func"]["pipeline"] == "cifti-partial":
            cifti = nb.load(  # type: ignore
                self.fmap["func"][self.subj][self.ses]["surface_smooth"]
            )
            # get number of volumes (assumes time is the first dimension)
            func_len = cifti.shape[0]  # type: ignore
        # calculate interpolation time points
        frame_times = func_tr * (np.arange(func_len) + 0.5)
        # round to two decimal points
        frame_times = np.round(frame_times, 2)
        return frame_times

    def _load_physio(self) -> Tuple[dict[str, pd.Series], dict[str, int]]:
        """
        load physio signals (dataset-specific)

        Returns
        -------
        signals: dict[str, pd.Series]
            physio signals
        sf: dict[str, int]
            sampling frequency of physio signals
        """
        # get physio loader
        if self.dataset == "vanderbilt":
            signals, sf = ds.load_physio_vanderbilt(
                physio_fp=self.fmap["physio"][self.subj][self.ses]["raw"]["physio"],
                eeg_fp=self.fmap["physio"][self.subj][self.ses]["raw"]["eeg"],
            )
        else:
            raise ValueError(
                f"{self.dataset} is not a valid dataset for physio preprocessing"
            )
        # if motion signals are present, load fsl motion parameters
        if "motion" in self.params["physio"]["signals"]:
            signals["motion"] = load_fsl_motion_params(  # type: ignore
                fp=self.fmap["physio"][self.subj][self.ses]["raw"]["motion"]
            )
            sf["motion"] = 1 / self.params["func"]["tr"]
        # check physio signals and signal names in params match
        for p in self.params["physio"]["signals"]:
            if p not in signals:
                raise ValueError(f"""
                    {p} is specified in params.json but was not loaded in
                    physio preprocessing pipeline
                """)
            # if plot_physio is True, save figure
            if self.plot_physio:
                fp_raw_base = os.path.splitext(
                    self.fmap["physio"][self.subj][self.ses]["signal"][p]
                )[0]
                fp = f"{fp_raw_base}.png"
                fig, ax = plt.subplots(figsize=(15, 5))
                if p == "motion":
                    for motion_param in signals[p]:
                        signal_t = np.arange(len(signals[p][motion_param])) * (
                            1 / sf[p]
                        )
                        ax.plot(signal_t, signals[p][motion_param], label=motion_param)
                else:
                    signal_t = np.arange(len(signals[p])) * (1 / sf[p])
                    ax.plot(signal_t, signals[p], label=p)
                ax.legend()
                plt.savefig(fp)
                plt.close()

        return signals, sf  # type: ignore

    def _physio_extract(
        self, signal: pd.Series, sf: float, physio_str: str
    ) -> dict[str, np.ndarray]:
        """
        Perform physio preprocessing

        Parameters
        ----------
        signal: np.ndarray
            physio signals
        sf: float
            sampling frequency of physio signal
        physio_str:
            physio label

        Returns
        -------
        signal_extract: dict[str, np.ndarray]
            preprocessed physio signals (key = physio label)
        """
        # get physio preproc func and output mapping from dict
        physio_func_info = self.physio_func_map[physio_str]
        # extract physio signal(s)
        signal_extract = physio_func_info["func"](signal, sf)

        # map extracted signals to output labels
        signal_extract_out = {}
        for physio_out in self.params["physio"]["out"][physio_str]:
            # find the internal key that maps to this output label
            for internal_key, output_label in physio_func_info["output_map"].items():
                if output_label == physio_out:
                    signal_extract_out[physio_out] = signal_extract[internal_key]
                    break
        return signal_extract_out

    def _resample_physio(
        self,
        signal: np.ndarray,
        sf: float,
        interp_method: Literal["cubic", "nearest"] = "cubic",
    ) -> np.ndarray:
        """
        Resample preprocessed physio signals to functional scan volumes using
        cubic or nearest neighbor interpolation to the self.func_t (functional times)
        attribute. For physio recordings, signals should be first low-pass
        filtered (< 0.2 Hz).

        Parameters
        ----------
        signal: np.ndarray
            physio signals
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
        # dont filter sample weights
        if interp_method not in ["cubic", "nearest"]:
            raise ValueError("interp method must be cubic or nearest")
        # get signal time points
        signal_t = np.arange(len(signal)) * (1 / sf)
        signal_resamp = nk.signal_interpolate(
            x_values=signal_t, y_values=signal, x_new=self.func_t, method=interp_method
        )
        return signal_resamp

    def _set_pipeline_steps(self, p: str) -> dict[str, bool]:
        """
        Set pipeline steps for a given physio signal.

        Parameters
        ----------
        p: str
            physio signal label

        """
        # first, check whether physio signal is in pipeline_skip
        if p in self.params["physio"]["pipeline_skip"]:
            # get pipeline skip params
            pipeline_skip = self.params["physio"]["pipeline_skip"][p]
            # check that steps are in pipeline_skip
            for s in pipeline_skip:
                if s not in self.STEPS:
                    raise ValueError(f"{s} is not a valid pipeline step")
        else:
            pipeline_skip = []
        # get pipeline steps
        pipeline_steps = {}
        for s in self.STEPS:
            if s in pipeline_skip:
                pipeline_steps[s] = False
            else:
                pipeline_steps[s] = True
        return pipeline_steps


def tedana_denoise(
    fps_in: str,
    echo_times: List[float],
    mask: str,
    out_dir: str,
    out_prefix: str,
):
    """
    Run tedana workflow - Multi-Echo ICA denoising

    Parameters
    ----------
        fps_in: str
            lists of filepaths to each echo in order
        echo_times: List[float]
            echo times in order
        mask: str
            file path to freesurfer binary brain mask in functional space
        out_dir: str
            output directory
        out_prefix: str
            file name prefix for all tedana outputs
    """
    # run tedana workflow
    tedana_workflow(
        data=fps_in,
        tes=echo_times,
        mask=mask,
        prefix=out_prefix,
        out_dir=out_dir,
        overwrite=True,
    )


def _par_execute_anat(params: str) -> None:
    # execution of anatomical pipeline for run_parallel()
    anat_proc = AnatomicalPipeline(**params)  # type: ignore
    anat_proc.run()


def _par_execute_func(params: str, func_pipe_type: str) -> None:
    # execution of functional pipeline for run_parallel()
    if func_pipe_type == "full":
        func_proc = FunctionalPipelineFull(**params)  # type: ignore
    elif func_pipe_type == "cifti-partial":
        func_proc = FunctionalPipelineCiftiPartial(**params)  # type: ignore
    else:
        raise ValueError(f"{func_pipe_type} is not a valid function pipeline type")
    func_proc.run()


def _par_execute_physio(params: str) -> None:
    # execution of physio pipeline for run_parallel()
    physio_proc = PhysioPipeline(**params)  # type: ignore
    physio_proc.run()

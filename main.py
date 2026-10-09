"""
Command-line utility for performing replication of SCAN network
analysis. The following analysis pipelines are available, including:

1) Modeling of hemodynamic responses to manually annotated 'yawns'
and 'sigh' events.

2) Functional connectivity analysis of ROI time series.

"""

import argparse
import os
from typing import List, Tuple

import numpy as np

from scan.io.load import DatasetLoad, DatasetOutput, DatasetOutputConcat, Gifti
from scan.model.fc import FCMapModel
from scan.model.glm import GLMSpline

# OUTPUT DIRECTORY
OUT_DIRECTORY = "results/main"

# SCAN ROI MASKS (fs_LR)
SCAN_LH_ROI_MASKS = [
    "template/lh.SCAN_bottom.label.gii",
    "template/lh.SCAN_middle.label.gii",
    "template/lh.SCAN_top.label.gii",
]
SCAN_RH_ROI_MASKS = [
    "template/rh.SCAN_bottom.label.gii",
    "template/rh.SCAN_middle.label.gii",
    "template/rh.SCAN_top.label.gii",
]
# SCAN ATLAS ROI
SCAN_LH_ATLAS_ROI_MASK = "template/lh.SCAN_fsLR.label.gii"
SCAN_RH_ATLAS_ROI_MASK = "template/rh.SCAN_fsLR.label.gii"

# Effector Atlas ROIs (fs_LR)
FOOT_LH_ROI_MASK = "template/lh.FOOT_fsLR.label.gii"
FOOT_RH_ROI_MASK = "template/rh.FOOT_fsLR.label.gii"
HAND_LH_ROI_MASK = "template/lh.HAND_fsLR.label.gii"
HAND_RH_ROI_MASK = "template/rh.HAND_fsLR.label.gii"
MOUTH_LH_ROI_MASK = "template/lh.MOUTH_fsLR.label.gii"
MOUTH_RH_ROI_MASK = "template/rh.MOUTH_fsLR.label.gii"

# FC analysis seeds
FC_SEEDS = {
    "SCAN": (SCAN_LH_ATLAS_ROI_MASK, SCAN_RH_ATLAS_ROI_MASK),
    "FOOT": (FOOT_LH_ROI_MASK, FOOT_RH_ROI_MASK),
    "HAND": (HAND_LH_ROI_MASK, HAND_RH_ROI_MASK),
    "MOUTH": (MOUTH_LH_ROI_MASK, MOUTH_RH_ROI_MASK),
}


def load_data(
    analysis: str,
    roi: bool = False,
    regress_global_signal: bool = False,
    left_roi_masks: List[str] | None = None,
    right_roi_masks: List[str] | None = None,
) -> Tuple[DatasetLoad, DatasetOutput | DatasetOutputConcat, Gifti]:
    """
    Load data from dataset.

    Parameters
    ----------
    analysis: str
        type of analysis to perform ('fc' for functional connectivity)
    roi: bool
        whether to load ROI time courses
    left_roi_masks: List[str] | None
        list of left hemisphere ROI masks to apply (required if roi is True)
    right_roi_masks: List[str] | None
        list of right hemisphere ROI masks to apply (required if roi is True)
    regress_global_signal: bool = False
        whether to regress out the global signal from func.gii data. Note,
        global signal regression should not be applied to ROI time courses.
        Default is False.

    Returns
    -------
    loader: DatasetLoad
        dataset loader object
    data: DatasetOutput | DatasetOutputConcat
        loaded functional data, concatenated across sessions if specified
    gii: Gifti
        Gifti object containing the full cortical time series
    """
    if roi and (left_roi_masks is None or right_roi_masks is None):
        raise ValueError("Must provide left and right ROI masks if roi is True")

    # determine whether to concat functional data across sessions based on analysis
    if analysis in ("fc",):
        concat = True
    else:
        concat = False

    # load all scans in dataset
    loader = DatasetLoad("vanderbilt")
    # load data with ROI masks if specified
    if roi:
        data, gii = loader.load(
            data_type="func",
            # high-pass filter functional data w/ 0.01 Hz cutoff
            func_high_pass=True,
            input_mask=True,
            # average left and right hemisphere ROI time courses together
            roi_avg_left_right=True,
            lh_roi_masks=left_roi_masks,
            rh_roi_masks=right_roi_masks,
            regress_global_signal=regress_global_signal,
        )
    else:
        data, gii = loader.load(
            data_type="func",
            # high-pass filter functional data w/ 0.01 Hz cutoff
            func_high_pass=True,
            regress_global_signal=regress_global_signal,
        )
        breakpoint()

    # concatenate data across sessions
    if concat:
        data = data.concatenate()

    assert gii is not None, (
        "Gifti object is None. Please check the data loading process."
    )

    return loader, data, gii


def main(analysis: str, out_dir: str):
    # create out directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    if analysis == "fc":
        # for seed-based FC analysis, we need to load ROI time courses and full cortical time series
        # load SCAN atlas ROI time courses
        loader, data_roi_scan, _ = load_data(
            analysis=analysis,
            roi=True,
            left_roi_masks=[
                FC_SEEDS["SCAN"][0],
                FC_SEEDS["FOOT"][0],
                FC_SEEDS["HAND"][0],
                FC_SEEDS["MOUTH"][0],
            ],
            right_roi_masks=[
                FC_SEEDS["SCAN"][1],
                FC_SEEDS["FOOT"][1],
                FC_SEEDS["HAND"][1],
                FC_SEEDS["MOUTH"][1],
            ],
        )
        assert isinstance(data_roi_scan, DatasetOutputConcat), (
            "Data is not ROI-based. Please check the data loading process."
        )

        # load full cortical time series with global signal regression
        _, data, gii = load_data(
            analysis=analysis, roi=False, regress_global_signal=True
        )
        assert isinstance(data, DatasetOutputConcat), (
            "Data is not concatenated. Please check the data loading process."
        )
        # loop through each SCAN atlas ROI and perform seed-based FC analysis
        for roi_idx, roi_name in enumerate(FC_SEEDS):
            # perform seed-based FC analysis using SCAN atlas ROIs as seeds
            seed_fc(
                seed_roi_ts=data_roi_scan.func[:, roi_idx],
                data=data,
                gii=gii,
                label=roi_name,
                out_dir=out_dir,
            )
    elif analysis == "resp-events":
        _, data, gii = load_data(analysis=analysis, roi=False)
        assert isinstance(data, DatasetOutput), (
            "Data is not session-based. Please check the data loading process."
        )
        resp_event_glm(data=data, gii=gii, out_dir=out_dir)

    else:
        raise ValueError(f"Unknown analysis: {analysis}")


def seed_fc(
    seed_roi_ts: np.ndarray,
    data: DatasetOutputConcat,
    gii: Gifti,
    label: str,
    out_dir: str,
):
    """
    Perform seed-based functional connectivity analysis. Fit FC map model from time series of each ROI.
    """
    # fit FC map model to predict time series of each ROI from time series of all other ROIs
    fc_map_model = FCMapModel()
    fc_map_res = fc_map_model.fit(seed_ts=seed_roi_ts, func_data=data.func)
    fc_map_res.write(gii, file_prefix=f"vanderbilt_fc_map_{label}_roi", out_dir=out_dir)


def resp_event_glm(
    data: DatasetOutput,
    gii: Gifti,
    out_dir: str,
):
    """
    Fit GLM spline model to predict fMRI time series from physiological events. Yawn
    and sigh events are modeled separately, and the resulting predicted
    time series are written to GIFTI files.
    """
    glm_sigh = GLMSpline(tr=2.1)
    glm_sigh.fit(fmri=data.func, events=data.sigh_events)
    glm_sigh_pred = glm_sigh.evaluate()
    glm_sigh_pred.write(gii, file_prefix="vanderbilt_glm_sigh", out_dir=out_dir)
    # yawns
    glm_yawn = GLMSpline(tr=2.1)
    glm_yawn.fit(fmri=data.func, events=data.yawn_events)
    glm_yawn_pred = glm_yawn.evaluate()
    glm_yawn_pred.write(gii, file_prefix="vanderbilt_glm_yawn", out_dir=out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replication of SCAN network analysis")
    parser.add_argument(
        "-a",
        "--analysis",
        type=str,
        required=True,
        help="Analysis to perform",
        choices=["fc", "resp-events"],
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        required=False,
        help="output directory to save analysis outputs",
        default=OUT_DIRECTORY,
    )
    args = parser.parse_args()
    main(args.analysis, args.out_dir)

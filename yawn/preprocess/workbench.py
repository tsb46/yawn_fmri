"""Workbench (V1.4.2) command line utilities"""

import os
from typing import Literal

import nipype.interfaces.workbench as wb


def cifti_smooth(fp_in: str, fp_out: str, fwhm: float) -> None:
    """
    Smooth CIFTI file using wb_command -cifti-smoothing

    Parameters
    ----------
        fp_in: str
            filepath to CIFTI file to smooth
        fp_out: str
            filepath to smoothed CIFTI file
        fwhm: float
            FWHM of the Gaussian kernel in mm
    """
    # template prefix
    template_prefix = "template/fsaverage"

    # convert fwhm to sigma (standard deviation)
    sigma = fwhm / 2.3548
    cifti_smooth = wb.CiftiSmooth()
    cifti_smooth.inputs.in_file = fp_in
    cifti_smooth.inputs.direction = "COLUMN"
    cifti_smooth.inputs.left_surf = f"{template_prefix}.L.inflated.32k_fs_LR.surf.gii"
    cifti_smooth.inputs.right_surf = f"{template_prefix}.R.inflated.32k_fs_LR.surf.gii"
    cifti_smooth.inputs.sigma_surf = sigma
    cifti_smooth.inputs.sigma_vol = sigma
    cifti_smooth.inputs.out_file = fp_out
    cifti_smooth.run()


def create_midthickness(
    hemi: Literal["rh", "lh"],
    fs_mid: str,
    lr_mid: str,
    sphere_out: str,
    fs_subj_dir: str,
) -> None:
    """
    Create midthickness files from Freesurfer subject recon-all outputs
    using wb_shortcuts -freesurfer-resample-prep

    Step 1 in Option B. FreeSurfer native individual data to fs_LR
    https://wiki.humanconnectome.org/docs/assets/Resampling-FreeSurfer-HCP_5_8.pdf

    Parameters
    ----------
        hemi: Literal[rh, lh]
            left or right hemisphere
        fs_mid : str
            output midthickness file for freesurfer midthickness
        lr_mid: str
            output midthickeness file for fs_LR midthickness
        sphere_out: str
            output sphere file
        fp_subj_dir: str
            filepath to subject freesurfer recon-all outputs
    """
    # create midthickness (mid) for left and right hemispheres
    template_prefix = "template/fs_LR-deformed_to-fsaverage"
    if hemi == "lh":
        hemi_lr = "L"
    elif hemi == "rh":
        hemi_lr = "R"
    else:
        raise ValueError(f'param hemi must be "lh" or "rh", not {hemi}')

    # get file paths to surfaces
    fs_white = f"{fs_subj_dir}/surf/{hemi}.white"
    fs_pial = f"{fs_subj_dir}/surf/{hemi}.pial"
    fs_sphere = f"{fs_subj_dir}/surf/{hemi}.sphere.reg"
    lr_sphere = f"{template_prefix}.{hemi_lr}.sphere.32k_fs_LR.surf.gii"

    # execute command
    os.system(f"""
        wb_shortcuts -freesurfer-resample-prep \
        {fs_white} {fs_pial} {fs_sphere} {lr_sphere} \
        {fs_mid} {lr_mid} {sphere_out}
    """)


def fs2fslr(
    hemi: str,
    fp_in: str,
    fp_out: str,
    fs_mid: str,
    lr_mid: str,
    fs_subj_dir: str,
    fs_sphere: str,
) -> None:
    """
    Resample functional .gii file to the fs_LR mesh using workbench metric-resample.

    Step 2 in Option B. FreeSurfer native individual data to fs_LR
    https://wiki.humanconnectome.org/docs/assets/Resampling-FreeSurfer-HCP_5_8.pdf

     Parameters
    ----------
        hemi: Literal[rh, lh]
            left or right hemisphere
        fp_in: str
            filepath to func.gii file from mri_vol2surf
        fs_sphere: str
            current gifti sphere from create_midthickness command (above)
        fs_mid : str
            current midthickness file from create_midthickness command (above)
        lr_mid: str
            fs_LR midthickness file from create_midthickness command (above)
        fp_subj_dir: str
            filepath to subject freesurfer recon-all outputs

    """
    # create midthickness (mid) for left and right hemispheres
    template_prefix = "template/fs_LR-deformed_to-fsaverage"
    if hemi == "lh":
        hemi_lr = "L"
        wb_label = "CORTEX_LEFT"
    elif hemi == "rh":
        hemi_lr = "R"
        wb_label = "CORTEX_RIGHT"
    else:
        raise ValueError(f'param hemi must be "lh" or "rh", not {hemi}')

    metres = wb.MetricResample()
    metres.inputs.in_file = fp_in
    metres.inputs.method = "ADAP_BARY_AREA"
    metres.inputs.current_sphere = fs_sphere
    metres.inputs.new_sphere = f"{template_prefix}.{hemi_lr}.sphere.32k_fs_LR.surf.gii"
    metres.inputs.current_area = fs_mid
    metres.inputs.new_area = lr_mid
    metres.inputs.area_surfs = True
    metres.inputs.out_file = fp_out
    metres.run()

    # set structure as left or right cortex to view in connectome workbench
    os.system(f"""
    wb_command -set-structure  {fp_out} {wb_label}
    """)

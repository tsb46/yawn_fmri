"""Freesurfer (V7.2.0) command line utilities"""

from typing import Literal

import nipype.interfaces.freesurfer as fs


def mask_to_func(
    mask: str,
    func_mean: str,
    mask_out: str,
    mat_lta: str
) -> None:
    """
    Take masked brain mask from Freesurfer and transform to functional space
    by inverting the transform from BBRegister, then create binary mask
    with Binarize

    Parameters
    ----------
        mask : str
            path to Freesurfer masked brain
        func_mean: str
            path to mean functional volume from MCFLIRT
        mask_out: str
            path to output brain binary mask in functional space
        mat_lta: str
            path to .lta transformation matrix file from BBRegister

    """
    # send masked brain to functional space
    mask2func = fs.ApplyVolTransform(inverse=True)
    mask2func.inputs.target_file = mask
    mask2func.inputs.reg_file = mat_lta
    mask2func.inputs.source_file = func_mean
    mask2func.inputs.transformed_file = mask_out
    mask2func.run()

    # binarize brain to make mask
    binarize = fs.Binarize(min=0.5, out_type='nii')
    binarize.inputs.in_file = mask_out
    binarize.inputs.binary_file = mask_out
    binarize.run()



def recon_all(t1: str, subj_label: str, dir_out: str) -> None:
    """
    Reconstruct cortical surface from T1w using Freesurfer
    recon-all

    Parameters
    ----------
        t1 : str
            path to raw T1w image
        subj: str
            subject label
        dir_out: str
            path to output directory
    """
    fsreconall = fs.ReconAll()
    fsreconall.inputs.subject_id = subj_label
    fsreconall.inputs.directive = 'all'
    fsreconall.inputs.subjects_dir = dir_out
    fsreconall.inputs.T1_files = t1
    fsreconall.run()


def surf_register(
    fp_reg_out: str,
    fp_mat_out: str,
    fp_fslmat_out: str,
    func_mean: str,
    subj: str,
    subjects_dir: str,
) -> None:
    """
    Boundary based coregistration of functional template with
    T1w with Freesurfer bbregister.

    Parameters
    ----------
        fp_reg_out: str
            path to output registration file
        fp_mat_out: str
            path to output transformation matrix file
        func_mean: str
            path to mean functional volume from MCFLIRT
        subj: str
            subject label
        subjects_dir: str
            path to freesurfer recon-all output directory
    """
    bbregister = fs.BBRegister()
    # initial affine registraction w/ FSL FLIRT
    bbregister.inputs.init = 'fsl'
    bbregister.inputs.contrast_type = 't2'
    bbregister.inputs.out_reg_file = fp_mat_out
    bbregister.inputs.out_fsl_file = fp_fslmat_out
    bbregister.inputs.registered_file = fp_reg_out
    bbregister.inputs.source_file = func_mean
    bbregister.inputs.subject_id = subj
    bbregister.inputs.subjects_dir = subjects_dir
    # FileNotFoundError after successful completion, ignore
    try:
        bbregister.run()
    except FileNotFoundError:
        pass


def vol2surf(
    hemi: Literal['lh', 'rh'],
    smooth_fwhm: float,
    subj: str,
    subjects_dir: str,
    fp_in: str,
    fp_out: str,
    fp_reg_mat: str
) -> None:
    """
    Map functional volumes to native cortical surface using Freesurfer's
    mri_vol2surf

    Parameters
    ----------
        hemi: Literal[rh, lh]
            left or right hemisphere
        smooth_fwhm: float
            fwhm with isotropic spatial smoothing on cortical surface
        subj: str
            subject label
        subjects_dir: str
            path to freesurfer recon-all output directory
        fp_reg_mat: str
            path to output registration file from bbregister
        fs_subj_dir: str
            filepath to subject freesurfer recon-all outputs
    """
    sampler = fs.SampleToSurface(hemi=hemi)
    sampler.inputs.source_file = fp_in
    sampler.inputs.reg_file = fp_reg_mat
    sampler.inputs.out_file = fp_out
    sampler.inputs.subject_id = subj
    sampler.inputs.subjects_dir = subjects_dir
    sampler.inputs.sampling_method = 'average'
    sampler.inputs.smooth_surf = smooth_fwhm
    sampler.inputs.sampling_units = "frac"
    sampler.inputs.sampling_range = (0, 1, 0.2)
    sampler.inputs.out_type = 'gii'
    sampler.run()






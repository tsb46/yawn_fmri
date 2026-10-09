# fmri_yawn
Analysis of the effect of spontaneous yawns and sighs on resting-state fMRI signals.

# Dataset
The dataset used in this study is simultaneous EEG, peripheral physiology and multi-echo fMRI recordings (27 subjects, 38 sessions) collected by the Neurdy lab at Vanderbilt University. 

# Abstract

Arousal fluctuations are a major driver of large-scale spontaneous signal changes in resting-state fMRI. Much of these spontaneous signal changes are mediated by cerebrovascular and neurovascular mechanisms associated with central and peripheral noradrenergic pathways. The influence of arousal on spontaneous fMRI signals may also occur indirectly via its association with overt respiratory behaviors, such as yawns and sighs. These distinct respiratory behaviors involve coordinated motor activity of the chest, throat and facial muscles (eye closure and jaw stretching for yawns). We hypothesized that the coordinated motor activity of yawns and sighs induce a spatially localized pattern of activity in the primary motor cortex that is synchronized with the widespread signal fluctuations associated with arousal. Using simultaneous resting-state EEG-multi-echo-fMRI recordings with electrooculography (EOG) measurements above the eye and electromyography (EMG) measurements of the jaw to detect yawns and sighs, we found a consistent pattern of activation overlapping areas of the primary motor cortex previously characterized as the somato-cognitive action network (SCAN), as well as the cingulo-opercular and salience networks. These findings establish that the coordinated motor movements of yawns and sighs are associated with a consistent pattern of spontaneous signal changes in the primary motor cortex that overlap the SCAN network.

# Installation
This repository uses `uv` for environment and dependency management. To create the virtual environment and install the project with the locked dependencies, run:
```
uv sync
```

Optional extras: `viz` (plotly/dash, used by the timecourse plots) and `templates` (neuromaps/templateflow, used by `scripts/gordon18_to_scan_roi.py`):
```
uv sync --extra viz --extra templates
```

The functional/anatomical preprocessing also calls external neuroimaging tools that must be on your `PATH`: FSL (`bet`, `mcflirt`, `slicetimer`), FreeSurfer (`recon-all`, `bbregister`, `mri_vol2surf`) and Connectome Workbench (`wb_command`). Multi-echo denoising uses `tedana` (installed by `uv sync`).

**All commands below should be run from the repository root**, since paths (e.g. `scan/meta/params.json`, `template/`, `results/`) are relative.

# Repository Layout

```
main.py            CLI for the group-level analyses (GLM of yawn/sigh events, seed-based FC)
scan/
  meta/            dataset metadata: params.json, participant list, slice order
  preprocess/      preprocessing pipelines (anatomical, functional, physio)
  io/              loading preprocessed data for analysis, writing GIFTI outputs
  model/           analysis models (GLMSpline, FCMapModel)
  plots/           surface snapshot, network summary and timecourse plotting
plots/             jupyter notebooks that generate figures
scripts/           miscellaneous standalone scripts
template/          fs_LR surfaces, SCAN/effector ROI labels and atlases
events/            manually annotated yawn/sigh events (one JSON per subject/session)
data/vanderbilt/   raw and preprocessed data (anat/, func/, physio/)
results/           analysis outputs (results/main) and figures (results/figures)
```

# Running the Repository

The workflow is: (1) preprocess the raw data, (2) load the preprocessed data, (3) run the analyses with `main.py`, (4) plot the results, (5) optionally run the miscellaneous scripts.

## 1. Preprocessing (`scan/preprocess`)

Preprocessing is orchestrated by `Pipeline` in `scan/preprocess/pipeline.py`. Dataset-specific settings (directories, file naming, TR, number of trimmed volumes, echo times, smoothing, physio signals to extract) are defined in `scan/meta/params.json`. The subjects/sessions to process are listed in `scan/meta/vanderbilt_participant.csv`.

Place raw data in the following layout (file names are set by `file_format` in `params.json`):
```
data/vanderbilt/anat/raw/       T1w images,                    vcon{subject}-anat.nii.gz
data/vanderbilt/func/raw/       multi-echo fMRI (per echo),    vcon{subject}-scan{session}_ecr_e{echo}.nii.gz
data/vanderbilt/physio/raw/     respiratory belt, EEG (EOG/EMG), motion parameters
```

Run the pipeline from Python:
```python
from scan.preprocess.pipeline import Pipeline

pipe = Pipeline(dataset="vanderbilt")
pipe.run()                   # sequential
# pipe.run_parallel(n_cores=4)  # parallel over subjects/sessions
```

Use the skip flags to re-run only part of the pipeline, e.g. physio only:
```python
Pipeline(dataset="vanderbilt", anat_skip=True, func_skip=True).run()
```
Available flags: `anat_skip`, `reconall_skip` (skip FreeSurfer `recon-all` but still build surfaces), `func_skip`, `physio_skip`.

The pipeline is composed of three sub-pipelines:

| Pipeline | Steps | Output directories |
|---|---|---|
| `AnatomicalPipeline` | FreeSurfer `recon-all`; midthickness/sphere surfaces for fs_LR resampling | `data/vanderbilt/anat/{freesurfer,proc1_fslr}` |
| `FunctionalPipelineFull` | trim dummy volumes, slice-timing correction, motion correction (MCFLIRT), BBRegister coregistration, tedana multi-echo denoising, volume-to-surface sampling with smoothing, resampling to fs_LR 32k | `data/vanderbilt/func/proc1_trim` ... `proc6_surfacelr` (final `*.lh.func.gii` / `*.rh.func.gii`) |
| `PhysioPipeline` | load respiratory belt, EEG/EOG/EMG and FSL motion parameters; extract features (respiratory amplitude/instantaneous frequency, EOG/EMG high-frequency amplitude, framewise displacement and motion parameters, sample weights); trim to match functional trimming; resample to 10 Hz; QC plots of raw signals | `data/vanderbilt/physio/proc1_physio` (`.txt` per signal) |

Helper modules: `freesurfer.py`, `fsl.py`, `workbench.py` (wrappers around the external tools), `physio.py` (feature extraction), `dataset.py` (dataset-specific physio/EEG loading) and `custom.py` (framewise displacement, CIFTI trimming).

## 2. Analyses (`scan/model` and `main.py`)

`main.py` is the command-line entry point for the group-level analyses:
```
python main.py --analysis {fc,resp-events} [--out_dir results/main]
```

| Analysis | Description | Model | Outputs (in `--out_dir`) |
|---|---|---|---|
| `resp-events` | Models the fMRI response to manually annotated yawn and sigh events separately, per scan (spline-based finite impulse response). | `scan.model.glm.GLMSpline` | `vanderbilt_glm_{yawn,sigh}.pkl` and `vanderbilt_glm_{yawn,sigh}_{lh,rh}.func.gii` |
| `fc` | Seed-based functional connectivity. Seeds are the SCAN, foot, hand and mouth effector ROIs (`template/*.label.gii`) averaged across hemispheres; maps are computed on globally-regressed, session-concatenated data. | `scan.model.fc.FCMapModel` | `vanderbilt_fc_map_{SCAN,FOOT,HAND,MOUTH}_roi.pkl` and `..._{lh,rh}.func.gii` |

Examples:
```
python main.py -a resp-events
python main.py -a fc -o results/main
```

## 3. Plots (`scan/plots` and `plots/`)

`scan/plots` provides reusable plotting code (importable from `scan.plots`):

- `snapshot.py`: `build_snapshot_plotter(...)` builds a `SurfaceSnapshotPlotter` for fs_LR surface snapshots from `.func.gii` (or CIFTI) inputs, with optional ROI/label contours (`roi_left`, `roi_right`, `label_*`), atlas overlays (`atlas="gordon"` or `"yeo"`, `atlas_label_index`), colour scaling (`vmin`, `vmax`, `auto_percentiles`, `threshold`) and camera settings (`surf_views`, `surf_zoom`, `surf_elev_*`, `surf_azim_*`). Call `render_figure()` for a standalone figure, or `render_into_axes(axes=[...])` to draw into axes of a larger matplotlib layout.
- `surface.py`: low-level surface rendering (`render_surface_figure`, `SurfaceOverlay`, `SurfaceRenderOptions`).
- `network_summary.py`: `plot_network_summary` for network-level bar summaries.
- `timecourse.py` / `timecourse_dash.py`: interactive Plotly time series plots (`plot_timeseries`) and a Dash app for annotating events (`create_timecourse_annotation_app`, requires the `viz` extra).
- `layout.py`, `utils.py`: legends, colorbars and layout helpers.

The notebooks in `plots/` use these modules to generate the figures. Run them from Jupyter/VS Code with the repository root as the working directory (or the relative paths to `results/` and `template/` will not resolve), after running the analyses in step 3:

| Notebook | Description |
|---|---|
| `plots/physio_scan.ipynb` | Reads the GLM outputs in `results/main/vanderbilt_glm_{yawn,sigh}*` and uses `build_snapshot_plotter` to render the predicted fMRI response on the cortical surface at a selected lag, with SCAN ROI contours and Gordon-atlas cingulo-opercular/salience network outlines, combined into a multi-panel figure. |
| `plots/resp_event_physio.ipynb` | Uses `DatasetLoad` to load fMRI and physio (native 10 Hz respiratory/EOG/EMG; TR-sampled head motion and global signal), then computes bootstrapped (subject-level) event-aligned averages around yawn/sigh events. |

## Scripts (`scripts/`)

Standalone scripts that are not part of the main analysis flow:

- `scripts/gordon18_to_scan_roi.py`: builds the SCAN, foot, hand and mouth ROI label files in `template/` from the Gordon18 atlas (`.mgh` in fsaverage6): converts to GIFTI, transforms to fs_LR 32k with `neuromaps`, and splits the SCAN ROI into top/middle/bottom. Requires the `templates` extra and the Gordon18 `.mgh` files in `template/` (download from https://github.com/pBFSLab/UNITE). Paths are relative to `scripts/`, so run it from there:
  ```
  cd scripts && python gordon18_to_scan_roi.py
  ```
- `scripts/figure2b_panel_b.py`: renders a Figure 2B-style summary of SCAN vs. effector (foot, hand, mouth) FC as presented in Gordon et al. (2023): a surface map of `inter-effector - max(foot, hand, mouth)` and per-network bars from a CIFTI atlas. Takes the FC maps produced by `main.py -a fc`:
  ```
  python scripts/figure2b_panel_b.py \
    --scan-left results/main/vanderbilt_fc_map_SCAN_roi_lh.func.gii \
    --scan-right results/main/vanderbilt_fc_map_SCAN_roi_rh.func.gii \
    --foot-left results/main/vanderbilt_fc_map_FOOT_roi_lh.func.gii \
    --foot-right results/main/vanderbilt_fc_map_FOOT_roi_rh.func.gii \
    --hand-left results/main/vanderbilt_fc_map_HAND_roi_lh.func.gii \
    --hand-right results/main/vanderbilt_fc_map_HAND_roi_rh.func.gii \
    --mouth-left results/main/vanderbilt_fc_map_MOUTH_roi_lh.func.gii \
    --mouth-right results/main/vanderbilt_fc_map_MOUTH_roi_rh.func.gii \
    --atlas gordon --output results/figures/figure2b_panel_b.png
  ```
  Run `python scripts/figure2b_panel_b.py --help` for the remaining options (colormap, scaling, network order/exclusion, figure size).


## Loading the dataset (`scan/io`)

`scan/io` loads the preprocessed outputs for analysis:

- `file.py`: `Participant` iterates over the subjects/sessions in the participant list and builds file paths.
- `load.py`: `DatasetLoad` loads fMRI (`proc6_surfacelr`), physio (`proc1_physio`) and the annotated yawn/sigh events (`events/subject-XX_session-YY.json`). `Gifti` handles left/right hemisphere concatenation/splitting (optionally excluding the medial wall).
- `write.py`: writes results to `.func.gii` / `.label.gii`.
- `utils.py`: ROI mask checks and filtering helpers.

```python
from scan.io.load import DatasetLoad

loader = DatasetLoad("vanderbilt")   # all subjects/sessions; or pass subj_ses_select=[("02", "01"), ...]
data, gii = loader.load(
    data_type=("func", "physio"),    # "func", "physio" or both
    norm="zscore",                   # "zscore", "demean", "robust_z" or None
    func_high_pass=True,             # 0.01 Hz high-pass on fMRI
    regress_global_signal=False,
    resample_physio=True,            # False keeps the native 10 Hz physio
)
data.func          # list of (time x vertices) arrays, one per scan
data.physio        # dict: signal name -> list of arrays
data.sigh_events   # list of event times (s), one list per scan
data.yawn_events
data = data.concatenate()   # concatenate across scans
```

To extract ROI time courses instead of vertex-wise data, pass `input_mask=True` with `lh_roi_masks` / `rh_roi_masks` (lists of fs_LR `.label.gii` files from `template/`) and optionally `roi_avg_left_right=True`.








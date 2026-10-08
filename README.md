# scan_physio
Analysis of the relationship between the recently identified SCAN network and global BOLD fluctuations.

In a recent manuscript by Gordon et al. (2023), a set of brain regions were identified that interrupts the somatotopic organization of the primary motor cortex (the 'homunculus'). The authors refer to this set of regions as the somato-cognitive action network (SCAN).  

In an open review by the Diedrichsen lab  (https://www.diedrichsenlab.org/BrainDataScience/or_gordon2023/index.htm), it is suggested that the co-activation of these brain regions may reflect muscle synergies induced by respiratory motion - i.e. abdmonimal, larynx and upper face muscular activity. To further assess these claims we analyzed the relationship between these regions and the amplitude of respiratory belt signals. This analysis was conduced on a small subset of randomly sampled HCP subjects (N=20) with respiratory belt recordings of sufficient quality. This code contains a set of command-line python scripts for replicating the analysis. The code was run with Python 3.11.3.

# Installation
This repository uses `uv` for environment and dependency management. To create the virtual environment and install the project with the locked dependencies, run:
```
uv sync
```

# Visualization Scripts
Static surface snapshots can be rendered from paired GIFTI metric files with:
```
uv run python scripts/gifti_snapshot.py --input path/to/lh.func.gii --index 0 --output snapshot.png
```

Surface movies can be rendered with the new GIFTI movie companion script:
```
uv run python scripts/gifti_movie.py --input path/to/lh.func.gii --output movie.mp4
```

Movie encoding uses `ffmpeg`. The script first tries a system `ffmpeg`, then falls back to the optional `imageio-ffmpeg` package. To install that fallback dependency, sync with the `viz` extra:
```
uv sync --extra viz
```

The same `viz` extra also enables interactive time-series plots via Plotly. The new `plot_timeseries` helper accepts either a 2D NumPy array plus column labels or a pandas DataFrame, defaults to vertically stacked synchronized subplots, and can switch to an overlaid WebGL view when needed.

The same extra now also includes a small Dash-based annotation workflow for manual marking and JSON export. The app reuses the Plotly timecourse figure, lets you click to add vertical markers, and saves a per-session JSON payload containing the sample index, time in seconds, and signal values for each marker. A minimal launch pattern looks like this:
```
from scan.plots.timecourse_dash import create_timecourse_annotation_app

app = create_timecourse_annotation_app(
	data,
	column_labels=[...],
	source_name="subject-01",
	labels=["A", "B", "C"],
)
app.run(debug=True)
```

When labels are provided, the app shows a label selector and saves the chosen label on each marker in the same JSON file.

# Command-Line Usage
The current analysis entry point is `main.py`. To view the available arguments, run:
```
uv run python main.py --help
```

To replicate the supported analyses, use one of the following commands:

* For principal component analysis (the first principal component corresponds to the 'global signal'):
```
python pca.py -n 10
```
* For traditional whole-brain GLM analysis of respiratory belt amplitudes regressed on all vertex time courses, use the following command:

```
python glm.py -p resp
```
Note, cubic spline bases of lagged respiratory belt time courses are used as regressors. Also note, no statistical testing is performed on these maps. The output of this analysis is a pickled (scikit-learn) model object (and meta parameters), along with predicted BOLD time courses at regularly spaced lags of the respiratory belt amplitude signal (the amplitude of the respiratory belt amplitude time course for prediction is set at one standard deviation above the mean - i.e. z-score-1). 

* For non-linear whole-brain GLM analyses of the respiratory belt amplitudes, using a distributed lag non-linear model (DLNM; Gasparrini et al. 2013), run the following command:

```
python glm_dlnm.py -p resp
```
Note, cubic spline basis for both the lagged respiratory belt time course and its amplitude values is used as regressors. Also note, no statistical testing is performed on these maps. The output of this analysis is a pickled (scikit-learn) model object (and meta parameters), along with predicted BOLD time courses at regularly spaced lags of the respiratory belt amplitude signal. To assess the response at different amplitudes of the respiratory belt amplitude signal, we assess the predicted BOLD time courses at several amplitude values - z-score = 1, 2, and 3. 

# Citations
Gasparrini, A., Armstrong, B., & Kenward, M. G. (2010). Distributed lag non-linear models. Statistics in Medicine, 29(21), 2224–2234. https://doi.org/10.1002/sim.3940


Gordon, E. M., Chauvin, R. J., Van, A. N., Rajesh, A., Nielsen, A., Newbold, D. J., Lynch, C. J., Seider, N. A., Krimmel, S. R., Scheidter, K. M., Monk, J., Miller, R. L., Metoki, A., Montez, D. F., Zheng, A., Elbau, I., Madison, T., Nishino, T., Myers, M. J., … Dosenbach, N. U. F. (2023). A somato-cognitive action network alternates with effector regions in motor cortex. Nature, 617(7960), Article 7960. https://doi.org/10.1038/s41586-023-05964-2







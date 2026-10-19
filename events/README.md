# Events

Manually annotated sigh and yawn events, one JSON file per session, named `subject-<ID>_session-<ID>.json`. Note, for scans with no yawns or sigh events, the 'markers' attribute is empty.

Events were marked by hand in the interactive Dash timecourse app (`yawn/plots/timecourse_dash.py`) while viewing the physiological signals (respiration, EOG, EMG).

## Format

```json
{
    "source_name": "subject-05_session-01",
    "markers": [
        {
            "sample_index": 4889,
            "time_seconds": 488.9,
            "label": "Yawn"
        }
    ]
}
```

- `source_name`: subject/session identifier, matches the file name.
- `markers`: list of annotated events.
  - `sample_index`: index into the physio timecourse (10 Hz sampling).
  - `time_seconds`: event time in seconds from the start of the physio timecourse (`sample_index / 10`).
  - `label`: `"Yawn"` or `"Sigh"` (sigh = augmented breath).

## Annotation criteria

- **Yawns**: large positive amplitude signals in the EOG, EMG, and respiratory amplitude, together with slowed respiratory rate, occurring closely in time.
- **Sighs (augmented breaths)**: large positive respiratory amplitude and slowed respiratory rate followed by a brief apnea, without large EOG/EMG signal fluctuations.

In total, 49 sighs (15 subjects, 20 sessions) and 20 yawns (14 subjects, 14 sessions) were identified for analysis.

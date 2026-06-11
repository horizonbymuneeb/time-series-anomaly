# Time-series anomaly detection on labelled synthetic series

Point anomalies and regime changes fail differently: a spike is a single bad
observation, while a level shift is a stretch of perfectly ordinary-looking points
that are wrong for a while. Comparing detectors therefore needs more than one
number. This project generates series with trend, a daily and a weekly cycle and
AR(1) noise, injects four labelled anomaly types (spike, level shift, variance
change, missing block), decomposes each series STL-style in numpy, and runs three
detectors that all return per-timestamp scores plus a boolean mask so they can be
scored on the same footing. Evaluation reports point precision/recall, precision@k,
region (event) recall, detection delay, and recall at a fixed false-positive budget,
with every threshold calibrated on an anomaly-free prefix of its own series.

## What is in the box

| Module | Job |
| --- | --- |
| `src/anomaly/config.py` | Typed, validated config: generator shape, anomaly magnitudes, detector hyperparameters, evaluation knobs |
| `src/anomaly/generator.py` | Trend + daily/weekly seasonality + AR(1) noise, four injected anomaly types at non-overlapping positions, per-event ground truth, and an anomaly-free calibration prefix |
| `src/anomaly/decomposition.py` | STL-style decomposition in numpy: reflect-padded centred moving average trend, phase means for the seasonal term, iterative refinement, optional bisquare robustness, MAD scale, strength diagnostics |
| `src/anomaly/detectors_base.py` | The detector contract: `DetectionResult(name, scores, mask, threshold, fp_budget)`, quantile threshold calibration from clean scores, score sanitising, mask helper |
| `src/anomaly/detector_zscore.py` | Causal rolling z-score against the previous `window` points (self-excluded so a spike cannot dilute its own score) |
| `src/anomaly/detector_ewma.py` | EWMA control-chart detector: exponentially weighted level, robust centre from the calibration window, control-band width `sigma * sqrt(alpha / (2 - alpha))`, deviation score |
| `src/anomaly/detector_iforest.py` | Isolation-forest baseline on sliding windows, fitted on calibration windows only, window scores aligned to the timestamp at the window's end |
| `src/anomaly/evaluation.py` | Point precision/recall/F1, precision@k, threshold at a false-positive budget, region recall, detection delays, per-detector aggregation, comparison table |
| `src/anomaly/pipeline.py` | Detector registry, per-series and per-detector execution, metric collection, JSON persistence, console report |
| `src/anomaly/run.py` | CLI: run the whole benchmark and write `outputs/results.json` |

## Quickstart

```bash
pip install -r requirements.txt

# Generates the labelled series, runs all three detectors on all series,
# prints the comparison table and writes outputs/results.json.
python -m src.anomaly.run --config config.yaml

# The suite.
make test
```

Everything is driven by `config.yaml`: series count and length, seasonality
periods, anomaly rate and magnitudes, detector windows and the false-positive
budget used for calibration. The default run (12 series x 1,500 points, 120-tree
isolation forest) takes well under a minute on one CPU core.

## Design notes

**The calibration window is clean by construction, and thresholds are per series.**
Detector thresholds are not tuned on the labels — that would make the evaluation
circular. Instead the generator keeps the first 30% of every series anomaly-free,
and each detector scores that prefix with itself to pick a threshold at the
`1 - fp_budget` quantile of clean scores. Per-series calibration matters because
two series with the same detector can differ in noise scale by a factor of three;
one global threshold would either over-alert on the quiet series or miss everything
on the noisy one. When the prefix is too short to resolve the requested budget, the
maximum clean score is used, which under-alerts rather than over-alerts.

**A detector returns scores and a mask, never just a decision.** Point scores are
needed for precision@k and ranking; the mask is what a monitoring system would
actually page on. Both are per timestamp and aligned to the input length, which
takes care in two places: the rolling z-score excludes the current point (a spike
that includes itself raises its own reference mean), and the isolation forest scores
windows, whose scores are mapped to the window's end timestamp with the first
`window - 1` positions backfilled from the first window. Without that alignment,
delay and region metrics would be off by a window length.

**The three detectors are there because they fail differently.** The rolling z-score
is the cheap, causal baseline that catches spikes and short excursions. EWMA averages
over time, so it misses a one-point spike but reacts to a sustained level shift that
a point z-score explains away as a single unusual observation. Isolation forest sees
a window shape rather than a level, so it can flag a variance change that leaves the
mean almost unchanged. Reporting them side by side — including the one that loses on
each anomaly type — is the point of the benchmark.

**Point recall and event recall are both reported, and delay is measured.** A
detector that flags one point in a 40-point level shift scores 1.0 precision and
0.025 point recall but has still found the event, so region recall answers "did we
catch it at all" while point recall answers "how much of it". Delay counts points
from the event start to the first flag inside `delay_window` points; a missed event
counts as -1 and is excluded from the mean delay rather than dragging it up.

## Tests

44 tests across 6 files:

| File | Tests | What they assert |
| --- | --- | --- |
| `test_config.py` | 7 | Default config loads; per-section overrides; missing sections named; invalid calibration fraction, EWMA alpha and anomaly kinds rejected; multiple problems reported in one error |
| `test_generator.py` | 10 | Same seed is bit-identical; calibration prefix contains no labelled points; every anomaly type appears exactly once per event; injected regions are bounded; events do not overlap; missing blocks are NaN; AR(1) noise has the expected autocorrelation; trend/seasonal components match analytic values |
| `test_decomposition.py` | 8 | Interpolation of gaps; centred moving average preserves constants; trend of a linear ramp; seasonal phase means sum to zero; `trend + seasonal + residual == observed` to 1e-9; robust weights shrink a 40-sigma spike; MAD scale exact on a known vector; residual scores are non-negative and finite |
| `test_detectors.py` | 8 | The `DetectionResult` contract; quantile threshold from clean scores; z-score is causal and spikes score high; EWMA recursion and control-band formula; EWMA catches an injected level shift; window construction and end-alignment for the forest; all detectors run through the registry with aligned score/mask shapes |
| `test_evaluation.py` | 6 | Contiguous regions; hand-computed precision/recall; precision@k on fixed scores; FP-budget threshold; region recall with a half-missed mask; delays including a missed event; aggregation and comparison-table ordering |
| `test_pipeline_cli.py` | 5 | Experiment produces one row per detector per series; comparison table has three detectors sorted by F1; metrics rows agree with detector results; JSON round-trip; the CLI writes a complete results file on a temp config |

Run them with `python -m pytest -q`; no network, no GPU, no model downloads, no
fixture data on disk.

## Limitations

- The series are synthetic and the anomalies are injected by hand, so the detectors
  are being measured against mechanisms their designers already know about. Numbers
  here are not evidence about any real monitoring feed.
- Seasonality periods are configured, not discovered. There is no periodogram or
  autocorrelation search, and STL is applied with one fixed period.
- Univariate only. No cross-series correlation, no multivariate detectors, no
  point-vs-collective distinction beyond event regions.
- Missing blocks are interpolated before decomposition; the interpolated line is
  what detectors score. A missing block is therefore detected mainly as a
  perturbation of the local shape, not as "data stopped arriving". The label still
  counts it as anomalous, which makes that anomaly type harder than the others.
- Thresholds assume the clean calibration prefix is representative of the future
  noise level. A change in noise scale invalidates the threshold until recalibration.
- The isolation forest is fitted on calibration windows only and uses
  `contamination="auto"`; its scores are only as comparable as the quantile
  calibration makes them.
- Everything is batch. There is no streaming state, no incremental update, no
  false-positive suppression across events, and no alarm de-duplication.
- Detection delay is only counted within `delay_window` points of an event start;
  later detections count as misses for the delay metric even if point recall sees
  them.

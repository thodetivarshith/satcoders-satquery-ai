# SatQuery AI — Benchmarks

## Purpose

This document defines the benchmarking and evaluation methodology for the SatQuery AI system.

The benchmark is used to measure system performance across model accuracy, grounding quality, confidence calibration, and end-to-end latency.

Final benchmark values will be added after the integrated SatQuery AI pipeline is evaluated on the designated test datasets.

## Evaluation Metrics

The following metrics will be used to evaluate the SatQuery AI system:

| Metric | Purpose |
|---|---|
| Accuracy | Measures the correctness of model predictions. |
| Precision | Measures the correctness of positive predictions. |
| Recall | Measures how many relevant positive cases are identified. |
| F1 Score | Provides a balance between precision and recall. |
| Grounding Accuracy | Measures the correctness of visual grounding results. |
| Confidence Calibration | Measures how well model confidence aligns with prediction correctness. |
| End-to-End Latency | Measures the total time taken by the system to process a query and produce a response. |

## Benchmark Methodology

The benchmarking process follows these stages:

1. Prepare the designated evaluation dataset and ground-truth data.
2. Run the integrated SatQuery AI system on the evaluation queries.
3. Collect the model predictions and system outputs.
4. Compare predictions with the corresponding ground-truth data.
5. Calculate the required evaluation metrics.
6. Measure end-to-end system latency.
7. Store the evaluation results for analysis and reporting.
8. Document the final benchmark results.

The same evaluation procedure will be used for final benchmark reporting to ensure consistent measurement.

## Benchmark Results

Final benchmark results will be recorded in the following format:

| Dataset | Task | Accuracy | Precision | Recall | F1 Score | Latency |
|---|---|---:|---:|---:|---:|---:|
| VRSBench | Visual Question Answering / Grounding | TBD | TBD | TBD | TBD | TBD |
| RSVQA | Remote Sensing VQA | TBD | TBD | TBD | TBD | TBD |
| CDVQA | Change Detection VQA | TBD | TBD | TBD | TBD | TBD |

> **Note:** `TBD` indicates that the final benchmark result has not yet been measured.

## Final Reporting

The final benchmark report will include:

- Dataset-wise evaluation results
- Model prediction performance
- Visual grounding performance
- Confidence calibration results
- End-to-end latency measurements
- Key observations and limitations

All reported values will be based on actual evaluation runs and recorded results.
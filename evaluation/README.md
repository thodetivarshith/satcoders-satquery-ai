# Evaluation & Benchmarking

This directory contains the evaluation and benchmarking framework for the SatQuery AI system.

The evaluation module is responsible for measuring model performance, validating predictions, running automated tests, and maintaining benchmark results.

## Evaluation Metrics

The evaluation framework uses the following metrics to measure prediction performance:

- **Accuracy** — Measures the proportion of correct predictions.
- **Precision** — Measures how many predicted positive results are actually positive.
- **Recall** — Measures how many actual positive results are correctly identified.
- **F1 Score** — Provides a balance between precision and recall.

These metrics are implemented in `metrics.py` and are used by `benchmark.py` during evaluation.

## Evaluation Files

| File | Purpose |
|---|---|
| `metrics.py` | Implements evaluation metric functions. |
| `test_data.py` | Contains automated tests for evaluation metrics. |
| `benchmark.py` | Runs benchmarking and generates evaluation results. |
| `datasets/` | Contains dataset split/reference files used for evaluation. |
| `results/` | Stores benchmark results and evaluation outputs. |

## Benchmark Workflow

The evaluation workflow follows these steps:

1. Prepare the evaluation dataset and ground-truth data.
2. Obtain predictions from the SatQuery AI system.
3. Compare the predictions with the ground-truth data.
4. Calculate Accuracy, Precision, Recall, and F1 Score.
5. Store the benchmark results.
6. Document and analyze the final evaluation results.

The current benchmark implementation provides the evaluation framework and sample validation. Final benchmark values will be added after the integrated SatQuery AI pipeline is evaluated on the designated test data.

## Running Tests

From the project root directory, run:

```bash
python -m pytest evaluation\test_data.py

## Dataset Evaluation

The evaluation framework is designed to work with the designated test datasets for the SatQuery AI system.

The evaluation datasets are maintained under the `evaluation/datasets/` directory using dataset split/reference files.

Final benchmark evaluation will be performed using the appropriate ground-truth data and model predictions once the integrated SatQuery AI pipeline is ready.

## Current Status

The evaluation framework currently includes:

- Metric calculation functions
- Automated unit tests
- Benchmark execution framework
- Dataset evaluation structure
- Documentation for running evaluation tests

The sample benchmark values are used only for validating the evaluation framework. They are not final SatQuery AI performance results.

Final benchmark results will be recorded after the integrated system is evaluated on the designated test data.




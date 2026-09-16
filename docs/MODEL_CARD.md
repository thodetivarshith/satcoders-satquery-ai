# SatQuery AI — Model Card

## 1. Model Overview

### Model Name

GeoChat Fine-Tuned Model

### Project

SatQuery AI — Agentic Remote Sensing Analysis for SIH 2026

### Model Type

Vision-Language Model (VLM)

### Base Model

MBZUAI/GeoChat-7B

### Fine-Tuned Model

GeoChat fine-tuned for remote-sensing and satellite-image analysis.

---

## 2. Model Purpose

The model is designed to analyze satellite and remote-sensing imagery using natural-language queries.

It enables users to ask questions about satellite images and receive AI-generated textual answers.

The model is integrated into the SatQuery AI system as the primary AI inference component.

---

## 3. Input

The model accepts:

- Satellite or remote-sensing image
- Natural-language query

Example:

```text
Input Image: Satellite image
Query: "What objects are visible in this image?"

---

## 4. Output   

The inference interface provides information such as:

- AI-generated answer
- Model identifier
- Confidence field
- Processing time

The confidence value is currently not treated as a calibrated probability and should not be interpreted as a validated confidence score until calibration is performed.

---

## 5. Model Integration

The model is accessed through the GeoChat inference interface.

The inference workflow is:

```text
Satellite Image
      ↓
Data Processing
      ↓
Query Routing
      ↓
GeoChat Inference
      ↓
Generated Answer
      ↓
Backend API
      ↓
Evaluation / Logging

The model weights and fine-tuned adapter are loaded from the runtime environment and are not stored directly in the GitHub repository.

---

## 6. Intended Use

The model is intended for:

- Remote-sensing image analysis
- Satellite-image question answering
- Natural-language interaction with satellite imagery
- AI-assisted geospatial analysis
- Research and educational demonstrations

---

## 7. Out-of-Scope Use

The model should not be treated as an authoritative source for:

- Safety-critical decisions
- Legal decisions
- Military or defense decisions
- Emergency response decisions
- Precise geospatial measurements without independent verification

AI-generated answers should be validated when used for high-impact applications.

---

## 8. Evaluation

The SatQuery AI evaluation framework is designed to measure system performance using quantitative metrics.

Current evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- End-to-End Latency

Additional evaluation dimensions include:

- Visual Grounding Accuracy
- Confidence Calibration

### Evaluation Datasets

The planned evaluation includes:

- VRSBench
- RSVQA
- CDVQA

Final dataset-based performance values will be added after running the integrated SatQuery AI system against the designated evaluation datasets.

---

## 9. Current Evaluation Status

The evaluation framework and automated metric tests have been implemented.

Current status:

| Component | Status |
|---|---|
| Accuracy metric | Implemented |
| Precision metric | Implemented |
| Recall metric | Implemented |
| F1 Score metric | Implemented |
| Benchmark framework | Implemented |
| Automated evaluation tests | Implemented |
| CI workflow | Configured |
| Integrated dataset evaluation | Pending |
| Final benchmark results | Pending |

No final model performance numbers are reported until they are obtained from actual evaluation runs.

---

## 10. Limitations

Current limitations include:

- Final benchmark performance has not yet been established on the integrated system.
- Confidence calibration has not yet been validated.
- Visual grounding performance requires appropriate ground-truth evaluation.
- Model inference may require significant computational resources.
- Performance may vary depending on the input imagery, query type, and runtime environment.
- AI-generated answers may contain errors and should be independently verified for critical applications.

---

## 11. Reproducibility

The project maintains evaluation-related code under the `evaluation/` directory.

The evaluation framework includes:

```text
evaluation/
├── datasets/
├── results/
├── benchmark.py
├── metrics.py
├── inference_adapter.py
├── test_data.py
└── __init__.py

---

## 12. Future Improvements

Planned improvements include:

- Running complete benchmark evaluations on VRSBench, RSVQA, and CDVQA.
- Adding dataset-specific evaluation pipelines.
- Evaluating visual grounding performance.
- Calibrating model confidence scores.
- Expanding integration and inference tests.
- Improving experiment tracking and performance monitoring.
- Updating this model card with final benchmark results and observations.

---

## 13. Evaluation Results

Final results will be added after integrated evaluation.

| Dataset | Task | Accuracy | Precision | Recall | F1 Score | Latency |
|---|---|---:|---:|---:|---:|---:|
| VRSBench | Remote-Sensing VQA / Grounding | TBD | TBD | TBD | TBD | TBD |
| RSVQA | Remote-Sensing VQA | TBD | TBD | TBD | TBD | TBD |
| CDVQA | Change Detection VQA | TBD | TBD | TBD | TBD | TBD |

**Note:** `TBD` indicates that the corresponding metric has not yet been measured on the integrated system.

---

## 14. Model Card Update Policy

This model card will be updated when significant changes are made to:

- The underlying model
- Fine-tuning configuration
- Inference pipeline
- Evaluation methodology
- Benchmark datasets
- Reported performance results

Final performance values will be based only on reproducible evaluation runs.

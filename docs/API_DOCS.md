# SatQuery AI — API Documentation

## 1. Overview

SatQuery AI provides a backend API for submitting satellite imagery and natural-language queries for AI-based analysis.

The API acts as the communication layer between the frontend and the AI inference pipeline.

---

## 2. Base URL

For local development:

```text
http://localhost:8000

---

## 3. Health Check

### Endpoint

```text
GET /api/health

### Purpose

Checks whether the SatQuery AI backend service is running.

### Example Response

```json
{
  "status": "ok"
}

## 4. Image Analysis

### Endpoint

```text
POST /api/analyze

### Purpose

Submits a satellite image and a natural-language query for AI-based analysis.

### Request

The request uses `multipart/form-data`.

### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | Satellite or remote-sensing image |
| `query` | String | Yes | Natural-language question about the image |

### Example Request

```text
POST /api/analyze
Content-Type: multipart/form-data

file: satellite_image.jpg
query: "What objects are visible in this image?"

## 5. Response

A successful analysis response contains information such as:

```json
{
  "filename": "satellite_image.jpg",
  "query": "What objects are visible in this image?",
  "ai_answer": "Generated answer from the AI model",
  "confidence": 0.0,
  "model": "geochat_v1_bigearth",
  "processing_time_ms": 1234,
  "status": "success"
}

### Response Fields

| Field | Description |
|---|---|
| `filename` | Name of the uploaded image |
| `query` | User's natural-language query |
| `ai_answer` | AI-generated answer |
| `confidence` | Confidence field returned by the inference interface |
| `model` | Identifier of the model used |
| `processing_time_ms` | Processing time in milliseconds |
| `status` | Status of the request |

**Note:** The current confidence value should not be interpreted as a calibrated probability until confidence calibration is performed.

---

## 6. API Processing Flow

The analysis request follows this flow:

```text
Frontend
   ↓
POST /api/analyze
   ↓
FastAPI Backend
   ↓
Image Validation & Temporary Storage
   ↓
GeoChat Inference
   ↓
AI Answer + Metadata
   ↓
Backend Response
   ↓
Frontend

---

## 7. Evaluation Integration

The evaluation layer can consume the `/api/analyze` response through the evaluation inference adapter.

The adapter extracts:

- Query
- AI prediction
- Confidence field
- Processing time
- Model identifier
- Request status

These outputs can be used for benchmarking and system-level evaluation.

---

## 8. Error Handling

The API should return an appropriate HTTP error response when:

- An image is not provided.
- A query is missing.
- The uploaded image cannot be processed.
- Model inference fails.
- An internal backend error occurs.

The exact error response format may depend on the backend implementation.

---

## 9. Local Testing

Start the backend using the project's configured FastAPI application.

Example:

```bash
uvicorn backend.app:app --reload

The API can then be accessed at:

```text
http://localhost:8000

FastAPI's interactive API documentation is normally available at:

```text
http://localhost:8000/docs

---

## 10. Evaluation Adapter

The evaluation adapter is located at:

evaluation/inference_adapter.py

It is responsible for sending an image and query to the backend analysis endpoint and converting the API response into a standardized evaluation record.

The adapter is intended to connect the integrated backend inference pipeline with the evaluation framework.

---

## 11. Current Integration Status

| Component | Status |
|---|---|
| FastAPI backend | Implemented |
| Analysis endpoint | Implemented |
| GeoChat inference interface | Implemented |
| Evaluation adapter | Implemented |
| Automated evaluation tests | Implemented |
| End-to-end evaluation | Pending integrated testing |
| Final benchmark results | Pending |

Final API behavior and response fields should be re-verified after the backend and model branches are fully integrated.

# SatQuery AI — Deployment Guide

## 1. Overview

SatQuery AI is designed as a modular application consisting of a frontend, FastAPI backend, AI inference components, and evaluation infrastructure.

This document describes the basic deployment and runtime setup for the project.

---

## 2. Deployment Components

The main deployment components are:

- Frontend application
- FastAPI backend
- GeoChat inference module
- Required Python dependencies
- Docker configuration
- GitHub Actions CI workflow

---

## 3. Prerequisites

Before running the application, ensure that the required environment is available.

### Software Requirements

- Python 3.10 or later
- Git
- Docker and Docker Compose (for containerized deployment)

### Model Requirements

The GeoChat model and fine-tuned adapter must be available in the configured runtime environment.

Model files are not stored directly in the GitHub repository.

---

## 4. Local Deployment

Clone the repository:

```text
git clone <repository-url>
cd satcoders-satquery-ai

Create and activate a Python virtual environment:

python -m venv .venv

On Windows:

.venv\Scripts\activate

---

Install the required Python dependencies:

pip install -r backend/requirements.txt

---

## 5. Run the Backend

Start the FastAPI backend using:

uvicorn backend.app:app --reload

The backend will be available locally at:

http://localhost:8000

FastAPI's interactive API documentation can be accessed at:

http://localhost:8000/docs

---

## 6. Docker Deployment

The project includes Docker configuration for containerized deployment.

Build and start the services using:

docker compose up --build

To run the services in the background:

docker compose up -d --build

To stop the running services:

docker compose down

---

## 7. Environment Configuration

Model and runtime configuration should be provided through environment variables or the deployment environment.

Important configuration may include:

- GeoChat base model
- Fine-tuned model adapter path
- Model repository path
- Backend configuration
- Runtime settings

Sensitive credentials and private configuration values should not be committed to the GitHub repository.

---

## 8. CI Workflow

GitHub Actions is used for continuous integration.

The CI workflow automatically runs evaluation tests when configured repository changes are pushed or pull requests are created.

The evaluation workflow helps verify that changes do not break the existing evaluation framework.

---

## 9. Evaluation Deployment

The evaluation components are maintained under:

evaluation/

The evaluation framework can be executed using:

python -m evaluation.benchmark

Automated evaluation tests can be executed using:

python -m pytest evaluation/test_data.py -v

Final dataset-based benchmark results will be added after the integrated model and backend are evaluated using the designated datasets.

---

## 10. Deployment Status

| Component | Status |
|---|---|
| Backend deployment configuration | Implemented |
| Docker configuration | Available |
| CI workflow | Configured |
| Evaluation tests | Implemented |
| GeoChat inference | Implemented |
| Integrated deployment | Pending final integration testing |
| Production deployment | Pending |

---

## 11. Deployment Notes

The current deployment setup is intended for development, testing, and evaluation of the SatQuery AI system.

The complete deployment should be verified after the frontend, backend, data-processing, query-routing, GeoChat, and evaluation components are fully integrated.

Production deployment will require additional validation of:

- Model availability
- API connectivity
- Runtime dependencies
- Resource requirements
- Security configuration
- End-to-end system performance

Final deployment details may be updated based on the integrated system and deployment environment.
# SatQuery AI — System Architecture

## 1. Overview

SatQuery AI is an AI-powered remote sensing analysis system designed to answer natural-language queries over satellite imagery.

The system combines image preprocessing, query routing, GeoChat-based analysis, visual grounding, result fusion, and evaluation.

---

## 2. High-Level Architecture

```text
User
  │
  ▼
Frontend
  │
  ▼
FastAPI Backend
  │
  ├── Image / Data Preprocessing
  │
  ├── Query Router
  │
  └── AI Model Layer
          │
          ├── GeoChat
          └── Grounding
  │
  ▼
Result Fusion
  │
  ▼
Final Answer
  │
  ▼
Evaluation & Benchmark Logging

## 3. Main Components

### 3.1 Frontend

The frontend provides the user interface for interacting with SatQuery AI.

Responsibilities:

- Uploading satellite imagery
- Accepting natural-language queries
- Sending user requests to the backend
- Displaying AI-generated answers
- Displaying visual grounding results when available

---

### 3.2 Backend

The backend acts as the central API layer of SatQuery AI.

Responsibilities:

- Receiving image and query inputs
- Validating incoming requests
- Connecting the frontend with AI services
- Invoking the appropriate analysis pipeline
- Returning model results to the frontend
- Providing execution information for evaluation

---

### 3.3 Data Processing

The data-processing layer prepares satellite imagery for downstream analysis.

Responsibilities:

- Reading remote-sensing image data
- Image preprocessing
- Preparing model-compatible inputs
- Maintaining required image metadata

---

### 3.4 Query Routing

The query-routing component determines the appropriate analysis path based on the user's query.

It helps identify the required task, such as:

- Visual Question Answering
- Object or location-based queries
- Visual grounding-related queries

---

### 3.5 GeoChat Model

The GeoChat model performs natural-language analysis of satellite imagery.

The fine-tuned GeoChat inference interface accepts:

- Satellite image
- Natural-language query

It produces information such as:

- Generated answer
- Model identifier
- Confidence field
- Processing time

The fine-tuned model adapter and base model are loaded from the runtime environment rather than being stored directly in the repository.

---

### 3.6 Visual Grounding

The visual grounding component is used for queries that require locating objects or regions within satellite imagery.

Grounding results may include bounding-box information representing the detected region.

---

### 3.7 Evaluation

The evaluation component measures the performance of the SatQuery AI system.

Current evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- End-to-End Latency

The evaluation framework also supports benchmark result storage and automated testing.

---

###3.8 MLOps

The MLOps component supports reliable development and testing of the project.

It includes:

- GitHub Actions
- Automated evaluation tests
- CI validation
- Reproducible benchmark execution
- Evaluation result tracking

## 4. Data Flow

The SatQuery AI system follows a sequential data flow from user input to AI-generated output.

### 4.1 User Input

The user provides:

- Satellite imagery
- Natural-language query

The request is submitted through the frontend.

### 4.2 Backend Request

The frontend sends the image and query to the backend API.

The backend validates the request and prepares it for further processing.

### 4.3 Data Processing

The satellite image is processed and prepared in a format suitable for downstream AI analysis.

### 4.4 Query Routing

The query-routing component analyzes the user's query and determines the appropriate analysis path.

### 4.5 AI Model Inference

The processed image and query are passed to the GeoChat inference module.

The model generates an answer along with available inference metadata such as:

- Model identifier
- Confidence field
- Processing time

### 4.6 Evaluation and Logging

The generated output and execution information can be collected by the evaluation layer for benchmarking and performance analysis.

### 4.7 Response

The backend returns the structured result to the frontend.

The frontend displays the generated answer and available visual grounding information to the user.

### Overall Flow

User
→ Frontend
→ Backend API
→ Data Processing
→ Query Routing
→ GeoChat Model
→ Evaluation / Logging
→ Backend Response
→ Frontend

## 5. API Flow

The backend API provides communication between the frontend and the AI inference layer.

### 5.1 Analysis Request

The frontend sends an analysis request containing:

- Satellite image
- Natural-language query

The request is received by the backend analysis endpoint.

### 5.2 Backend Processing

The backend:

1. Receives the uploaded image and query.
2. Validates the request.
3. Temporarily stores the uploaded image.
4. Invokes the GeoChat inference module.
5. Collects the generated result and execution metadata.

### 5.3 API Response

The backend returns a structured response containing available information such as:

- Filename
- Query
- AI-generated answer
- Confidence
- Model identifier
- Processing time
- Request status

### 5.4 Evaluation Integration

The evaluation layer can consume the backend response to record model outputs and execution information for benchmarking.

This allows evaluation to remain separated from the core inference implementation while still measuring the integrated system.

### API Flow

Frontend
→ POST analysis request
→ Backend API
→ GeoChat Inference
→ Structured Response
→ Frontend

Backend Response
→ Evaluation Adapter
→ Benchmarking / Metrics

## 6. Deployment Architecture

SatQuery AI is designed as a modular application where the frontend, backend, AI model, and evaluation components can operate as separate layers.

### 6.1 Runtime Components

The main runtime components are:

- Frontend application
- FastAPI backend
- Data-processing modules
- Query-routing component
- GeoChat inference module
- Visual grounding component
- Evaluation framework

### 6.2 Model Runtime

The GeoChat model and fine-tuned model adapter are loaded from the runtime environment.

The model weights are not required to be stored directly in the GitHub repository.

### 6.3 Containerization

The project includes Docker configuration to support reproducible deployment and environment setup.

Docker can be used to package the application and its required dependencies into a consistent runtime environment.

### 6.4 Continuous Integration

GitHub Actions is used for continuous integration.

The CI workflow can automatically:

- Install required dependencies
- Run evaluation tests
- Validate project changes
- Provide automated feedback on repository updates

### 6.5 Deployment Flow

Source Code
→ GitHub Repository
→ CI Validation
→ Build / Environment Setup
→ Application Deployment
→ Runtime Inference
→ Evaluation and Monitoring

## 7. Evaluation and MLOps Workflow

The Evaluation and MLOps workflow ensures that SatQuery AI can be tested, benchmarked, and validated in a reproducible manner.

### 7.1 Evaluation Workflow

The evaluation process follows these steps:

1. Prepare evaluation data and ground-truth answers.
2. Provide images and queries to the integrated SatQuery AI system.
3. Collect model predictions and execution metadata.
4. Compare predictions with the ground truth.
5. Calculate the required evaluation metrics.
6. Store benchmark results for analysis and reporting.

### 7.2 Evaluation Metrics

The evaluation framework currently includes:

- Accuracy
- Precision
- Recall
- F1 Score
- End-to-End Latency

Additional evaluation dimensions planned for the integrated system include:

- Visual Grounding Accuracy
- Confidence Calibration

### 7.3 Automated Testing

Unit tests are maintained under the `evaluation/` directory.

The current test suite validates the implemented evaluation metrics and helps detect regressions when evaluation code is modified.

### 7.4 Continuous Integration

GitHub Actions is used to automate evaluation testing.

The CI workflow:

1. Checks out the repository.
2. Sets up the required Python environment.
3. Installs evaluation dependencies.
4. Runs the evaluation test suite.
5. Reports the test status.

### 7.5 Benchmark Results

Benchmark results are stored under:

`evaluation/results/`

Final dataset-based performance values will be added after running the integrated system against the designated evaluation datasets.

No performance value is reported as a final model result until it is obtained from an actual evaluation run.

### 7.6 Reproducibility

The evaluation workflow is designed to provide reproducible testing by keeping:

- Evaluation code
- Metric implementations
- Test cases
- Benchmark configuration
- Result files

under version control.

## 8. Repository Structure

The repository is organized into separate modules to maintain a clear separation of responsibilities.

```text
satcoders-satquery-ai/
│
├── backend/
│   ├── app.py
│   └── routes/
│
├── data_processing/
│
├── demo/
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── BENCHMARKS.md
│   └── SETUP.md
│
├── evaluation/
│   ├── datasets/
│   ├── results/
│   ├── benchmark.py
│   ├── metrics.py
│   ├── inference_adapter.py
│   ├── test_data.py
│   └── __init__.py
│
├── frontend/
│
├── models/
│   └── geochat_finetuned/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── docker-compose.yml
├── README.md
└── QUICKSTART.md

## 9. Limitations and Future Improvements

### 9.1 Current Limitations

The current system has the following limitations:

- Final dataset-based benchmark results are still pending integrated system evaluation.
- Confidence calibration requires validated confidence outputs from the integrated model.
- Visual grounding performance requires evaluation using appropriate grounding ground truth.
- End-to-end performance depends on the runtime environment and model inference resources.
- Large AI models may require significant computational resources for inference.

### 9.2 Future Improvements

Planned improvements include:

- Running comprehensive benchmarks on the designated evaluation datasets.
- Adding dataset-specific evaluation pipelines.
- Improving visual grounding evaluation.
- Calibrating and validating model confidence scores.
- Expanding automated integration and model inference tests.
- Improving CI/CD validation and deployment workflows.
- Adding detailed performance monitoring and experiment tracking.
- Improving documentation based on final integrated-system results.

### 9.3 Evaluation Roadmap

The evaluation roadmap is:

1. Complete integration of the backend and AI inference pipeline.
2. Connect the evaluation adapter to the integrated API.
3. Run benchmark datasets through the complete system.
4. Calculate and record evaluation metrics.
5. Analyze latency and other system-level metrics.
6. Document final benchmark results.
7. Use the results for final reporting and presentation.

## 10. Conclusion

SatQuery AI provides a modular architecture for intelligent analysis of remote-sensing imagery using natural-language queries.

The system separates frontend interaction, backend API processing, data preparation, query routing, AI inference, visual grounding, evaluation, and MLOps into dedicated components.

The evaluation and MLOps layers provide a structured approach for testing system performance, maintaining reproducible benchmarks, and supporting continuous validation through automated CI workflows.

As the remaining system components are integrated, the evaluation framework can be used to generate final benchmark results and support quantitative analysis of the complete SatQuery AI system.

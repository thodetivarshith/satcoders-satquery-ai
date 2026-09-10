# SatQuery AI — Frontend

Agentic remote-sensing analysis UI for SIH26167 (team SatCoders). Upload a
satellite image, ask a natural-language question, and see the AI's answer,
confidence, visual evidence, and execution trace.

## Setup

```bash
npm install
cp .env.example .env   # point VITE_API_URL at your FastAPI backend
npm run dev
```

`npm run build` produces a production bundle in `dist/`.

## Backend contract

The frontend posts to `POST {VITE_API_URL}/api/analyze` with `multipart/form-data`
fields `image` and `query`, and expects:

```json
{
  "query": "...",
  "answer": "...",
  "confidence": 0.87,
  "task_type": "vqa",
  "execution_trace": {
    "router_decision": "vqa",
    "models_called": ["geochat"],
    "processing_time_ms": 1234
  },
  "bounding_boxes": [],
  "metadata": {}
}
```

All fields except `answer` and `confidence` are optional — the UI degrades
gracefully when they're missing.

`bounding_boxes` items are currently assumed to carry normalized (0–1)
coordinates as `x_min`/`y_min`/`x_max`/`y_max` plus `label` and `confidence`
(see `src/components/BoundingBoxOverlay.jsx`). Confirm this shape with the
CV/grounding lead — if the real API differs, only `toPercentRect()` in that
file needs to change.

## Structure

```
src/
├── components/     UI building blocks (viewer, composer, result panel, trace, etc.)
├── data/samples.js Synthetic demo scenes for "Try a sample scene" (no backend needed)
├── services/api.js API client — reads VITE_API_URL, normalizes errors
├── utils/formatters.js  Shared formatting helpers
└── App.jsx         Top-level state and layout
```

## Demo mode

The "Try a sample scene" strip loads a procedurally generated placeholder
image and a mocked analysis result — clearly labeled `DEMO DATA` in the UI —
so the full flow can be shown without a live backend. Replace
`src/data/samples.js`'s images with real sample scenes before a live SIH demo.

## Known limitations

- Loading-state pipeline stages advance on a timer, not real backend
  progress events (the current contract is a single request/response).
- GeoTIFF files are accepted for upload but the browser can't natively
  preview them; the UI shows a "preview unavailable" message and still
  sends the file.
- Analysis history is session-only (in memory), not persisted.

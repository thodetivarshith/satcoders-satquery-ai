import { useEffect, useRef, useState } from "react";
import Header from "./components/Header";
import Hero from "./components/Hero";
import MissionBrief from "./components/MissionBrief";
import SatelliteViewer from "./components/SatelliteViewer";
import QueryComposer from "./components/QueryComposer";
import PipelineProgress from "./components/PipelineProgress";
import ResultPanel from "./components/ResultPanel";
import SampleDemo from "./components/SampleDemo";
import AnalysisHistory from "./components/AnalysisHistory";
import ErrorBanner from "./components/ErrorBanner";
import Toast from "./components/Toast";
import { analyzeSatelliteImage } from "./services/api";

function makeId() {
  return typeof crypto !== "undefined" && crypto.randomUUID
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function App() {
  const [imageFile, setImageFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [dimensions, setDimensions] = useState(null);
  const [previewUnavailable, setPreviewUnavailable] = useState(false);
  const [demoSample, setDemoSample] = useState(null);

  const [query, setQuery] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [selectedBoxIndex, setSelectedBoxIndex] = useState(null);

  const [history, setHistory] = useState([]);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [toastMessage, setToastMessage] = useState("");

  const abortRef = useRef(null);
  const objectUrlRef = useRef(null);

  useEffect(
    () => () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      abortRef.current?.abort();
    },
    []
  );

  const handleImageSelect = (file) => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;

    setDemoSample(null);
    setImageFile(file);
    setImageUrl(url);
    setDimensions(null);
    setPreviewUnavailable(false);
    setResult(null);
    setError(null);
    setSelectedBoxIndex(null);
  };

  const handleClear = () => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
    setImageFile(null);
    setImageUrl(null);
    setDimensions(null);
    setDemoSample(null);
    setResult(null);
    setError(null);
    setSelectedBoxIndex(null);
  };

  const handleSelectSample = (sample) => {
    setDemoSample(sample);
    setImageFile({ name: `${sample.id}-sample.svg`, size: 0, type: "image/svg+xml" });
    setImageUrl(sample.image);
    setDimensions({ width: 640, height: 480 });
    setPreviewUnavailable(false);
    setQuery(sample.query);
    setResult(null);
    setError(null);
    setSelectedBoxIndex(null);
  };

  const handleReopenHistory = (entry) => {
    setDemoSample(null);
    setImageFile({ name: "Restored from history", size: 0, type: "" });
    setImageUrl(entry.imageUrl);
    setDimensions(null);
    setQuery(entry.query);
    setResult(entry.fullResult);
    setError(null);
    setSelectedBoxIndex(null);
    setHistoryOpen(false);
  };

  const handleAnalyze = async () => {
    if (!imageFile || !query.trim()) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setError(null);
    setResult(null);
    setSelectedBoxIndex(null);
    setIsAnalyzing(true);

    try {
      let data;
      if (demoSample) {
        // Explicit demo mode: no network call, clearly mocked result.
        await new Promise((resolve) => setTimeout(resolve, 2400));
        data = demoSample.mockResult;
      } else {
        data = await analyzeSatelliteImage(imageFile, query.trim(), {
          signal: controller.signal,
        });
      }

      const normalized = {
        query: data.query ?? query.trim(),
        answer: data.answer ?? "No answer returned.",
        confidence: data.confidence ?? 0,
        task_type: data.task_type ?? null,
        bounding_boxes: data.bounding_boxes ?? [],
        execution_trace: data.execution_trace ?? {},
        metadata: data.metadata ?? {},
      };

      setResult(normalized);
      setHistory((prev) =>
        [
          {
            id: makeId(),
            thumbnail: imageUrl,
            imageUrl,
            query: normalized.query,
            answer: normalized.answer,
            confidence: normalized.confidence,
            timestamp: new Date(),
            fullResult: normalized,
          },
          ...prev,
        ].slice(0, 20)
      );
    } catch (err) {
      if (err?.kind !== "cancelled") {
        console.error("Analysis failed:", err);
        setError(err);
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-bg text-text" style={{ fontFamily: "var(--font-body)" }}>
      <Header onToggleHistory={() => setHistoryOpen((v) => !v)} historyCount={history.length} />

      <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6 sm:py-8">
        <Hero />

        <div className="mt-5">
          <SampleDemo
            onSelectSample={handleSelectSample}
            disabled={isAnalyzing}
            activeSampleId={demoSample?.id}
          />
        </div>

        <div className="mt-4">
          <MissionBrief
            hasImage={Boolean(imageFile)}
            hasQuery={Boolean(query.trim())}
            isAnalyzing={isAnalyzing}
            hasResult={Boolean(result)}
          />
        </div>

        <div className="mt-5 grid gap-5 lg:grid-cols-[1.3fr_1fr] lg:items-start">
          <SatelliteViewer
            file={imageFile}
            imageUrl={imageUrl}
            dimensions={dimensions}
            onImageSelect={handleImageSelect}
            onClear={handleClear}
            onDimensionsResolved={setDimensions}
            onPreviewError={() => setPreviewUnavailable(true)}
            previewUnavailable={previewUnavailable}
            result={result}
            selectedBoxIndex={selectedBoxIndex}
            onSelectBox={setSelectedBoxIndex}
            isAnalyzing={isAnalyzing}
          />

          <div className="space-y-5">
            <QueryComposer
              value={query}
              onChange={setQuery}
              onSubmit={handleAnalyze}
              disabled={!imageFile}
              isAnalyzing={isAnalyzing}
            />

            <ErrorBanner error={error} />

            {isAnalyzing && <PipelineProgress />}

            {!isAnalyzing && result && (
              <ResultPanel
                result={result}
                selectedBoxIndex={selectedBoxIndex}
                onSelectBox={setSelectedBoxIndex}
                onCopied={setToastMessage}
              />
            )}

            {!isAnalyzing && !result && !error && (
              <section className="panel border-dashed p-6 text-center">
                <p className="text-sm text-text-muted">
                  {imageFile
                    ? "Ask a question about the selected image."
                    : "Upload a satellite image to begin analysis."}
                </p>
              </section>
            )}
          </div>
        </div>
      </main>

      <footer className="border-t border-border">
        <p className="mx-auto max-w-7xl px-4 py-4 text-xs text-text-faint sm:px-6">
          SatQuery AI · Agentic Remote-Sensing Analyst · SIH26167
        </p>
      </footer>

      {historyOpen && (
        <>
          <button
            type="button"
            aria-label="Close history overlay"
            className="fixed inset-0 z-10 bg-black/50"
            onClick={() => setHistoryOpen(false)}
          />
          <AnalysisHistory
            entries={history}
            onReopen={handleReopenHistory}
            onClose={() => setHistoryOpen(false)}
          />
        </>
      )}

      <Toast message={toastMessage} onDismiss={() => setToastMessage("")} />
    </div>
  );
}

export default App;

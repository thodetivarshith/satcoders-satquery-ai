import { useState } from "react";
import ConfidenceGauge from "./ConfidenceGauge";
import ExecutionTrace from "./ExecutionTrace";
import { BoundingBoxLegend } from "./BoundingBoxOverlay";
import { formatDuration, formatTaskType } from "../utils/formatters";

/**
 * Primary/expandable split (brief §18): the answer is always visible
 * in plain language; "how SatQuery reached this answer" is tucked
 * behind a toggle for judges who want the technical depth.
 */
function ResultPanel({ result, selectedBoxIndex, onSelectBox, onCopied }) {
  const [traceOpen, setTraceOpen] = useState(false);
  if (!result) return null;

  const processingMs = result.execution_trace?.processing_time_ms;
  const isDemo = Boolean(result.metadata?.demo);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(result.answer);
      onCopied?.("Answer copied to clipboard");
    } catch {
      onCopied?.("Couldn't copy — select the text manually");
    }
  };

  return (
    <section className="rise-in panel panel-lg p-4 sm:p-5">
      <div className="flex items-center justify-between gap-2">
        <h2 className="panel-title">AI answer</h2>
        <div className="flex items-center gap-2">
          {isDemo && (
            <span className="rounded border border-sar/40 bg-sar/10 px-2 py-0.5 text-[10px] font-medium tracking-wide text-sar">
              DEMO DATA
            </span>
          )}
          <button
            type="button"
            onClick={handleCopy}
            className="rounded border border-border px-2 py-1 text-[10px] text-text-muted transition hover:border-border-strong hover:text-text"
          >
            Copy
          </button>
        </div>
      </div>

      <p className="mt-3 text-base leading-7 text-text">{result.answer}</p>

      <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {[
          <div
            key="confidence"
            className="flex flex-col items-center justify-center rounded-lg border border-border-hair bg-bg-elevated/60 px-2 py-3"
          >
            <ConfidenceGauge confidence={result.confidence} />
            <span className="mt-1 text-[10px] uppercase tracking-wide text-text-faint">
              Confidence
            </span>
          </div>,
          <Stat key="task" label="Task" value={formatTaskType(result.task_type)} />,
          <Stat key="model" label="Model" value="GeoChat v1" sub="BigEarthNet" />,
          <Stat key="processing" label="Processing" value={formatDuration(processingMs)} />,
        ].map((node, index) => (
          <div key={node.key} className="rise-in" style={{ animationDelay: `${index * 70}ms` }}>
            {node}
          </div>
        ))}
      </div>

      {result.bounding_boxes?.length > 0 && (
        <div className="rise-in mt-4" style={{ animationDelay: "260ms" }}>
          <p className="mb-2 text-xs font-medium uppercase tracking-wide text-text-muted">
            Visual evidence
          </p>
          <BoundingBoxLegend
            boxes={result.bounding_boxes}
            selectedIndex={selectedBoxIndex}
            onSelect={onSelectBox}
          />
        </div>
      )}

      <div className="mt-4 border-t border-border pt-3">
        <button
          type="button"
          onClick={() => setTraceOpen((v) => !v)}
          className="flex w-full items-center justify-between text-left text-xs font-medium text-text-muted transition hover:text-text"
          aria-expanded={traceOpen}
        >
          <span>How SatQuery reached this answer</span>
          <span className={`transition-transform ${traceOpen ? "rotate-180" : ""}`}>⌄</span>
        </button>
        {traceOpen && (
          <div className="rise-in mt-3">
            <ExecutionTrace query={result.query} trace={result.execution_trace} />
          </div>
        )}
      </div>
    </section>
  );
}

function Stat({ label, value, sub }) {
  return (
    <div className="flex h-full flex-col justify-center rounded-lg border border-border-hair bg-bg-elevated/60 px-3 py-3">
      <span className="text-[10px] uppercase tracking-wide text-text-faint">{label}</span>
      <span className="mt-1 truncate text-sm font-medium text-text" title={value}>
        {value}
      </span>
      {sub && <span className="truncate text-[10px] text-text-faint">{sub}</span>}
    </div>
  );
}

export default ResultPanel;

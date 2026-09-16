import { useEffect, useState } from "react";

// A single HTTP request/response is all the backend contract offers
// today, so this can't reflect true server-side progress. It advances
// on a fixed cadence up to the second-to-last stage, then holds there
// — honestly — until the real response actually arrives. No invented
// percentages. (Future work: stream real stage events over SSE/WS.)
const STAGES = [
  "Image received",
  "Preprocessing",
  "Query classification",
  "Running GeoChat",
  "Evidence analysis",
  "Preparing result",
];
const STEP_INTERVAL_MS = 650;

function PipelineProgress() {
  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    setActiveIndex(0);
    const timer = setInterval(() => {
      setActiveIndex((i) => Math.min(i + 1, STAGES.length - 1));
    }, STEP_INTERVAL_MS);
    return () => clearInterval(timer);
  }, []);

  return (
    <section
      className="panel p-4 sm:p-5"
      role="status"
      aria-live="polite"
    >
      <div className="flex items-center gap-3">
        <div className="relative h-8 w-8 shrink-0" aria-hidden="true">
          <span
            className="orbit-ring-a absolute inset-0 rounded-full border border-dashed border-optical/50"
            style={{ transformOrigin: "50% 50%" }}
          />
          <span className="absolute inset-[6px] rounded-full bg-optical/15" />
          <span className="pulse-dot absolute inset-[10px] rounded-full bg-optical" />
        </div>
        <h2 className="panel-title gap-0! before:hidden">Analyzing</h2>
      </div>
      <ul className="mt-3 space-y-2">
        {STAGES.map((stage, index) => {
          const state =
            index < activeIndex ? "done" : index === activeIndex ? "active" : "pending";
          return (
            <li key={stage} className="flex items-center gap-2.5 text-sm">
              <span
                className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[10px] ${
                  state === "done"
                    ? "bg-good/20 text-good"
                    : state === "active"
                      ? "border border-optical text-optical"
                      : "border border-border text-text-faint"
                }`}
                aria-hidden="true"
              >
                {state === "done" ? "✓" : state === "active" ? "◉" : "○"}
              </span>
              <span
                className={
                  state === "pending" ? "text-text-faint" : "text-text-muted"
                }
              >
                {stage}
              </span>
              {state === "active" && (
                <span className="pulse-dot ml-auto text-[10px] text-optical">live</span>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

export default PipelineProgress;

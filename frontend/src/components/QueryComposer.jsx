import { useState } from "react";

const EXAMPLE_QUERIES = [
  "What type of land cover is visible?",
  "Is there agricultural land?",
  "Where are the water bodies?",
  "Describe this scene.",
  "Identify buildings.",
  "Are there signs of change?",
];

const MAX_LENGTH = 500;
const isMac = typeof navigator !== "undefined" && /Mac/i.test(navigator.platform);

function QueryComposer({ value, onChange, onSubmit, disabled, isAnalyzing }) {
  const [localFocused, setLocalFocused] = useState(false);

  const handleKeyDown = (e) => {
    const submitCombo = (e.metaKey || e.ctrlKey) && e.key === "Enter";
    if (submitCombo) {
      e.preventDefault();
      if (!disabled && value.trim()) onSubmit();
    }
  };

  return (
    <section className="panel p-4 sm:p-5">
      <div className="flex items-center justify-between">
        <h2 className="panel-title">Ask about this image</h2>
        {value && (
          <button
            type="button"
            onClick={() => onChange("")}
            className="text-xs text-text-muted transition hover:text-text"
            disabled={isAnalyzing}
          >
            Clear
          </button>
        )}
      </div>

      <div className="mt-3 flex flex-wrap gap-1.5">
        {EXAMPLE_QUERIES.map((example, index) => (
          <button
            key={example}
            type="button"
            onClick={() => onChange(example)}
            disabled={disabled}
            className="rise-in rounded-full border border-border px-2.5 py-1 text-[11px] text-text-muted transition hover:-translate-y-0.5 hover:border-optical hover:text-text disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:translate-y-0"
            style={{ animationDelay: `${index * 40}ms` }}
          >
            {example}
          </button>
        ))}
      </div>

      <div
        className={`mt-3 rounded-lg border bg-bg-elevated/40 transition ${
          localFocused ? "border-optical/60 glow-optical" : "border-border"
        } ${disabled ? "opacity-60" : ""}`}
      >
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value.slice(0, MAX_LENGTH))}
          onFocus={() => setLocalFocused(true)}
          onBlur={() => setLocalFocused(false)}
          onKeyDown={handleKeyDown}
          rows={3}
          disabled={disabled}
          placeholder={
            disabled
              ? "Upload a satellite image to begin analysis."
              : "Ask anything about this satellite image..."
          }
          className="w-full resize-none bg-transparent p-3 text-sm text-text outline-none placeholder:text-text-faint"
          aria-label="Natural-language question about the satellite image"
        />
        <div className="flex items-center justify-between border-t border-border px-3 py-1.5 text-[11px] text-text-faint">
          <span>{isMac ? "⌘" : "Ctrl"} + Enter to analyze</span>
          <span>{value.length}/{MAX_LENGTH}</span>
        </div>
      </div>

      <button
        type="button"
        onClick={onSubmit}
        disabled={disabled || !value.trim() || isAnalyzing}
        className="btn-primary mt-3 flex w-full items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:translate-y-0 disabled:hover:shadow-none"
      >
        {isAnalyzing ? (
          <>
            <span
              className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-bg border-t-transparent"
              aria-hidden="true"
            />
            Analyzing
          </>
        ) : (
          <>✦ Analyze satellite image</>
        )}
      </button>

      {disabled && (
        <p className="mt-2 text-xs text-text-faint">
          Upload a satellite image to begin analysis.
        </p>
      )}
    </section>
  );
}

export default QueryComposer;

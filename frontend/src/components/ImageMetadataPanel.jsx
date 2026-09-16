import { useState } from "react";
import { formatBytes } from "../utils/formatters";

/**
 * Progressive-disclosure details panel: collapsed by default so the
 * primary user isn't confronted with technical fields, expandable for
 * judges/analysts who want them.
 */
function ImageMetadataPanel({ file, dimensions, backendMetadata }) {
  const [expanded, setExpanded] = useState(false);

  const backendEntries = backendMetadata
    ? Object.entries(backendMetadata).filter(([key]) => key !== "demo")
    : [];

  return (
    <div className="border-t border-border">
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="flex w-full items-center justify-between px-4 py-2.5 text-xs font-medium text-text-muted transition hover:text-text sm:px-5"
        aria-expanded={expanded}
      >
        <span>Image details</span>
        <span className={`transition-transform ${expanded ? "rotate-180" : ""}`}>
          ⌄
        </span>
      </button>

      {expanded && (
        <div className="rise-in grid gap-x-6 gap-y-2 border-t border-border px-4 py-3 text-xs sm:grid-cols-2 sm:px-5">
          <Row label="File" value={file?.name} />
          <Row
            label="Dimensions"
            value={dimensions ? `${dimensions.width} × ${dimensions.height} px` : "—"}
          />
          <Row label="Size" value={file ? formatBytes(file.size) : "—"} />
          <Row label="Format" value={file?.type || "—"} />
          {backendEntries.map(([key, value]) => (
            <Row key={key} label={key.replace(/_/g, " ")} value={String(value)} />
          ))}
        </div>
      )}
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div className="flex items-center justify-between gap-3 py-0.5">
      <span className="capitalize text-text-faint">{label}</span>
      <span
        className="truncate text-text-muted"
        style={{ fontFamily: "var(--font-mono)" }}
        title={value}
      >
        {value || "—"}
      </span>
    </div>
  );
}

export default ImageMetadataPanel;

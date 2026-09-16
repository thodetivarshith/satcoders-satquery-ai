import { formatConfidencePercent } from "../utils/formatters";

/**
 * Draws grounding boxes over the satellite image and exposes a legend.
 *
 * ASSUMED CONTRACT: each box has normalized (0-1) coordinates as
 * x_min/y_min/x_max/y_max. Confirm this with Abhinay (CV/grounding) —
 * if the real API returns pixel coordinates or [x, y, w, h] instead,
 * only `toPercentRect` below needs to change.
 */
function toPercentRect(box) {
  const xMin = box.x_min ?? box.xmin ?? 0;
  const yMin = box.y_min ?? box.ymin ?? 0;
  const xMax = box.x_max ?? box.xmax ?? xMin;
  const yMax = box.y_max ?? box.ymax ?? yMin;
  return {
    left: `${xMin * 100}%`,
    top: `${yMin * 100}%`,
    width: `${Math.max(xMax - xMin, 0) * 100}%`,
    height: `${Math.max(yMax - yMin, 0) * 100}%`,
  };
}

function BoundingBoxOverlay({ boxes, selectedIndex, onSelect }) {
  if (!boxes?.length) return null;

  return (
    <div className="pointer-events-none absolute inset-0" aria-hidden={false}>
      {boxes.map((box, index) => {
        const isSelected = selectedIndex === index;
        return (
          <button
            key={index}
            type="button"
            onClick={() => onSelect(isSelected ? null : index)}
            className="pointer-events-auto absolute rounded-sm border-2 transition-all duration-200"
            style={{
              ...toPercentRect(box),
              borderColor: isSelected ? "var(--sar)" : "rgba(245,166,35,0.55)",
              backgroundColor: isSelected
                ? "rgba(245,166,35,0.12)"
                : "transparent",
              boxShadow: isSelected ? "0 0 0 1px rgba(245,166,35,0.3)" : "none",
            }}
            aria-pressed={isSelected}
            aria-label={`${box.label || "Detected region"}, confidence ${formatConfidencePercent(box.confidence)} percent`}
          >
            <span
              className="absolute -top-6 left-0 whitespace-nowrap rounded-md border px-1.5 py-0.5 text-[10px] font-medium backdrop-blur-md"
              style={{
                fontFamily: "var(--font-mono)",
                borderColor: isSelected ? "var(--sar)" : "var(--border-strong)",
                background: isSelected ? "rgba(245,166,35,0.14)" : "rgba(19,23,31,0.75)",
                color: isSelected ? "var(--sar)" : "var(--text-muted)",
                boxShadow: isSelected ? "0 0 16px rgba(245,166,35,0.3)" : "none",
              }}
            >
              {box.label || "region"} · {formatConfidencePercent(box.confidence)}%
            </span>
          </button>
        );
      })}
    </div>
  );
}

export function BoundingBoxLegend({ boxes, selectedIndex, onSelect }) {
  if (!boxes?.length) return null;

  return (
    <ul className="grid gap-2 sm:grid-cols-2">
      {boxes.map((box, index) => {
        const isSelected = selectedIndex === index;
        return (
          <li key={index}>
            <button
              type="button"
              onClick={() => onSelect(isSelected ? null : index)}
              className={`flex w-full items-center justify-between rounded-lg border px-3 py-2 text-left text-sm transition ${
                isSelected
                  ? "border-sar/60 bg-[rgba(245,166,35,0.1)] glow-sar"
                  : "border-border-hair bg-bg-elevated/50 hover:border-border-strong"
              }`}
            >
              <span className="flex items-center gap-2">
                <span
                  className="h-2 w-2 rounded-full"
                  style={{ background: isSelected ? "var(--sar)" : "var(--border-strong)" }}
                />
                <span className="text-text">{box.label || "Detected region"}</span>
              </span>
              <span
                className="text-xs text-text-muted"
                style={{ fontFamily: "var(--font-mono)" }}
              >
                {formatConfidencePercent(box.confidence)}%
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

export default BoundingBoxOverlay;

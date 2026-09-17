import { formatConfidencePercent } from "../utils/formatters";

/**
 * Draws grounding boxes over the satellite image.
 *
 * Backend contract:
 * bbox = [x_min, y_min, x_max, y_max] in PIXEL coordinates.
 *
 * Example for an 800x800 image:
 * [121, 195, 224, 265]
 *
 * The overlay converts pixel coordinates to percentages so the
 * boxes remain correctly positioned when the image scales.
 */

function toPercentRect(box) {
  let xMin;
  let yMin;
  let xMax;
  let yMax;

  // Real backend format: bbox: [x_min, y_min, x_max, y_max]
  if (Array.isArray(box?.bbox) && box.bbox.length >= 4) {
    [xMin, yMin, xMax, yMax] = box.bbox;
  } else {
    // Backward compatibility with object-style coordinates.
    xMin = box?.x_min ?? box?.xmin ?? 0;
    yMin = box?.y_min ?? box?.ymin ?? 0;
    xMax = box?.x_max ?? box?.xmax ?? xMin;
    yMax = box?.y_max ?? box?.ymax ?? yMin;
  }

  // Backend currently processes the 800x800 demo image.
  // These values are replaced dynamically when dimensions are supplied.
  const imageWidth = box?.image_width ?? 800;
  const imageHeight = box?.image_height ?? 800;

  return {
    left: `${(xMin / imageWidth) * 100}%`,
    top: `${(yMin / imageHeight) * 100}%`,
    width: `${Math.max(xMax - xMin, 0) / imageWidth * 100}%`,
    height: `${Math.max(yMax - yMin, 0) / imageHeight * 100}%`,
  };
}

function BoundingBoxOverlay({
  boxes,
  selectedIndex,
  onSelect,
  imageWidth,
  imageHeight,
}) {
  if (!boxes?.length) return null;

  return (
    <div className="pointer-events-none absolute inset-0" aria-label="Detected regions">
      {boxes.map((box, index) => {
        const isSelected = selectedIndex === index;

        const boxWithDimensions = {
          ...box,
          image_width: imageWidth,
          image_height: imageHeight,
        };

        return (
          <button
            key={box.candidate_id || index}
            type="button"
            onClick={() => onSelect(isSelected ? null : index)}
            className="pointer-events-auto absolute rounded-sm border-2 transition-all duration-200"
            style={{
              ...toPercentRect(boxWithDimensions),
              borderColor: isSelected
                ? "var(--sar)"
                : "rgba(245,166,35,0.75)",
              backgroundColor: isSelected
                ? "rgba(245,166,35,0.12)"
                : "rgba(245,166,35,0.03)",
              boxShadow: isSelected
                ? "0 0 0 1px rgba(245,166,35,0.3)"
                : "none",
            }}
            aria-pressed={isSelected}
            aria-label={`${box.label || "Detected region"}, confidence ${formatConfidencePercent(box.confidence)} percent`}
          >
            <span
              className="absolute -top-6 left-0 whitespace-nowrap rounded-md border px-1.5 py-0.5 text-[10px] font-medium backdrop-blur-md"
              style={{
                fontFamily: "var(--font-mono)",
                borderColor: isSelected
                  ? "var(--sar)"
                  : "var(--border-strong)",
                background: isSelected
                  ? "rgba(245,166,35,0.14)"
                  : "rgba(19,23,31,0.85)",
                color: isSelected
                  ? "var(--sar)"
                  : "var(--text-muted)",
                boxShadow: isSelected
                  ? "0 0 16px rgba(245,166,35,0.3)"
                  : "none",
              }}
            >
              {box.label || "region"} ·{" "}
              {formatConfidencePercent(box.confidence)}%
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
          <li key={box.candidate_id || index}>
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
                  style={{
                    background: isSelected
                      ? "var(--sar)"
                      : "var(--border-strong)",
                  }}
                />
                <span className="text-text">
                  {box.label || "Detected region"}
                </span>
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

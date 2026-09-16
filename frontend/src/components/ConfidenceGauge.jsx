import { useEffect, useRef, useState } from "react";
import { formatConfidencePercent } from "../utils/formatters";

const prefersReducedMotion =
  typeof window !== "undefined" &&
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

function useCountUp(target, durationMs = 700) {
  const [value, setValue] = useState(prefersReducedMotion ? target : 0);
  const frameRef = useRef(null);

  useEffect(() => {
    if (prefersReducedMotion) {
      setValue(target);
      return;
    }
    const start = performance.now();
    const from = 0;
    const tick = (now) => {
      const progress = Math.min((now - start) / durationMs, 1);
      const eased = 1 - Math.pow(1 - progress, 3); // ease-out-cubic
      setValue(Math.round(from + (target - from) * eased));
      if (progress < 1) frameRef.current = requestAnimationFrame(tick);
    };
    frameRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameRef.current);
  }, [target, durationMs]);

  return value;
}

function confidenceLabel(percent) {
  if (percent >= 75) return "High confidence";
  if (percent >= 45) return "Moderate confidence";
  return "Low confidence";
}

/**
 * A small arc gauge — deliberately not a generic progress bar or a
 * bare "Confidence: 0.87" line, per the brief.
 */
function ConfidenceGauge({ confidence, size = 76, showLabel = false }) {
  const targetPercent = formatConfidencePercent(confidence);
  const animatedPercent = useCountUp(targetPercent);
  const radius = (size - 10) / 2;
  const circumference = Math.PI * radius; // half circle
  const offset = circumference * (1 - targetPercent / 100);
  const color =
    targetPercent >= 75 ? "var(--good)" : targetPercent >= 45 ? "var(--sar)" : "var(--danger)";

  return (
    <div
      className="flex flex-col items-center"
      role="img"
      aria-label={`Confidence ${targetPercent} percent, ${confidenceLabel(targetPercent)}`}
    >
      <svg width={size} height={size / 2 + 6} viewBox={`0 0 ${size} ${size / 2 + 6}`}>
        <defs>
          <linearGradient id={`gauge-grad-${size}`} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor={color} stopOpacity="0.55" />
            <stop offset="100%" stopColor={color} />
          </linearGradient>
        </defs>
        <path
          d={`M 5 ${size / 2} A ${radius} ${radius} 0 0 1 ${size - 5} ${size / 2}`}
          fill="none"
          stroke="var(--border)"
          strokeWidth="6"
          strokeLinecap="round"
        />
        <path
          d={`M 5 ${size / 2} A ${radius} ${radius} 0 0 1 ${size - 5} ${size / 2}`}
          fill="none"
          stroke={`url(#gauge-grad-${size})`}
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{
            transition: "stroke-dashoffset 0.7s ease-out",
            filter: `drop-shadow(0 0 6px ${color}66)`,
          }}
        />
      </svg>
      <span
        className="-mt-3 text-lg font-semibold text-text"
        style={{ fontFamily: "var(--font-display)" }}
      >
        {animatedPercent}%
      </span>
      {showLabel && (
        <span className="mt-0.5 text-[10px]" style={{ color }}>
          {confidenceLabel(targetPercent)}
        </span>
      )}
    </div>
  );
}

export default ConfidenceGauge;

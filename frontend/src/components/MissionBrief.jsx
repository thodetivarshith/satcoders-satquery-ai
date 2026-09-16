const STEPS = ["Upload image", "Ask question", "Analyze", "Evidence"];

/**
 * A persistent, state-driven pipeline strip. This is what lets a judge
 * understand the workflow in ~5 seconds without reading anything —
 * the highlighted node always reflects what the app is actually doing,
 * it never just decorates.
 */
function MissionBrief({ hasImage, hasQuery, isAnalyzing, hasResult }) {
  let activeIndex = 0;
  if (hasResult) activeIndex = 3;
  else if (isAnalyzing) activeIndex = 2;
  else if (hasImage && hasQuery) activeIndex = 2;
  else if (hasImage) activeIndex = 1;

  return (
    <ol
      className="flex items-center gap-1.5 overflow-x-auto panel px-3 py-2.5 sm:gap-2 sm:px-4"
      aria-label="Analysis pipeline progress"
    >
      {STEPS.map((step, index) => {
        const isDone = index < activeIndex || (hasResult && index <= activeIndex);
        const isActive = index === activeIndex && !hasResult;
        const isActiveDone = index === activeIndex && hasResult;
        return (
          <li key={step} className="flex shrink-0 items-center gap-1.5 sm:gap-2">
            <span
              className={`flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] transition-colors ${
                isDone || isActiveDone
                  ? "bg-good/15 text-good"
                  : isActive
                    ? "border border-optical text-optical"
                    : "border border-border text-text-faint"
              }`}
              aria-hidden="true"
            >
              {isDone || isActiveDone ? "✓" : index + 1}
            </span>
            <span
              className={`whitespace-nowrap text-xs ${
                isActive ? "font-medium text-text" : isDone || isActiveDone ? "text-text-muted" : "text-text-faint"
              }`}
            >
              {step}
              {isActive && isAnalyzing && index === 2 && (
                <span className="pulse-dot ml-1.5 text-optical">·</span>
              )}
            </span>
            {index < STEPS.length - 1 && (
              <span
                className={`mx-0.5 h-px w-4 sm:w-8 ${
                  index < activeIndex ? "bg-good/40" : "bg-border"
                }`}
                aria-hidden="true"
              />
            )}
          </li>
        );
      })}
    </ol>
  );
}

export default MissionBrief;

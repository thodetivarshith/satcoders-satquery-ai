import { formatConfidencePercent, formatRelativeTime, truncate } from "../utils/formatters";

/**
 * Lightweight, session-only history (per brief §9 — "keep this
 * lightweight"). No persistence layer requested or built; wire to
 * localStorage or a backend endpoint later if judges/users need
 * analyses to survive a page reload.
 */
function AnalysisHistory({ entries, onReopen, onClose }) {
  return (
    <aside
      className="rise-in fixed inset-y-0 right-0 z-20 flex w-full max-w-sm flex-col border-l border-border-hair shadow-2xl backdrop-blur-xl"
      style={{ background: "rgba(13,15,20,0.88)" }}
      aria-label="Analysis history"
    >
      <div className="flex items-center justify-between border-b border-border-hair px-4 py-3">
        <h2 className="panel-title">History</h2>
        <button
          type="button"
          onClick={onClose}
          className="text-text-muted transition hover:text-text"
          aria-label="Close history panel"
        >
          ✕
        </button>
      </div>

      <div className="scroll-thin flex-1 overflow-y-auto p-3">
        {entries.length === 0 ? (
          <p className="mt-8 text-center text-xs text-text-faint">
            Analyses you run this session will appear here.
          </p>
        ) : (
          <ul className="space-y-2">
            {entries.map((entry) => (
              <li key={entry.id}>
                <button
                  type="button"
                  onClick={() => onReopen(entry)}
                  className="flex w-full gap-3 rounded-lg border border-border-hair bg-bg-elevated/60 p-2 text-left transition hover:border-optical/40 hover:bg-bg-elevated"
                >
                  <img
                    src={entry.thumbnail}
                    alt=""
                    className="h-14 w-14 shrink-0 rounded object-cover"
                  />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-xs font-medium text-text">
                      {truncate(entry.query, 48)}
                    </p>
                    <p className="mt-0.5 truncate text-xs text-text-muted">
                      {truncate(entry.answer, 60)}
                    </p>
                    <div className="mt-1 flex items-center justify-between text-[10px] text-text-faint">
                      <span>{formatConfidencePercent(entry.confidence)}% confidence</span>
                      <span>{formatRelativeTime(entry.timestamp)}</span>
                    </div>
                  </div>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </aside>
  );
}

export default AnalysisHistory;

import { useEffect, useState } from "react";
import { getApiBaseUrl } from "../services/api";

function useUtcClock() {
  const [time, setTime] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return time.toISOString().slice(11, 19);
}

/**
 * Header keeps its own lightweight backend reachability check so the
 * status pill reflects reality instead of always saying "online."
 * A failed check never blocks the app — demo mode still works offline.
 */
function Header({ onToggleHistory, historyCount }) {
  const [backendStatus, setBackendStatus] = useState("checking");
  const utcClock = useUtcClock();

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);

    fetch(getApiBaseUrl(), { signal: controller.signal })
      .then(() => !cancelled && setBackendStatus("online"))
      .catch(() => !cancelled && setBackendStatus("unreachable"))
      .finally(() => clearTimeout(timeout));

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, []);

  const statusConfig = {
    checking: { color: "bg-text-faint", label: "CHECKING LINK" },
    online: { color: "bg-good", label: "SYSTEM ONLINE" },
    unreachable: { color: "bg-sar", label: "BACKEND UNREACHABLE" },
  }[backendStatus];

  return (
    <header className="sticky top-0 z-30 border-b border-border-hair backdrop-blur-xl" style={{ background: "rgba(7,8,11,0.72)" }}>
      <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <div className="flex items-center gap-3">
          <div
            className="glow-optical relative flex h-10 w-10 shrink-0 items-center justify-center overflow-hidden rounded-lg border border-border-strong bg-bg-elevated text-optical"
            aria-hidden="true"
          >
            <span className="radar-sweep absolute inset-0" />
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" className="relative">
              <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.6" />
              <path
                d="M12 2v4M12 18v4M2 12h4M18 12h4"
                stroke="currentColor"
                strokeWidth="1.6"
                strokeLinecap="round"
              />
            </svg>
          </div>
          <div>
            <h1
              className="text-gradient text-xl font-semibold leading-none tracking-tight"
              style={{ fontFamily: "var(--font-display)" }}
            >
              SatQuery AI
            </h1>
            <p className="mt-1.5 text-xs text-text-muted">
              Agentic Remote-Sensing Analyst
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-2 rounded-full border border-border bg-surface-raised/60 px-3 py-1.5">
            <span
              className={`h-1.5 w-1.5 rounded-full ${statusConfig.color} ${
                backendStatus === "checking" ? "pulse-dot" : ""
              }`}
              aria-hidden="true"
            />
            <span
              className="text-[11px] font-medium tracking-wide text-text-muted"
              style={{ fontFamily: "var(--font-mono)" }}
              role="status"
            >
              {statusConfig.label}
            </span>
          </div>

          <div className="hidden rounded-full border border-border bg-surface-raised/60 px-3 py-1.5 text-[11px] text-text-muted sm:block">
            GeoChat v1
            <span className="mx-1.5 text-text-faint">/</span>
            Remote Sensing AI
          </div>

          <div
            className="hidden rounded-full border border-border bg-surface-raised/60 px-3 py-1.5 text-[11px] text-text-faint md:block"
            style={{ fontFamily: "var(--font-mono)" }}
            title="UTC time"
          >
            {utcClock} UTC
          </div>

          <button
            type="button"
            onClick={onToggleHistory}
            className="rounded-full border border-border bg-surface-raised/60 px-3 py-1.5 text-[11px] font-medium text-text-muted transition hover:border-optical/50 hover:text-text"
          >
            History{historyCount > 0 ? ` (${historyCount})` : ""}
          </button>
        </div>
      </div>
    </header>
  );
}

export default Header;

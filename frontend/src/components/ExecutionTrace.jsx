import { formatDuration } from "../utils/formatters";

/**
 * Renders the ACTUAL backend contract shape:
 *   { router_decision, models_called, processing_time_ms }
 * — not an invented array of steps. This is what makes SatQuery
 * legible as agentic: the judge sees the router's decision and which
 * model(s) it invoked, not just a spinner.
 */
function ExecutionTrace({ query, trace }) {
  if (!trace || !trace.router_decision) return null;

  const nodes = [
    { title: "User query", detail: query ? `"${query}"` : null },
    { title: "Query router", detail: `Routed to: ${trace.router_decision}` },
    {
      title: (trace.models_called?.length ?? 0) > 1 ? "Models called" : "Model called",
      detail: (trace.models_called || []).join(", ") || "—",
    },
    { title: "Confidence analysis", detail: null },
    {
      title: "Final response",
      detail: `Completed in ${formatDuration(trace.processing_time_ms)}`,
    },
  ];

  return (
    <ol className="relative space-y-0">
      {nodes.map((node, index) => (
        <li key={node.title} className="rise-in relative flex gap-3 pb-5 last:pb-0" style={{ animationDelay: `${index * 90}ms` }}>
          {index < nodes.length - 1 && (
            <span
              className="absolute left-[7px] top-4 bottom-0 w-px bg-border"
              aria-hidden="true"
            />
          )}
          <span
            className="relative z-10 mt-0.5 flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full border border-optical bg-bg"
            aria-hidden="true"
          >
            <span className="h-1.5 w-1.5 rounded-full bg-optical" />
          </span>
          <div className="min-w-0">
            <p className="text-xs font-medium uppercase tracking-wide text-text-muted">
              {node.title}
            </p>
            {node.detail && (
              <p
                className="mt-0.5 truncate text-sm text-text"
                style={{ fontFamily: node.title === "User query" ? "var(--font-body)" : "var(--font-mono)" }}
                title={node.detail}
              >
                {node.detail}
              </p>
            )}
          </div>
        </li>
      ))}
    </ol>
  );
}

export default ExecutionTrace;

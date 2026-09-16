/**
 * Copy matches the brief's specified error states exactly. Never
 * surfaces a raw JS/axios error to the user — AnalysisError.kind
 * picks which sentence renders; the real message is still in
 * console.error for debugging.
 */
const MESSAGES = {
  network: "SatQuery AI backend is currently unavailable.",
  server: null, // uses the backend-provided detail message directly
  client: "This image format could not be processed.",
  cancelled: null,
};

function ErrorBanner({ error }) {
  if (!error) return null;
  const message = MESSAGES[error.kind] ?? error.message;
  if (!message) return null;

  return (
    <div
      className="rise-in flex items-start gap-3 rounded-lg border border-danger/30 bg-danger/5 p-4"
      role="alert"
    >
      <span className="mt-0.5 text-danger" aria-hidden="true">
        ⚠
      </span>
      <div>
        <p className="text-sm font-medium text-danger">{message}</p>
        {error.kind === "network" && (
          <p className="mt-1 text-xs text-text-faint">
            Make sure the backend is running and VITE_API_URL points to it.
          </p>
        )}
      </div>
    </div>
  );
}

export default ErrorBanner;

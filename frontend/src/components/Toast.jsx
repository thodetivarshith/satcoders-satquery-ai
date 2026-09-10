import { useEffect } from "react";

/**
 * Minimal transient confirmation (e.g. "Answer copied") — a small
 * premium touch instead of a jarring browser alert().
 */
function Toast({ message, onDismiss }) {
  useEffect(() => {
    if (!message) return;
    const id = setTimeout(onDismiss, 2000);
    return () => clearTimeout(id);
  }, [message, onDismiss]);

  if (!message) return null;

  return (
    <div
      className="rise-in panel fixed bottom-5 left-1/2 z-30 -translate-x-1/2 px-4 py-2 text-xs text-text"
      role="status"
    >
      {message}
    </div>
  );
}

export default Toast;

// Shared formatting helpers.
// Centralized so confidence/size/time formatting isn't duplicated
// (and drifting) across components.

export function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)} ${units[unitIndex]}`;
}

export function formatConfidencePercent(value) {
  const num = Number(value);
  if (Number.isNaN(num)) return 0;
  // Tolerate a confidence already expressed as 0-100 as well as 0-1.
  const normalized = num > 1 ? num / 100 : num;
  return Math.round(clamp(normalized, 0, 1) * 100);
}

export function formatDuration(ms) {
  const num = Number(ms);
  if (Number.isNaN(num) || num < 0) return "—";
  if (num < 1000) return `${Math.round(num)} ms`;
  return `${(num / 1000).toFixed(2)} s`;
}

export function formatTaskType(taskType) {
  const map = {
    vqa: "Visual Question Answering",
    captioning: "Scene Captioning",
    grounding: "Visual Grounding",
    change_detection: "Change Detection",
    classification: "Land Cover Classification",
  };
  if (!taskType) return "Unclassified";
  return map[taskType] || taskType.replace(/_/g, " ");
}

export function formatRelativeTime(isoOrDate) {
  const date = isoOrDate instanceof Date ? isoOrDate : new Date(isoOrDate);
  const diffSeconds = Math.round((Date.now() - date.getTime()) / 1000);
  if (diffSeconds < 5) return "just now";
  if (diffSeconds < 60) return `${diffSeconds}s ago`;
  const diffMinutes = Math.round(diffSeconds / 60);
  if (diffMinutes < 60) return `${diffMinutes}m ago`;
  const diffHours = Math.round(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  return date.toLocaleDateString();
}

export function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

export function truncate(text, maxLength) {
  if (!text) return "";
  return text.length > maxLength ? `${text.slice(0, maxLength - 1)}…` : text;
}

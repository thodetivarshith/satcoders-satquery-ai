// SatQuery AI — API client.
// Talks to the FastAPI backend at POST /api/analyze.
// Base URL is environment-configurable (Vite exposes VITE_-prefixed vars).

import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const ANALYZE_PATH = "/api/analyze";
const REQUEST_TIMEOUT_MS = 60_000;

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: REQUEST_TIMEOUT_MS,
});

/**
 * A normalized error shape so the UI never has to branch on
 * axios internals. `kind` drives which message the UI shows.
 */
export class AnalysisError extends Error {
  constructor(message, kind, status) {
    super(message);
    this.name = "AnalysisError";
    this.kind = kind; // "network" | "server" | "client" | "cancelled"
    this.status = status;
  }
}

/**
 * Sends the image + natural-language query to the backend.
 * @param {File} file
 * @param {string} query
 * @param {{ signal?: AbortSignal }} [options]
 * @returns {Promise<object>} parsed analysis response
 */
export async function analyzeSatelliteImage(file, query, options = {}) {
  const formData = new FormData();
  formData.append("image", file);
  formData.append("query", query);

  try {
    const response = await client.post(ANALYZE_PATH, formData, {
      signal: options.signal,
    });
    return response.data;
  } catch (err) {
    if (axios.isCancel(err) || err.code === "ERR_CANCELED") {
      throw new AnalysisError("Request cancelled.", "cancelled");
    }
    if (err.response) {
      // Backend responded, but with an error status.
      const detail =
        err.response.data?.detail ||
        err.response.data?.message ||
        `The backend returned an error (${err.response.status}).`;
      throw new AnalysisError(detail, "server", err.response.status);
    }
    if (err.request) {
      // Request went out, no response came back.
      throw new AnalysisError(
        "SatQuery AI backend is currently unavailable.",
        "network"
      );
    }
    throw new AnalysisError(
      err.message || "This image or request could not be processed.",
      "client"
    );
  }
}

export function getApiBaseUrl() {
  return API_BASE_URL;
}

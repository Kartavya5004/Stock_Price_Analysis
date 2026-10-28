// Same-origin by default (FastAPI serves this folder). Override with ?api=http://127.0.0.1:8000
// when the page is served from somewhere else, e.g. a dev server on :5173 (already allowed by CORS).
const params = new URLSearchParams(location.search);
export const API_BASE = (params.get("api") || "").replace(/\/$/, "");

// Mirrors TICKER_PATTERN in src/utils/config.py
export const TICKER_RE = /^[A-Za-z0-9][A-Za-z0-9.\-]{0,9}$/;
export const MAX_TEXT_CHARS = 1000;

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status; // 0 = network failure
  }
}

// FastAPI returns detail as a string for HTTPException and as a list for request validation errors.
function detailText(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg).filter(Boolean).join("; ");
  return "";
}

async function request(path, options) {
  let res;
  try {
    res = await fetch(API_BASE + path, options);
  } catch {
    throw new ApiError(0, "Network error");
  }
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON error body */
  }
  if (!res.ok) throw new ApiError(res.status, detailText(body?.detail));
  return body;
}

export const getHealth = () => request("/health");
export const getPrediction = (ticker) => request(`/predict/${encodeURIComponent(ticker)}`);
export const getHistory = (ticker, days = 252) => request(`/history/${encodeURIComponent(ticker)}?days=${days}`);
export const getMetrics = () => request("/metrics");
export const scoreText = (text) =>
  request("/sentiment", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });

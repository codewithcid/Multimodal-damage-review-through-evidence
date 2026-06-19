"""Central configuration: paths, model choices, pricing, and tunables.

Everything tunable lives here so the pipeline and evaluation share one source of
truth. Secrets are read from the environment only (never hardcoded).
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
# config.py -> evidence_review/ -> code/ -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = Path(os.environ.get("EVIDENCE_DATASET_DIR", REPO_ROOT / "dataset"))
# image_paths in the CSVs are relative to the dataset directory
IMAGE_BASE = DATASET_DIR

CLAIMS_CSV = DATASET_DIR / "claims.csv"
SAMPLE_CLAIMS_CSV = DATASET_DIR / "sample_claims.csv"
USER_HISTORY_CSV = DATASET_DIR / "user_history.csv"
EVIDENCE_REQUIREMENTS_CSV = DATASET_DIR / "evidence_requirements.csv"

DEFAULT_OUTPUT_CSV = Path(os.environ.get("EVIDENCE_OUTPUT_CSV", REPO_ROOT / "output.csv"))
RUNS_DIR = REPO_ROOT / "code" / "runs"  # run stats / cache artifacts

# --- Model -------------------------------------------------------------------
# Default vision model. Override per-run with --model.
DEFAULT_MODEL = os.environ.get("EVIDENCE_MODEL", "gemini-2.5-flash")

# Thinking budget per model (Gemini 2.5). 0 disables "thinking" on flash/lite to
# cut latency + cost; pro requires a small minimum.
THINKING_BUDGET = {
    "gemini-2.5-flash": 0,
    "gemini-2.5-flash-lite": 0,
    "gemini-2.5-pro": 128,
    "gemini-2.0-flash": 0,
}

# Public list pricing (USD per 1M tokens). Used ONLY for the operational cost
# estimate in evaluation_report.md. Verify against current Google AI pricing
# before quoting; treat as documented assumptions, not contractual numbers.
PRICING_USD_PER_M = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},  # <=200k context tier
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}

TEMPERATURE = float(os.environ.get("EVIDENCE_TEMPERATURE", "0"))
MAX_OUTPUT_TOKENS = int(os.environ.get("EVIDENCE_MAX_OUTPUT_TOKENS", "1200"))

# --- Images ------------------------------------------------------------------
# Resize the long edge before upload. ~1024px keeps damage detail while capping
# Gemini image-tile token cost (768px tiles, 258 tokens each).
MAX_IMAGE_LONG_EDGE = int(os.environ.get("EVIDENCE_MAX_IMAGE_LONG_EDGE", "1024"))
JPEG_QUALITY = int(os.environ.get("EVIDENCE_JPEG_QUALITY", "85"))

# --- Concurrency / retries / pacing ------------------------------------------
MAX_WORKERS = int(os.environ.get("EVIDENCE_MAX_WORKERS", "4"))
MAX_RETRIES = int(os.environ.get("EVIDENCE_MAX_RETRIES", "5"))
RETRY_BASE_DELAY = float(os.environ.get("EVIDENCE_RETRY_BASE_DELAY", "2.0"))
RETRY_MAX_DELAY = float(os.environ.get("EVIDENCE_RETRY_MAX_DELAY", "30.0"))
# Minimum seconds between request *starts* (RPM pacing). 0 = unpaced (paid tier).
# For free tier set ~4-6s to respect ~10-15 RPM.
MIN_REQUEST_INTERVAL_S = float(os.environ.get("EVIDENCE_MIN_REQUEST_INTERVAL", "0"))

# Ordered free-tier fallback chain. Each distinct model id has its own per-day
# quota bucket, so failover across these multiplies available free capacity.
FREE_FALLBACK_MODELS = [
    "gemini-2.5-flash", "gemini-flash-latest",
    "gemini-2.5-flash-lite", "gemini-flash-lite-latest",
]


def get_api_key() -> str | None:
    """Return the Gemini API key from the environment, or None if unset."""
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

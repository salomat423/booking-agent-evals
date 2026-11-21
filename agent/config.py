"""Runtime settings, read from environment variables so nothing secret lives in code."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TRACES_DIR = Path(os.getenv("BOOKING_TRACES_DIR", ROOT / "traces"))

# Model for the agent itself. Thinking is always on for this model; effort controls depth.
AGENT_MODEL = os.getenv("BOOKING_AGENT_MODEL", "claude-opus-5-5")
AGENT_EFFORT = os.getenv("BOOKING_AGENT_EFFORT", "medium")  # low | medium | high | xhigh | max

# Cheaper model that grades answers in evals (LLM-as-judge).
JUDGE_MODEL = os.getenv("BOOKING_JUDGE_MODEL", "claude-sonnet-5-5")

# If the model declines a request, let the API retry it on a fallback model inside the same call.
USE_FALLBACKS = os.getenv("BOOKING_USE_FALLBACKS", "1") == "1"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

MAX_STEPS = int(os.getenv("BOOKING_MAX_STEPS", "8"))
MAX_TOKENS = int(os.getenv("BOOKING_MAX_TOKENS", "16000"))

"""
Shared configuration for the custom_automation pipeline.

Single source of truth for paths, constants, HTTP session setup, and logging.
Import this module in every pipeline script to avoid duplicated magic strings.
"""

import logging
import os
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of the custom_automation package (directory containing this file).
PACKAGE_DIR = Path(__file__).resolve().parent

# All intermediate / generated files go here.
ARTIFACTS_DIR = PACKAGE_DIR / "artifacts"

# Graph file served to the viewer.
GRAPH_FILE = PACKAGE_DIR / "test_graphs" / "the-capital-of-the-state-containing-oakl.json"

# Artifact filenames.
PRUNED_ACTIVATIONS_FILE = ARTIFACTS_DIR / "pruned_activations.json"
FEATURE_DESCRIPTIONS_FILE = ARTIFACTS_DIR / "feature_descriptions.json"
FEATURE_GROUPS_FILE = ARTIFACTS_DIR / "feature_groups.json"
VALIDATION_REPORT_FILE = ARTIFACTS_DIR / "validation_report.json"
VALIDATION_HISTORY_FILE = ARTIFACTS_DIR / "validation_history.json"
VIEWER_URL_FILE = ARTIFACTS_DIR / "viewer_url.txt"

# ---------------------------------------------------------------------------
# HuggingFace
# ---------------------------------------------------------------------------

HF_REPO = "mwhanna/gemma-scope-transcoders"
HF_FEATURES_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main/features"

# ---------------------------------------------------------------------------
# Pipeline Defaults
# ---------------------------------------------------------------------------

# Pruning: nodes with influence <= this threshold are kept.
DEFAULT_PRUNING_THRESHOLD: float = 0.40

# For Gemma-2B this excludes the final logit projection layer(s).
MAX_LAYER_INDEX: int = 26

MAX_WORKERS: int = 10
CHECKPOINT_INTERVAL: int = 10

# Removed: # Explainer model used by add_description.py.
# EXPLAINER_MODEL_ID = "Transluce/llama_8b_explainer"

# ---------------------------------------------------------------------------
# Grouping (OpenAI) — used by group_features.py
# ---------------------------------------------------------------------------

# Model for semantic clustering.
GROUPING_MODEL: str = "gpt-5-mini"
GROUPING_TOP_K_SEED: int = 50
GROUPING_BATCH_SIZE: int = 50

# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

# Base URL for the circuit-tracer viewer.
# Override with VIEWER_URL env var for RunPod / remote setups.
VIEWER_BASE_URL: str = os.environ.get("VIEWER_URL", "https://4p8fvpyf87mb0a-8041.proxy.runpod.net/")

# ---------------------------------------------------------------------------
# HTTP Session with Retries
# ---------------------------------------------------------------------------

def make_session(
    retries: int = 3,
    backoff_factor: float = 0.5,
    status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Session:
    """Return a ``requests.Session`` with automatic retry on transient errors."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=list(status_forcelist),
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(level: int | None = None) -> logging.Logger:
    """Configure and return the pipeline-wide logger.

    Respects the ``LOG_LEVEL`` environment variable (DEBUG, INFO, WARNING …).
    """
    if level is None:
        level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("custom_automation")


# ---------------------------------------------------------------------------
# Ensure artifacts directory exists on import
# ---------------------------------------------------------------------------

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
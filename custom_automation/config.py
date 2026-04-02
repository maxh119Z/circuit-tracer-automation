"""
Shared configuration for the custom_automation pipeline.

Source for paths, constants, HTTP session setup, and logging.
Import this module in every pipeline script.
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

REPO_ROOT = PACKAGE_DIR.parent

# ---------------------------------------------------------------------------
# Slug-aware paths (for batch mode)
# Set CURRENT_SLUG env var to route artifacts + graph file per-slug.
# ---------------------------------------------------------------------------

CURRENT_SLUG: str | None = os.environ.get("CURRENT_SLUG")

# Graph file served to the viewer.
if CURRENT_SLUG:
    GRAPH_FILE = REPO_ROOT / "test_graphs" / f"{CURRENT_SLUG}.json"
    ARTIFACTS_DIR = PACKAGE_DIR / "artifacts" / CURRENT_SLUG
else:
    GRAPH_FILE = REPO_ROOT / "test_graphs" / "test-run.json"
    ARTIFACTS_DIR = PACKAGE_DIR / "artifacts"

PRUNED_ACTIVATIONS_FILE = ARTIFACTS_DIR / "pruned_activations.json"
MANUAL_GROUPS_FILE = ARTIFACTS_DIR / "manual_groups.json"
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

# ---------------------------------------------------------------------------
# Descriptions — used by generate_description.py
# ---------------------------------------------------------------------------

# Which description prompt variant to use (v0=original concise, v1=detailed,
# v2=label+explanation, v3=say-classification focused).  Override with env var.
DESCRIPTION_VARIANT: str = os.environ.get("DESCRIPTION_VARIANT", "v2")

# Artifact filenames — namespaced by variant so multiple versions coexist.
# """Old (unversioned): ARTIFACTS_DIR / "feature_descriptions.json" etc."""
FEATURE_DESCRIPTIONS_FILE = ARTIFACTS_DIR / f"feature_descriptions_{DESCRIPTION_VARIANT}.json"
FEATURE_GROUPS_FILE = ARTIFACTS_DIR / f"feature_groups_{DESCRIPTION_VARIANT}.json"
VALIDATION_REPORT_FILE = ARTIFACTS_DIR / f"validation_report_{DESCRIPTION_VARIANT}.json"
VALIDATION_HISTORY_FILE = ARTIFACTS_DIR / f"validation_history_{DESCRIPTION_VARIANT}.json"

# ---------------------------------------------------------------------------
# Grouping (OpenAI) — used by generate_supernodes.py
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
VIEWER_BASE_URL: str = os.environ.get("VIEWER_URL", "http://localhost:8041/")

# ---------------------------------------------------------------------------
# HTTP Session with Retries
# ---------------------------------------------------------------------------

def make_session(
    retries: int = 3,
    backoff_factor: float = 0.5,
    status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504),
) -> requests.Session:
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
# Logging (Terminal)
# ---------------------------------------------------------------------------


def setup_logging(level: int | None = None) -> logging.Logger:
    """Configure and return the pipeline-wide logger.
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
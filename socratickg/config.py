"""Central configuration for the SocraticKG pipeline.

All credentials are loaded from environment variables. Set them before
running, e.g. via a `.env` file (see `.env.example`) or the shell:

    export OPENAI_API_KEY="..."
    export OPENAI_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o"
"""

import os
from pathlib import Path


# -----------------------------------------------------------------------------
# LLM API configuration
# -----------------------------------------------------------------------------
API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
BASE_URL: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o")

# Generation hyperparameters
MAX_NEW_TOKENS: int = int(os.environ.get("MAX_NEW_TOKENS", "16000"))
TEMPERATURE: float = float(os.environ.get("TEMPERATURE", "0"))


# -----------------------------------------------------------------------------
# Embedding model used for canonicalization clustering
# -----------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------------------------------------------------------
# Canonicalization hyperparameters
# -----------------------------------------------------------------------------
CLUSTER_SIZE: int = int(os.environ.get("CLUSTER_SIZE", "128"))
TOP_K_CANDIDATES: int = int(os.environ.get("TOP_K_CANDIDATES", "16"))


# -----------------------------------------------------------------------------
# Output paths (overridable via CLI)
# -----------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR: Path = Path(os.environ.get("SOKG_OUTPUT_DIR", "outputs/SoKG"))


def build_output_dirs(base: Path) -> dict:
    """Create the standard subdirectory layout under `base` and return handles."""
    dirs = {
        "qa": base / "QA",
        "raw_triples": base / "Raw_Triples",
        "final_triples": base / "Aggregated_Triples",
        "usage": base / "Usage",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def validate_credentials() -> None:
    """Raise a clear error if required credentials are missing."""
    if not API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it or create a .env file "
            "(see .env.example)."
        )

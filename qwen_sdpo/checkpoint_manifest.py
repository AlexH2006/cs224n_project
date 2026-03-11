"""
TLDR: Local manifest for multi-problem batch SDPO runs — append-only JSON of checkpoint metadata.

Each problem in a batch persists its run on Modal and locally; this module records
each run in a single JSON file (no weights): problem_idx, problem_id, modal_run_dir,
local_run_dir, base_model, max_iterations, success, etc. Used by run_main_batch to
record results after each problem so the manifest is always up to date (incremental
persistence). Enables later tooling to discover checkpoints without storing model
weights locally.

Public API:
  init_manifest(manifest_path, base_model, max_iterations)  → creates empty manifest
  append_checkpoint_entry(manifest_path, entry)            → appends one run record

Used by: entrypoint.run_main_batch (called after each run_main).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


# -----------------------------------------------------------------------------
# Manifest schema
# -----------------------------------------------------------------------------

# Top-level keys in the manifest JSON file.
MANIFEST_KEY_CHECKPOINTS = "checkpoints"
MANIFEST_KEY_BASE_MODEL = "base_model"
MANIFEST_KEY_MAX_ITERATIONS = "max_iterations"
MANIFEST_KEY_CREATED = "created"

# Per-entry keys (each element of the checkpoints list).
ENTRY_PROBLEM_IDX = "problem_idx"
ENTRY_PROBLEM_ID = "problem_id"
ENTRY_BASE_MODEL = "base_model"
ENTRY_MAX_ITERATIONS = "max_iterations"
ENTRY_MODAL_RUN_DIR = "modal_run_dir"
ENTRY_LOCAL_RUN_DIR = "local_run_dir"
ENTRY_SUCCESS = "success"
ENTRY_TIMESTAMP = "timestamp"


def init_manifest(
    manifest_path: Path,
    base_model: str,
    max_iterations: int,
) -> None:
    """Create a new manifest file with no checkpoints.

    Overwrites the file if it exists. Call once at the start of a batch before
    the first append_checkpoint_entry.

    Args:
        manifest_path: Path to the JSON file (e.g. sdpo_results/checkpoint_manifest_<id>.json).
        base_model: Base HuggingFace model name (e.g. Qwen/Qwen3.5-4B).
        max_iterations: max_iterations used for this batch.
    """
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        MANIFEST_KEY_CHECKPOINTS: [],
        MANIFEST_KEY_BASE_MODEL: base_model,
        MANIFEST_KEY_MAX_ITERATIONS: max_iterations,
        MANIFEST_KEY_CREATED: datetime.now().isoformat(),
    }
    with open(manifest_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Manifest initialized: {manifest_path}")


def append_checkpoint_entry(manifest_path: Path, entry: dict[str, Any]) -> None:
    """Append one checkpoint record to the manifest file.

    Reads the existing file (or creates one with minimal defaults if missing),
    appends the entry to the checkpoints list, and writes back. Safe for
    incremental persistence: call after each problem so a crash does not lose
    prior problem records.

    Args:
        manifest_path: Path to the manifest JSON file.
        entry: Dict with at least: problem_idx, problem_id, base_model,
               max_iterations, modal_run_dir, local_run_dir, success.
               Optional: timestamp (defaults to now if omitted).
    """
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        with open(manifest_path) as f:
            data = json.load(f)
    else:
        data = {
            MANIFEST_KEY_CHECKPOINTS: [],
            MANIFEST_KEY_BASE_MODEL: entry.get(ENTRY_BASE_MODEL, ""),
            MANIFEST_KEY_MAX_ITERATIONS: entry.get(ENTRY_MAX_ITERATIONS, 0),
            MANIFEST_KEY_CREATED: datetime.now().isoformat(),
        }

    # Normalize entry: ensure required keys and optional timestamp.
    record = {
        ENTRY_PROBLEM_IDX: entry[ENTRY_PROBLEM_IDX],
        ENTRY_PROBLEM_ID: entry[ENTRY_PROBLEM_ID],
        ENTRY_BASE_MODEL: entry.get(ENTRY_BASE_MODEL, data.get(MANIFEST_KEY_BASE_MODEL, "")),
        ENTRY_MAX_ITERATIONS: entry.get(ENTRY_MAX_ITERATIONS, data.get(MANIFEST_KEY_MAX_ITERATIONS, 0)),
        ENTRY_MODAL_RUN_DIR: entry[ENTRY_MODAL_RUN_DIR],
        ENTRY_LOCAL_RUN_DIR: entry[ENTRY_LOCAL_RUN_DIR],
        ENTRY_SUCCESS: entry.get(ENTRY_SUCCESS, False),
        ENTRY_TIMESTAMP: entry.get(ENTRY_TIMESTAMP, datetime.now().isoformat()),
    }
    data[MANIFEST_KEY_CHECKPOINTS].append(record)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(data, f, indent=2)


def read_manifest(manifest_path: Path) -> dict[str, Any]:
    """Read the manifest file and return the full dict. Used by tests and tooling."""
    with open(manifest_path) as f:
        return json.load(f)

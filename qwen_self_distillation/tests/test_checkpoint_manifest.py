"""
Unit tests for qwen_sdpo/checkpoint_manifest.py.

TLDR: Ensures init_manifest creates the expected schema and append_checkpoint_entry
appends and reads back correctly. No Modal, no network.

Run with:
    python3 -m pytest qwen_sdpo/tests/test_checkpoint_manifest.py -v
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest

from qwen_sdpo.checkpoint_manifest import (
    MANIFEST_KEY_CHECKPOINTS,
    MANIFEST_KEY_BASE_MODEL,
    MANIFEST_KEY_MAX_ITERATIONS,
    MANIFEST_KEY_CREATED,
    init_manifest,
    append_checkpoint_entry,
    read_manifest,
    ENTRY_PROBLEM_IDX,
    ENTRY_PROBLEM_ID,
    ENTRY_BASE_MODEL,
    ENTRY_MAX_ITERATIONS,
    ENTRY_MODAL_RUN_DIR,
    ENTRY_LOCAL_RUN_DIR,
    ENTRY_SUCCESS,
)


def test_init_manifest_creates_schema(tmp_path: Path) -> None:
    """init_manifest creates a file with checkpoints list and metadata."""
    manifest_path = tmp_path / "manifest.json"
    init_manifest(manifest_path, base_model="Qwen/Qwen3.5-4B", max_iterations=5)
    assert manifest_path.exists()
    data = read_manifest(manifest_path)
    assert data[MANIFEST_KEY_CHECKPOINTS] == []
    assert data[MANIFEST_KEY_BASE_MODEL] == "Qwen/Qwen3.5-4B"
    assert data[MANIFEST_KEY_MAX_ITERATIONS] == 5
    assert MANIFEST_KEY_CREATED in data


def test_append_checkpoint_entry_after_init(tmp_path: Path) -> None:
    """append_checkpoint_entry after init adds one entry and preserves structure."""
    manifest_path = tmp_path / "manifest.json"
    init_manifest(manifest_path, base_model="Qwen/Qwen3.5-4B", max_iterations=5)
    entry = {
        ENTRY_PROBLEM_IDX: 7,
        ENTRY_PROBLEM_ID: "mathd_algebra_44",
        "base_model": "Qwen/Qwen3.5-4B",
        "max_iterations": 5,
        ENTRY_MODAL_RUN_DIR: "/output/Qwen3.5-4B/full_output/run_Qwen3.5-4B_7_20260310_120000",
        ENTRY_LOCAL_RUN_DIR: str(tmp_path / "sdpo_results/Qwen3.5-4B/full_output/run_Qwen3.5-4B_7_20260310_120000"),
        ENTRY_SUCCESS: True,
    }
    append_checkpoint_entry(manifest_path, entry)
    data = read_manifest(manifest_path)
    assert len(data[MANIFEST_KEY_CHECKPOINTS]) == 1
    rec = data[MANIFEST_KEY_CHECKPOINTS][0]
    assert rec[ENTRY_PROBLEM_IDX] == 7
    assert rec[ENTRY_PROBLEM_ID] == "mathd_algebra_44"
    assert rec[ENTRY_MODAL_RUN_DIR] == entry[ENTRY_MODAL_RUN_DIR]
    assert rec[ENTRY_LOCAL_RUN_DIR] == entry[ENTRY_LOCAL_RUN_DIR]
    assert rec[ENTRY_SUCCESS] is True
    assert "timestamp" in rec


def test_append_two_entries(tmp_path: Path) -> None:
    """Appending two entries yields two records in order."""
    manifest_path = tmp_path / "manifest.json"
    init_manifest(manifest_path, base_model="Qwen/Qwen3.5-4B", max_iterations=5)
    append_checkpoint_entry(manifest_path, {
        ENTRY_PROBLEM_IDX: 0,
        ENTRY_PROBLEM_ID: "prob_0",
        ENTRY_MODAL_RUN_DIR: "/output/run_0",
        ENTRY_LOCAL_RUN_DIR: str(tmp_path / "run_0"),
        ENTRY_SUCCESS: False,
    })
    append_checkpoint_entry(manifest_path, {
        ENTRY_PROBLEM_IDX: 1,
        ENTRY_PROBLEM_ID: "prob_1",
        ENTRY_MODAL_RUN_DIR: "/output/run_1",
        ENTRY_LOCAL_RUN_DIR: str(tmp_path / "run_1"),
        ENTRY_SUCCESS: True,
    })
    data = read_manifest(manifest_path)
    assert len(data[MANIFEST_KEY_CHECKPOINTS]) == 2
    assert data[MANIFEST_KEY_CHECKPOINTS][0][ENTRY_PROBLEM_IDX] == 0
    assert data[MANIFEST_KEY_CHECKPOINTS][1][ENTRY_PROBLEM_IDX] == 1


def test_append_creates_file_if_missing(tmp_path: Path) -> None:
    """append_checkpoint_entry creates manifest with defaults when file does not exist."""
    manifest_path = tmp_path / "new_manifest.json"
    assert not manifest_path.exists()
    append_checkpoint_entry(manifest_path, {
        ENTRY_PROBLEM_IDX: 3,
        ENTRY_PROBLEM_ID: "id_3",
        ENTRY_BASE_MODEL: "Qwen/Qwen3.5-4B",
        ENTRY_MAX_ITERATIONS: 5,
        ENTRY_MODAL_RUN_DIR: "/out/run_3",
        ENTRY_LOCAL_RUN_DIR: str(tmp_path / "run_3"),
        ENTRY_SUCCESS: True,
    })
    data = read_manifest(manifest_path)
    assert len(data[MANIFEST_KEY_CHECKPOINTS]) == 1
    assert data[MANIFEST_KEY_CHECKPOINTS][0][ENTRY_PROBLEM_IDX] == 3
    assert data[MANIFEST_KEY_BASE_MODEL] == "Qwen/Qwen3.5-4B"

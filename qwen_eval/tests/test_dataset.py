"""
Unit tests for qwen_eval/dataset.py.

Covers load_problem_indices_from_file() and get_field/load_problems behavior.
No HuggingFace or Modal; use temp files for index-file tests.

Run with:
    python -m pytest qwen_eval/tests/test_dataset.py -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qwen_eval.dataset import load_problem_indices_from_file


class TestLoadProblemIndicesFromFile:
    """load_problem_indices_from_file: JSON format and error cases."""

    def test_problem_indices_key(self, tmp_path: Path) -> None:
        """JSON with 'problem_indices' only returns that list in order."""
        path = tmp_path / "indices.json"
        path.write_text(json.dumps({"problem_indices": [48, 52, 53]}))
        assert load_problem_indices_from_file(str(path)) == [48, 52, 53]

    def test_problems_key(self, tmp_path: Path) -> None:
        """JSON with 'problems' only returns problem_idx from each dict."""
        path = tmp_path / "problems.json"
        path.write_text(
            json.dumps({
                "problems": [
                    {"problem_idx": 10, "problem_id": "a"},
                    {"problem_idx": 20, "problem_id": "b"},
                ]
            })
        )
        assert load_problem_indices_from_file(str(path)) == [10, 20]

    def test_missing_file_raises(self) -> None:
        """Missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_problem_indices_from_file("/nonexistent/path/indices.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        """Invalid JSON raises (e.g. json.JSONDecodeError)."""
        path = tmp_path / "bad.json"
        path.write_text("not json {")
        with pytest.raises(json.JSONDecodeError):
            load_problem_indices_from_file(str(path))

    def test_missing_keys_raises_value_error(self, tmp_path: Path) -> None:
        """JSON with neither 'problem_indices' nor 'problems' raises ValueError."""
        path = tmp_path / "empty_keys.json"
        path.write_text(json.dumps({"n_sampled": 50}))
        with pytest.raises(ValueError) as exc_info:
            load_problem_indices_from_file(str(path))
        assert "problem_indices" in str(exc_info.value)
        assert "problems" in str(exc_info.value)

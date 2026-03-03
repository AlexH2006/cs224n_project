#!/usr/bin/env python3
"""
Sort existing sdpo_results/kimina_2b run folders into dataset-named subfolders.

Reads config.dataset_name from each run's logs.json and moves the run to
sdpo_results/kimina_2b/{dataset_folder}/run_*.

Dataset folder = last segment of dataset name, e.g.:
  cat-searcher/minif2f-lean4 -> minif2f-lean4
  amitayusht/PutnamBench -> PutnamBench

Run from project root: python scripts/sort_kimina_2b_results.py
"""

import json
from pathlib import Path

KIMINA_DIR = Path("sdpo_results/kimina_2b")


def dataset_to_folder(dataset_name: str) -> str:
    """Convert dataset ID to folder name (filesystem-safe)."""
    name = dataset_name.split("/")[-1] if "/" in dataset_name else dataset_name
    return name.replace("/", "_")


def main():
    kimina = Path(KIMINA_DIR)
    if not kimina.is_dir():
        print(f"Not found: {kimina}")
        return

    # Only consider run_* dirs that are direct children (not already in a dataset subfolder)
    run_dirs = [d for d in kimina.iterdir() if d.is_dir() and d.name.startswith("run_")]
    # Exclude runs that are inside a subfolder (we only want top-level run_*)
    # Actually all direct children are either run_* or (after we create them) dataset folders.
    # So any run_* that's a direct child should be moved.
    moved = 0
    skipped = 0
    errors = []

    for run_dir in sorted(run_dirs):
        logs_path = run_dir / "logs.json"
        if not logs_path.exists():
            errors.append(f"{run_dir.name}: no logs.json")
            skipped += 1
            continue

        try:
            with open(logs_path) as f:
                logs = json.load(f)
        except Exception as e:
            errors.append(f"{run_dir.name}: failed to read logs.json - {e}")
            skipped += 1
            continue

        config = logs.get("config") or {}
        dataset_name = config.get("dataset_name") or "unknown"
        folder_name = dataset_to_folder(dataset_name)
        target_dir = kimina / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / run_dir.name

        if target_path.exists():
            errors.append(f"{run_dir.name}: target already exists {target_path}")
            skipped += 1
            continue

        run_dir.rename(target_path)
        print(f"  {run_dir.name} -> {folder_name}/")
        moved += 1

    print(f"\nMoved {moved} runs into dataset subfolders under {kimina}.")
    if skipped:
        print(f"Skipped {skipped} runs.")
    if errors:
        for e in errors:
            print(f"  - {e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Recursively traverse model training output folders, recalculate R2 from CSV tables and update metrics.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute regression evaluation metrics (consistent with evaluation.py)
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    relative_error = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8)
    mean_relative_error = np.mean(relative_error)

    return {
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MRE": float(mean_relative_error),
    }


def update_metrics_for_split(split_dir: Path) -> bool:
    """
    Recalculate metrics for a single split_xxx folder and update metrics.json

    Returns:
        bool: Whether successfully updated
    """
    metrics_path = split_dir / "metrics.json"
    tables_dir = split_dir / "tables"

    if not metrics_path.exists():
        print(f"  Skipping: metrics.json not found: {metrics_path}")
        return False

    if not tables_dir.exists():
        print(f"  Skipping: tables directory not found: {tables_dir}")
        return False

    # Read original metrics
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    updated = False
    splits = ["train", "val", "test"]

    for split in splits:
        csv_path = tables_dir / f"{split}_predictions.csv"
        metrics_key = f"{split}"

        if not csv_path.exists():
            print(f"  Warning: {csv_path} not found")
            continue

        if metrics_key not in metrics:
            print(f"  Warning: {metrics_key} missing in metrics.json")
            continue

        # Read CSV
        df = pd.read_csv(csv_path)

        if "actual" not in df.columns or "predicted" not in df.columns:
            print(f"  Warning: {csv_path} missing actual or predicted column")
            continue

        # Drop NaN values
        valid_mask = df["actual"].notna() & df["predicted"].notna()
        if not valid_mask.all():
            print(f"  Info: {len(df) - valid_mask.sum()} NaN rows ignored in {csv_path}")

        y_true = df.loc[valid_mask, "actual"].values
        y_pred = df.loc[valid_mask, "predicted"].values

        if len(y_true) < 2:
            print(f"  Warning: Not enough valid samples in {csv_path}, skipping")
            continue

        # Recalculate metrics
        new_metrics = compute_metrics(y_true, y_pred)
        old_metrics = metrics[metrics_key]

        # Compare differences
        r2_diff = abs(new_metrics["R2"] - old_metrics["R2"])
        if r2_diff > 1e-8:
            print(f"  {split}: R2 {old_metrics['R2']:.6f} -> {new_metrics['R2']:.6f} (diff: {r2_diff:.6f})")
            metrics[metrics_key] = new_metrics
            updated = True
        else:
            print(f"  {split}: R2 unchanged ({old_metrics['R2']:.6f})")

    # Save updated metrics
    if updated:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Updated: {metrics_path}")
    else:
        print(f"  - No update needed: {metrics_path}")

    return updated


def find_split_dirs(root_path: Path) -> list[Path]:
    """
    Recursively find all split_xxx folders
    """
    split_dirs = []
    for path in root_path.rglob("split_*"):
        if path.is_dir() and path.name.startswith("split_"):
            split_dirs.append(path)
    return sorted(split_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Recursively traverse model training output folders, recalculate R2 from CSV tables and update metrics.json"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Root path to scan (can be outputs/ or a single exp_xxx folder)"
    )
    args = parser.parse_args()

    root_path = args.path

    if not root_path.exists():
        print(f"Error: Path does not exist: {root_path}")
        return 1

    print(f"Scanning path: {root_path}")

    # Find all split_xxx folders
    split_dirs = find_split_dirs(root_path)

    if not split_dirs:
        print(f"No split_xxx folders found")
        return 0

    print(f"Found {len(split_dirs)} split_xxx folders\n")

    # Process each folder
    updated_count = 0
    for i, split_dir in enumerate(split_dirs, 1):
        print(f"[{i}/{len(split_dirs)}] Processing: {split_dir}")
        if update_metrics_for_split(split_dir):
            updated_count += 1
        print()

    print(f"Done! Updated {updated_count}/{len(split_dirs)} folders")
    return 0


if __name__ == "__main__":
    exit(main())

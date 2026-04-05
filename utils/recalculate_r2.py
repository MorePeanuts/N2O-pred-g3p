#!/usr/bin/env python3
"""
递归遍历模型训练输出文件夹，根据 tables 中的 CSV 重新计算 R2 并更新 metrics.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    计算回归评估指标（与 evaluation.py 保持一致）
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
    为单个 split_xxx 文件夹重新计算 metrics 并更新 metrics.json

    Returns:
        bool: 是否成功更新
    """
    metrics_path = split_dir / "metrics.json"
    tables_dir = split_dir / "tables"

    if not metrics_path.exists():
        print(f"  跳过: 找不到 metrics.json: {metrics_path}")
        return False

    if not tables_dir.exists():
        print(f"  跳过: 找不到 tables 目录: {tables_dir}")
        return False

    # 读取原始 metrics
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    updated = False
    splits = ["train", "val", "test"]

    for split in splits:
        csv_path = tables_dir / f"{split}_predictions.csv"
        metrics_key = f"{split}_metrics"

        if not csv_path.exists():
            print(f"  警告: 找不到 {csv_path}")
            continue

        if metrics_key not in metrics:
            print(f"  警告: metrics.json 中缺少 {metrics_key}")
            continue

        # 读取 CSV
        df = pd.read_csv(csv_path)

        if "actual" not in df.columns or "predicted" not in df.columns:
            print(f"  警告: {csv_path} 缺少 actual 或 predicted 列")
            continue

        # 删除 NaN 值
        valid_mask = df["actual"].notna() & df["predicted"].notna()
        if not valid_mask.all():
            print(f"  提示: {csv_path} 中有 {len(df) - valid_mask.sum()} 行 NaN 值已忽略")

        y_true = df.loc[valid_mask, "actual"].values
        y_pred = df.loc[valid_mask, "predicted"].values

        if len(y_true) < 2:
            print(f"  警告: {csv_path} 有效样本数不足，跳过")
            continue

        # 重新计算指标
        new_metrics = compute_metrics(y_true, y_pred)
        old_metrics = metrics[metrics_key]

        # 比较差异
        r2_diff = abs(new_metrics["R2"] - old_metrics["R2"])
        if r2_diff > 1e-8:
            print(f"  {split}: R2 {old_metrics['R2']:.6f} -> {new_metrics['R2']:.6f} (diff: {r2_diff:.6f})")
            metrics[metrics_key] = new_metrics
            updated = True
        else:
            print(f"  {split}: R2 无变化 ({old_metrics['R2']:.6f})")

    # 保存更新后的 metrics
    if updated:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"  ✓ 已更新: {metrics_path}")
    else:
        print(f"  - 无需更新: {metrics_path}")

    return updated


def find_split_dirs(root_path: Path) -> list[Path]:
    """
    递归查找所有 split_xxx 文件夹
    """
    split_dirs = []
    for path in root_path.rglob("split_*"):
        if path.is_dir() and path.name.startswith("split_"):
            split_dirs.append(path)
    return sorted(split_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="递归遍历模型训练输出文件夹，根据 tables 中的 CSV 重新计算 R2 并更新 metrics.json"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="要扫描的根路径（可以是 outputs/ 或单个 exp_xxx 文件夹）"
    )
    args = parser.parse_args()

    root_path = args.path

    if not root_path.exists():
        print(f"错误: 路径不存在: {root_path}")
        return 1

    print(f"扫描路径: {root_path}")

    # 查找所有 split_xxx 文件夹
    split_dirs = find_split_dirs(root_path)

    if not split_dirs:
        print(f"未找到任何 split_xxx 文件夹")
        return 0

    print(f"找到 {len(split_dirs)} 个 split_xxx 文件夹\n")

    # 处理每个文件夹
    updated_count = 0
    for i, split_dir in enumerate(split_dirs, 1):
        print(f"[{i}/{len(split_dirs)}] 处理: {split_dir}")
        if update_metrics_for_split(split_dir):
            updated_count += 1
        print()

    print(f"完成! 共更新 {updated_count}/{len(split_dirs)} 个文件夹")
    return 0


if __name__ == "__main__":
    exit(main())

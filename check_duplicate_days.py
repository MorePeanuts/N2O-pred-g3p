#!/usr/bin/env python3
"""检查数据中是否存在同一天有多个观测值的情况"""

import pickle
from pathlib import Path
import numpy as np

data_path = Path("datasets/data_EUR_processed.pkl")

print(f"加载数据: {data_path}")
with open(data_path, "rb") as f:
    sequences = pickle.load(f)

print(f"总序列数: {len(sequences)}")

total_obs = 0
total_duplicate_days = 0
duplicate_details = []

for seq_idx, seq in enumerate(sequences):
    seq_id = seq["seq_id"]
    sowdurs = np.array(seq["sowdurs"])
    seq_len = seq["seq_length"]
    total_obs += seq_len

    # 检查同一天的重复值
    unique_days, counts = np.unique(sowdurs, return_counts=True)
    duplicate_mask = counts > 1

    if np.any(duplicate_mask):
        duplicate_days = unique_days[duplicate_mask]
        duplicate_counts = counts[duplicate_mask]
        n_duplicate = int(np.sum(duplicate_counts - 1))
        total_duplicate_days += n_duplicate

        duplicate_details.append({
            "seq_idx": seq_idx,
            "seq_id": seq_id,
            "duplicate_days": list(zip(duplicate_days, duplicate_counts))
        })

        print(f"\n序列 {seq_idx} ({seq_id}):")
        print(f"  总观测数: {seq_len}")
        print(f"  重复天数: {len(duplicate_days)}")
        for day, cnt in zip(duplicate_days, duplicate_counts):
            print(f"    第 {int(day)} 天: {cnt} 个观测值")

print(f"\n\n{'='*60}")
print(f"总观测数: {total_obs}")
print(f"总重复观测数（会被RNN丢失的）: {total_duplicate_days}")
print(f"受影响的序列数: {len(duplicate_details)}")

if total_duplicate_days > 0:
    print(f"\n问题确认: 数据中存在同一天有多个观测值的情况！")
    print(f"这些重复值在 RNN-daily 模型中会由于字典键覆盖而丢失。")
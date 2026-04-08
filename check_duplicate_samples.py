#!/usr/bin/env python3
"""查看同一天的重复观测值的具体数据"""

import pickle
from pathlib import Path
import numpy as np

data_path = Path("datasets/data_EUR_processed.pkl")

with open(data_path, "rb") as f:
    sequences = pickle.load(f)

# 查看几个有重复天数的序列
sample_seqs = [152, 200, 215, 232, 289]

for seq_idx in sample_seqs:
    seq = sequences[seq_idx]
    seq_id = seq["seq_id"]
    sowdurs = np.array(seq["sowdurs"])

    # 找到重复的天数
    unique_days, indices, counts = np.unique(sowdurs, return_inverse=True, return_counts=True)
    duplicate_mask = counts > 1
    duplicate_days = unique_days[duplicate_mask]

    print(f"\n{'='*60}")
    print(f"序列 {seq_idx} {seq_id}")
    print(f"{'='*60}")

    for day in duplicate_days:
        day_indices = np.where(sowdurs == day)[0]
        print(f"\n第 {int(day)} 天有 {len(day_indices)} 个观测值:")
        for i, idx in enumerate(day_indices):
            print(f"  观测 {i+1}:")
            print(f"    No. of obs: {seq['No. of obs'][idx]}")
            print(f"    targets: {seq['targets'][idx]}")
            print(f"    numeric_dynamic: {seq['numeric_dynamic'][idx]}")
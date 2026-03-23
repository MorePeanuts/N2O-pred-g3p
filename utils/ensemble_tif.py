#!/usr/bin/env python3
"""
TIF预测结果集成脚本
计算多个预测结果的均值和方差
"""

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from tqdm import tqdm


def find_tif_files(input_path: Path) -> list[Path]:
    """
    递归查找所有TIF文件

    Args:
        input_path: 输入路径（文件或文件夹）

    Returns:
        TIF文件路径列表
    """
    if input_path.is_file():
        if input_path.suffix.lower() in ['.tif', '.tiff']:
            return [input_path]
        else:
            return []
    else:
        tif_files = []
        for ext in ['*.tif', '*.tiff']:
            tif_files.extend(input_path.rglob(ext))
        return sorted(tif_files)


def group_tif_files_by_name(tif_files: list[Path]) -> dict[str, list[Path]]:
    """
    按文件名分组TIF文件（不包含扩展名）

    Args:
        tif_files: TIF文件路径列表

    Returns:
        分组后的字典，键为文件名（不含扩展名），值为文件路径列表
    """
    groups: dict[str, list[Path]] = {}
    for tif_file in tif_files:
        name = tif_file.stem
        if name not in groups:
            groups[name] = []
        groups[name].append(tif_file)
    return groups


def load_tif_data(tif_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """
    加载TIF文件数据和元数据

    Args:
        tif_path: TIF文件路径

    Returns:
        (数据数组, 元数据字典)
    """
    with rasterio.open(tif_path) as src:
        data = src.read()  # shape: (n_bands, height, width)
        profile = src.profile.copy()
    return data, profile


def compute_ensemble_stats(data_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    计算多个预测结果的均值和方差

    Args:
        data_list: 数据数组列表，每个数组形状为 (n_bands, height, width)

    Returns:
        (均值数组, 方差数组)
    """
    # 堆叠所有数据
    stacked = np.stack(data_list, axis=0)  # shape: (n_files, n_bands, height, width)

    # 计算均值和方差（忽略NaN）
    mean_data = np.nanmean(stacked, axis=0)
    var_data = np.nanvar(stacked, axis=0)

    return mean_data, var_data


def save_tif_data(data: np.ndarray, profile: dict[str, Any], output_path: Path):
    """
    保存数据到TIF文件

    Args:
        data: 数据数组，形状为 (n_bands, height, width)
        profile: 元数据字典
        output_path: 输出路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 更新profile
    output_profile = profile.copy()
    output_profile.update(
        count=data.shape[0],
        dtype='float32',
    )

    with rasterio.open(output_path, 'w', **output_profile) as dst:
        dst.write(data.astype(np.float32))


def process_group(
    group_name: str,
    tif_files: list[Path],
    output_dir: Path,
    skip_var: bool = False,
):
    """
    处理一组TIF文件组

    Args:
        group_name: 组名（文件名不含扩展名）
        tif_files: TIF文件路径列表
        output_dir: 输出目录
        skip_var: 是否跳过方差计算
    """
    print(f"\n处理组: {group_name} ({len(tif_files)} 个文件)")

    # 加载所有数据
    data_list = []
    common_profile = None

    for tif_file in tqdm(tif_files, desc="加载文件"):
        data, profile = load_tif_data(tif_file)
        data_list.append(data)

        if common_profile is None:
            common_profile = profile
        else:
            # 验证形状一致
            if data.shape != data_list[0].shape:
                raise ValueError(
                    f"文件 {tif_file} 形状不匹配: "
                    f"{data.shape} vs {data_list[0].shape}"
                )

    # 计算统计量
    print("计算统计量...")
    mean_data, var_data = compute_ensemble_stats(data_list)

    # 保存均值
    mean_output_path = output_dir / f"{group_name}_mean.tif"
    save_tif_data(mean_data, common_profile, mean_output_path)
    print(f"均值已保存到: {mean_output_path}")

    # 保存方差
    if not skip_var:
        var_output_path = output_dir / f"{group_name}_var.tif"
        save_tif_data(var_data, common_profile, var_output_path)
        print(f"方差已保存到: {var_output_path}")


def main():
    parser = argparse.ArgumentParser(description="TIF预测结果集成脚本")
    parser.add_argument(
        "--inputs", "-i",
        nargs="+",
        required=True,
        help="输入路径（可以是多个文件或文件夹）",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--skip-var",
        action="store_true",
        help="跳过方差计算，只保存均值",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有TIF文件
    all_tif_files = []
    for input_path_str in args.inputs:
        input_path = Path(input_path_str)
        if not input_path.exists():
            print(f"警告: 路径不存在: {input_path}")
            continue
        tif_files = find_tif_files(input_path)
        all_tif_files.extend(tif_files)
        print(f"从 {input_path} 找到 {len(tif_files)} 个TIF文件")

    if not all_tif_files:
        print("错误: 未找到任何TIF文件")
        return

    print(f"\n总共找到 {len(all_tif_files)} 个TIF文件")

    # 按文件名分组
    groups = group_tif_files_by_name(all_tif_files)
    print(f"按文件名分组为 {len(groups)} 组")

    # 处理每个组
    for group_name, tif_files in groups.items():
        process_group(group_name, tif_files, output_dir, args.skip_var)

    print("\n完成!")


if __name__ == "__main__":
    main()
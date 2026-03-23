#!/usr/bin/env python3
"""
TIF预测结果集成脚本
计算多个预测结果的均值和方差
保持输入源的目录结构
"""

import argparse
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


def build_relative_path_map(input_dir: Path, tif_files: list[Path]) -> dict[Path, Path]:
    """
    构建相对路径到完整路径的映射

    Args:
        input_dir: 输入目录
        tif_files: TIF文件路径列表

    Returns:
        相对路径 -> 完整路径 的字典
    """
    path_map = {}
    for tif_file in tif_files:
        rel_path = tif_file.relative_to(input_dir)
        path_map[rel_path] = tif_file
    return path_map


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


def process_relative_path(
    rel_path: Path,
    input_files: list[Path],
    output_dir: Path,
    skip_var: bool = False,
):
    """
    处理一个相对路径对应的所有文件

    Args:
        rel_path: 相对路径
        input_files: 对应的输入文件列表
        output_dir: 输出目录
        skip_var: 是否跳过方差计算
    """
    print(f"\n处理: {rel_path} ({len(input_files)} 个文件)")

    # 加载所有数据
    data_list = []
    common_profile = None

    for tif_file in tqdm(input_files, desc="加载文件", leave=False):
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
    mean_data, var_data = compute_ensemble_stats(data_list)

    # 构建输出路径：在原文件名后加后缀
    # 例如: crop_fert_appl.tif -> crop_fert_appl_mean.tif
    stem = rel_path.stem
    suffix = rel_path.suffix

    # 保存均值
    mean_rel_path = rel_path.parent / f"{stem}_mean{suffix}"
    mean_output_path = output_dir / mean_rel_path
    save_tif_data(mean_data, common_profile, mean_output_path)
    print(f"  均值: {mean_rel_path}")

    # 保存方差
    if not skip_var:
        var_rel_path = rel_path.parent / f"{stem}_var{suffix}"
        var_output_path = output_dir / var_rel_path
        save_tif_data(var_data, common_profile, var_output_path)
        print(f"  方差: {var_rel_path}")


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

    # 处理每个输入源
    input_dirs = [Path(p) for p in args.inputs]

    # 验证所有输入都是目录（混合模式暂不支持，需要明确目录结构）
    for input_dir in input_dirs:
        if not input_dir.is_dir():
            print(f"错误: 输入必须都是目录，以保持一致的目录结构: {input_dir}")
            return

    # 为每个输入源构建相对路径映射
    input_path_maps = []
    all_rel_paths = set()

    print("扫描输入目录...")
    for input_dir in input_dirs:
        tif_files = find_tif_files(input_dir)
        path_map = build_relative_path_map(input_dir, tif_files)
        input_path_maps.append(path_map)
        all_rel_paths.update(path_map.keys())
        print(f"  {input_dir}: {len(tif_files)} 个TIF文件")

    # 只处理所有输入源都有的相对路径
    common_rel_paths = set(all_rel_paths)
    for path_map in input_path_maps:
        common_rel_paths.intersection_update(path_map.keys())

    common_rel_paths = sorted(common_rel_paths)
    print(f"\n共有的相对路径: {len(common_rel_paths)} 个")

    if not common_rel_paths:
        print("错误: 输入目录之间没有找到共有的TIF文件")
        return

    # 处理每个共有相对路径
    for rel_path in common_rel_paths:
        input_files = [path_map[rel_path] for path_map in input_path_maps]
        process_relative_path(rel_path, input_files, output_dir, args.skip_var)

    print("\n完成!")


if __name__ == "__main__":
    main()
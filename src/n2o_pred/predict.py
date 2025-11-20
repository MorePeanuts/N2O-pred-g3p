"""
预测工具模块
"""

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (
    BaseN2ODataset,
    N2ODatasetForDailyStepRNN,
    N2ODatasetForObsStepRNN,
    TifDataLoader,
    collate_fn,
)
from .evaluation import compute_metrics, save_predictions_to_csv
from .rf import N2OPredictorRF
from .rnn import N2OPredictorRNN
from .utils import create_logger, load_json

logger = create_logger(__name__)


class N2OPredictor:
    """N2O排放预测器（统一接口）"""

    def __init__(self, model_dir: Path | str):
        """
        Args:
            model_dir: 模型目录（包含模型文件和配置）
        """
        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(f"模型目录不存在: {self.model_dir}")

        # 加载配置
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        self.config = load_json(config_path)

        # 确定模型类型（从父目录的summary.json获取）
        summary_path = self.model_dir.parent / "summary.json"
        if summary_path.exists():
            summary = load_json(summary_path)
            self.model_type = summary["model_type"]
        else:
            # 尝试从配置推断
            if "rnn_type" in self.config:
                # 需要额外信息确定是obs还是daily
                logger.warning("无法从summary.json确定模型类型，请手动指定")
                self.model_type = "rnn-obs"  # 默认
            else:
                self.model_type = "rf"

        logger.info(f"加载模型类型: {self.model_type}")

        # 加载模型
        self.model = self._load_model()

        # 加载预处理器（RNN需要）
        if self.model_type.startswith("rnn"):
            scalers_path = self.model_dir / "scalers.pkl"
            if scalers_path.exists():
                with open(scalers_path, "rb") as f:
                    self.scalers = pickle.load(f)
            else:
                logger.warning("未找到scalers.pkl，预测可能失败")
                self.scalers = None

    def _load_model(self) -> Any:
        """加载模型"""
        if self.model_type == "rf":
            model_path = self.model_dir / "model.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            return N2OPredictorRF.load(model_path)

        else:  # RNN模型
            model_path = self.model_dir / "best_model.pt"
            if not model_path.exists():
                # 尝试另一个可能的路径
                model_path = self.model_dir / "model.pt"
                if not model_path.exists():
                    raise FileNotFoundError(f"模型文件不存在")

            # 加载编码器以获取cardinality
            from .preprocessing import (
                CATEGORICAL_STATIC_FEATURES,
                CATEGORICAL_DYNAMIC_FEATURES,
            )

            encoders_path = Path(__file__).parents[2] / "preprocessor" / "encoders.pkl"
            with open(encoders_path, "rb") as f:
                encoders = pickle.load(f)

            categorical_static_cardinalities = [
                len(encoders[feat].classes_) for feat in CATEGORICAL_STATIC_FEATURES
            ]
            categorical_dynamic_cardinalities = [
                len(encoders[feat].classes_) for feat in CATEGORICAL_DYNAMIC_FEATURES
            ]

            # 确定动态特征数量
            if self.model_type == "rnn-obs":
                num_dynamic_numeric = 7  # 包含time_delta
            else:
                num_dynamic_numeric = 6

            # 创建模型
            model = N2OPredictorRNN(
                num_numeric_static=6,
                num_numeric_dynamic=num_dynamic_numeric,
                categorical_static_cardinalities=categorical_static_cardinalities,
                categorical_dynamic_cardinalities=categorical_dynamic_cardinalities,
                embedding_dim=self.config.get("embedding_dim", 8),
                hidden_size=self.config.get("hidden_size", 96),
                num_layers=self.config.get("num_layers", 2),
                rnn_type=self.config.get("rnn_type", "GRU"),
                dropout=self.config.get("dropout", 0.2),
            )

            # 加载权重
            if str(model_path).endswith("best_model.pt"):
                checkpoint = torch.load(model_path, map_location="cpu")
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(torch.load(model_path, map_location="cpu"))

            model.eval()
            return model

    def predict(
        self,
        data: BaseN2ODataset | pd.DataFrame,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """
        在新数据上进行预测

        Args:
            data: 输入数据（BaseN2ODataset或DataFrame）
            device: 设备
            batch_size: 批次大小（RNN使用）

        Returns:
            预测结果字典
        """
        if self.model_type == "rf":
            return self._predict_rf(data)
        elif self.model_type == "rnn-obs":
            return self._predict_rnn_obs(data, device, batch_size)
        elif self.model_type == "rnn-daily":
            return self._predict_rnn_daily(data, device, batch_size)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def _predict_rf(self, data: pd.DataFrame | BaseN2ODataset) -> dict[str, Any]:
        """随机森林预测"""
        is_base_dataset = isinstance(data, BaseN2ODataset)

        if is_base_dataset:
            # 如果是BaseN2ODataset，展开为DataFrame（使用RF专用方法）
            data_df = data.flatten_to_dataframe_for_rf()
        else:
            data_df = data

        predictions = self.model.predict(data_df)

        from .preprocessing import LABELS

        # 检查是否有标签
        has_labels = LABELS[0] in data_df.columns

        if has_labels:
            targets = data_df[LABELS[0]].values
            metrics = compute_metrics(targets, predictions)
        else:
            targets = None
            metrics = None

        # 添加预测值到DataFrame
        data_df_with_pred = data_df.copy()
        data_df_with_pred["predicted_daily_fluxes"] = predictions

        # 如果输入是BaseN2ODataset，转换回序列格式并添加预测字段
        if is_base_dataset:
            predicted_dataset = BaseN2ODataset.from_dataframe(data_df_with_pred)
            # 为每个序列添加预测值
            for i, seq in enumerate(predicted_dataset.sequences):
                seq["predicted_targets"] = seq["targets"]  # 重命名原来的targets为predicted_targets
                # 从DataFrame中提取该序列的预测值
                seq_pred = data_df_with_pred[
                    (data_df_with_pred["Publication"] == seq["seq_id"][0])
                    & (data_df_with_pred["control_group"] == seq["seq_id"][1])
                ]["predicted_daily_fluxes"].values
                seq["predicted_targets"] = list(seq_pred)
        else:
            predicted_dataset = data_df_with_pred

        return {
            "predictions": predictions,
            "targets": targets,
            "metrics": metrics,
            "data_with_predictions": predicted_dataset,
        }

    def _predict_rnn_obs(
        self, data: BaseN2ODataset, device: str, batch_size: int
    ) -> dict[str, Any]:
        """观测步长RNN预测"""
        if not isinstance(data, BaseN2ODataset):
            raise TypeError("RNN模型需要BaseN2ODataset格式的数据")

        # 创建数据集
        dataset = N2ODatasetForObsStepRNN(data, fit_scalers=False, scalers=self.scalers)

        # 创建数据加载器
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

        # 预测
        device = torch.device(device)
        self.model = self.model.to(device)
        self.model.eval()

        all_predictions = []
        all_targets = []
        predictions_by_seq = []  # 按序列组织的预测结果

        with torch.no_grad():
            for batch in loader:
                static_numeric = batch["static_numeric"].to(device)
                dynamic_numeric = batch["dynamic_numeric"].to(device)
                static_categorical = batch["static_categorical"].to(device)
                dynamic_categorical = batch["dynamic_categorical"].to(device)
                seq_lengths = batch["seq_lengths"].to(device)
                targets_original = batch["targets_original"]

                predictions = self.model(
                    static_numeric,
                    static_categorical,
                    dynamic_numeric,
                    dynamic_categorical,
                    seq_lengths,
                )

                predictions_np = predictions.cpu().numpy()

                for i in range(len(seq_lengths)):
                    seq_len = seq_lengths[i].item()
                    pred_scaled = predictions_np[i, :seq_len]
                    target_orig = targets_original[i, :seq_len].numpy()

                    # 逆转换
                    pred_orig = dataset.inverse_transform_targets(pred_scaled)

                    all_predictions.extend(pred_orig)
                    all_targets.extend(target_orig)
                    predictions_by_seq.append(list(pred_orig))

        predictions = np.array(all_predictions)
        targets = np.array(all_targets)

        # 检查是否有有效的标签
        has_labels = not np.all(targets == 0)
        if has_labels:
            metrics = compute_metrics(targets, predictions)
        else:
            metrics = None

        # 添加预测值到原始数据集
        predicted_dataset = BaseN2ODataset(sequences=[])
        for i, seq in enumerate(data.sequences):
            new_seq = seq.copy()
            new_seq["predicted_targets"] = predictions_by_seq[i]
            predicted_dataset.sequences.append(new_seq)

        return {
            "predictions": predictions,
            "targets": targets if has_labels else None,
            "metrics": metrics,
            "data_with_predictions": predicted_dataset,
        }

    def _predict_rnn_daily(
        self, data: BaseN2ODataset, device: str, batch_size: int
    ) -> dict[str, Any]:
        """每日步长RNN预测"""
        if not isinstance(data, BaseN2ODataset):
            raise TypeError("RNN模型需要BaseN2ODataset格式的数据")

        # 创建数据集
        dataset = N2ODatasetForDailyStepRNN(
            data, fit_scalers=False, scalers=self.scalers
        )

        # 创建数据加载器
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

        # 预测
        device = torch.device(device)
        self.model = self.model.to(device)
        self.model.eval()

        all_predictions = []
        all_targets = []
        predictions_by_seq = []  # 按序列组织的预测结果（只包含真实测量点）

        with torch.no_grad():
            for batch in loader:
                static_numeric = batch["static_numeric"].to(device)
                dynamic_numeric = batch["dynamic_numeric"].to(device)
                static_categorical = batch["static_categorical"].to(device)
                dynamic_categorical = batch["dynamic_categorical"].to(device)
                seq_lengths = batch["seq_lengths"].to(device)
                targets_original = batch["targets_original"]
                mask = batch["mask"]

                predictions = self.model(
                    static_numeric,
                    static_categorical,
                    dynamic_numeric,
                    dynamic_categorical,
                    seq_lengths,
                )

                predictions_np = predictions.cpu().numpy()

                for i in range(len(seq_lengths)):
                    seq_len = seq_lengths[i].item()
                    pred_scaled = predictions_np[i, :seq_len]
                    target_orig = targets_original[i, :seq_len].numpy()
                    mask_i = mask[i, :seq_len].numpy()

                    # 逆转换
                    pred_orig = dataset.inverse_transform_targets(pred_scaled)

                    # 只保留真实测量点
                    all_predictions.extend(pred_orig[mask_i])
                    all_targets.extend(target_orig[mask_i])
                    predictions_by_seq.append(list(pred_orig[mask_i]))

        predictions = np.array(all_predictions)
        targets = np.array(all_targets)

        # 检查是否有有效的标签
        has_labels = not np.all(targets == 0)
        if has_labels:
            metrics = compute_metrics(targets, predictions)
        else:
            metrics = None

        # 添加预测值到原始数据集
        predicted_dataset = BaseN2ODataset(sequences=[])
        for i, seq in enumerate(data.sequences):
            new_seq = seq.copy()
            new_seq["predicted_targets"] = predictions_by_seq[i]
            predicted_dataset.sequences.append(new_seq)

        return {
            "predictions": predictions,
            "targets": targets if has_labels else None,
            "metrics": metrics,
            "data_with_predictions": predicted_dataset,
        }


def predict_with_model(
    model_dir: Path | str,
    data_path: Path | str,
    output_path: Path | str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    使用训练好的模型进行预测的便捷函数

    Args:
        model_dir: 模型目录
        data_path: 数据路径（.pkl或.csv文件）
        output_path: 输出路径（保存带预测结果的数据）
        device: 设备

    Returns:
        预测结果
    """
    # 加载预测器
    predictor = N2OPredictor(model_dir)

    # 加载数据
    data_path = Path(data_path)
    if data_path.suffix == ".pkl":
        # 假设是序列数据
        with open(data_path, "rb") as f:
            sequences = pickle.load(f)
        data = BaseN2ODataset(sequences)
        is_sequence_data = True
        # 将序列数据转换为DataFrame以获取定位信息
        # RF模型使用专用的flatten方法
        if predictor.model_type == "rf":
            data_df_for_location = data.flatten_to_dataframe_for_rf()
        else:
            data_df_for_location = data.flatten_to_dataframe()
    elif data_path.suffix == ".csv":
        # 假设是DataFrame
        data = pd.read_csv(data_path)
        is_sequence_data = False
        data_df_for_location = data.copy()
    else:
        raise ValueError(f"不支持的数据格式: {data_path.suffix}")

    logger.info(f"从 {data_path} 加载数据")

    # 预测
    results = predictor.predict(data, device=device)

    logger.info(f"预测完成")
    if results["metrics"]:
        logger.info(f"评估指标: {results['metrics']}")
    else:
        logger.info("未提供标签，跳过评估指标计算")

    # 保存结果
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存带预测值的数据
        if is_sequence_data:
            # 保存为pkl文件
            if output_path.suffix != ".pkl":
                output_path = output_path.with_suffix(".pkl")
            with open(output_path, "wb") as f:
                pickle.dump(results["data_with_predictions"].sequences, f)
            logger.info(f"带预测值的序列数据已保存到 {output_path}")
        else:
            # 保存为csv文件
            if output_path.suffix != ".csv":
                output_path = output_path.with_suffix(".csv")
            results["data_with_predictions"].to_csv(output_path, index=False)
            logger.info(f"带预测值的表格数据已保存到 {output_path}")

        # 额外保存预测结果的CSV（用于快速查看，包含定位信息）
        metrics_output = output_path.parent / f"{output_path.stem}_predictions.csv"

        # 构建包含定位信息的预测结果DataFrame
        pred_df = pd.DataFrame()

        # 添加定位信息
        if "No. of obs" in data_df_for_location.columns:
            pred_df["No. of obs"] = data_df_for_location["No. of obs"].values
        if "Publication" in data_df_for_location.columns:
            pred_df["Publication"] = data_df_for_location["Publication"].values
        if "control_group" in data_df_for_location.columns:
            pred_df["control_group"] = data_df_for_location["control_group"].values
        if "sowdur" in data_df_for_location.columns:
            pred_df["sowdur"] = data_df_for_location["sowdur"].values

        # 添加预测值和真实值
        pred_df["predicted"] = results["predictions"]
        if results["targets"] is not None:
            pred_df["actual"] = results["targets"]
            pred_df["error"] = results["targets"] - results["predictions"]
            pred_df["abs_error"] = np.abs(results["targets"] - results["predictions"])
        else:
            pred_df["actual"] = np.nan
            pred_df["error"] = np.nan
            pred_df["abs_error"] = np.nan

        pred_df.to_csv(metrics_output, index=False)
        logger.info(f"预测结果摘要已保存到 {metrics_output}")

    return results


def predict_tif_data(
    model_dir: Path | str,
    tif_dir: Path | str,
    output_dir: Path | str,
    device: str = "cuda:0",
    batch_size: int = 256,
) -> dict[str, Any]:
    """
    使用训练好的RNN模型对TIF格式数据进行预测

    Args:
        model_dir: 模型目录
        tif_dir: TIF数据目录（如 input_2020）
        output_dir: 输出目录
        device: 设备
        batch_size: 批次大小

    Returns:
        预测结果摘要
    """
    model_dir = Path(model_dir)
    tif_dir = Path(tif_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载预测器
    predictor = N2OPredictor(model_dir)

    # 检查模型类型
    if not predictor.model_type.startswith("rnn"):
        raise ValueError(f"TIF预测只支持RNN模型，当前模型类型: {predictor.model_type}")

    # 加载TIF数据
    logger.info(f"从 {tif_dir} 加载TIF数据...")
    tif_loader = TifDataLoader(tif_dir)

    # 获取所有有效组合
    combinations = tif_loader.get_prediction_combinations()
    logger.info(f"共 {len(combinations)} 个有效组合")

    # 准备设备
    device_obj = torch.device(device)
    predictor.model = predictor.model.to(device_obj)
    predictor.model.eval()

    # 记录结果
    results = {
        "model_dir": str(model_dir),
        "tif_dir": str(tif_dir),
        "output_dir": str(output_dir),
        "total_combinations": len(combinations),
        "completed_files": [],
        "total_pixels_processed": 0,
    }

    # 记录总开始时间
    total_start_time = time.time()

    # 遍历所有组合进行预测
    progress_bar = tqdm(combinations, desc="预测进度")
    for idx, (crop, fert, appl) in enumerate(progress_bar, 1):
        combination_name = f"{crop}_{fert}_{appl}"
        combination_start_time = time.time()

        # 更新进度条描述
        progress_bar.set_description(f"预测 [{idx}/{len(combinations)}] {combination_name}")

        # 创建数据集
        logger.info(f"[{idx}/{len(combinations)}] 正在加载 {combination_name} 数据...")
        dataset_start_time = time.time()
        dataset = tif_loader.create_rnn_dataset(
            crop, fert, appl, predictor.scalers, model_type=predictor.model_type
        )
        dataset_load_time = time.time() - dataset_start_time

        if len(dataset) == 0:
            logger.warning(f"跳过空数据集: {combination_name}")
            continue

        n_pixels = len(dataset)
        logger.info(f"  数据加载完成: {n_pixels} 像素, 耗时 {dataset_load_time:.1f}s")

        # 创建数据加载器
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        # 计算批次数量
        n_batches = (n_pixels + batch_size - 1) // batch_size
        logger.info(f"  共 {n_batches} 个批次 (batch_size={batch_size})")

        # 预测
        all_predictions = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, 1):
                if batch_idx % 10 == 0 or batch_idx == n_batches:
                    logger.info(f"  批次进度: {batch_idx}/{n_batches}")
                static_numeric = batch["static_numeric"].to(device_obj)
                dynamic_numeric = batch["dynamic_numeric"].to(device_obj)
                static_categorical = batch["static_categorical"].to(device_obj)
                dynamic_categorical = batch["dynamic_categorical"].to(device_obj)

                # TIF数据集所有样本的序列长度相同
                seq_len = dataset.n_days
                batch_size_actual = len(static_numeric)
                seq_lengths = torch.tensor(
                    [seq_len] * batch_size_actual, device=device_obj
                )

                predictions = predictor.model(
                    static_numeric,
                    static_categorical,
                    dynamic_numeric,
                    dynamic_categorical,
                    seq_lengths,
                )

                predictions_np = predictions.cpu().numpy()

                # 逆转换每个像素的预测值
                for i in range(len(predictions_np)):
                    pred_scaled = predictions_np[i, :seq_len]
                    pred_orig = dataset.inverse_transform_targets(pred_scaled)
                    all_predictions.append(pred_orig)

        # 转换为数组
        predictions_array = np.array(all_predictions)  # shape: (n_pixels, n_days)

        # 获取有效天掩码
        valid_masks = dataset.get_valid_masks()  # shape: (n_pixels, n_days)

        # 将休耕期（无效天）的预测值设为 NaN
        predictions_array[~valid_masks] = np.nan

        # 获取像素索引
        pixel_indices = dataset.get_pixel_indices()

        # 保存预测结果（包含 NaN 的休耕期）
        output_path = tif_loader.save_predictions(
            predictions_array,
            pixel_indices,
            crop,
            fert,
            appl,
            output_dir,
        )

        results["completed_files"].append(str(output_path))
        results["total_pixels_processed"] += len(pixel_indices)

        # 计算用时
        combination_time = time.time() - combination_start_time

        # 显示详细信息
        logger.info(
            f"[{idx}/{len(combinations)}] {combination_name}: "
            f"{n_pixels} 像素, 耗时 {combination_time:.1f}s, "
            f"累计 {results['total_pixels_processed']} 像素"
        )

    # 计算总用时
    total_time = time.time() - total_start_time
    avg_time_per_combination = total_time / len(results['completed_files']) if results['completed_files'] else 0

    logger.info(f"\n预测完成！")
    logger.info(f"  生成文件数: {len(results['completed_files'])}")
    logger.info(f"  总处理像素数: {results['total_pixels_processed']}")
    logger.info(f"  总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"  平均每组合: {avg_time_per_combination:.1f}s")

    return results

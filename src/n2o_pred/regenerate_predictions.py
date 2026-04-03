"""
重新生成预测表格模块
用于在训练结束后重新生成包含定位列的预测CSV文件
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split as sklearn_split
from torch.utils.data import DataLoader

from .dataset import (
    BaseN2ODataset,
    N2ODatasetForDailyStepRNN,
    N2ODatasetForObsStepRNN,
    collate_fn,
)
from .evaluation import save_predictions_to_csv
from .rf import N2OPredictorRF
from .rnn import N2OPredictorRNN
from .utils import create_logger, load_json, set_seed

logger = create_logger(__name__)


def get_location_columns_from_df(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    从DataFrame中提取定位列

    Args:
        df: 包含定位信息的DataFrame

    Returns:
        定位列字典
    """
    cols = {}
    if "No. of obs" in df.columns:
        cols["No. of obs"] = df["No. of obs"].values
    if "Publication" in df.columns:
        cols["Publication"] = df["Publication"].values
    if "control_group" in df.columns:
        cols["control_group"] = df["control_group"].values
    if "sowdur" in df.columns:
        cols["sowdur"] = df["sowdur"].values
    return cols


def get_location_columns_from_sequences(
    sequences: list[dict[str, Any]],
    masks: list[np.ndarray] | None = None,
) -> dict[str, list]:
    """
    从序列数据中提取定位列

    Args:
        sequences: 序列数据列表
        masks: 可选的掩码列表（用于daily-step RNN，只保留真实测量点）

    Returns:
        定位列字典
    """
    cols: dict[str, list] = {
        "No. of obs": [],
        "Publication": [],
        "control_group": [],
        "sowdur": [],
    }

    for i, seq in enumerate(sequences):
        seq_len = seq["seq_length"]
        pub, ctrl = seq["seq_id"]

        if masks is not None:
            # 使用掩码只保留真实测量点
            mask = masks[i]
            indices = np.where(mask)[0]
        else:
            indices = np.arange(seq_len)

        for idx in indices:
            cols["No. of obs"].append(seq["No. of obs"][idx])
            cols["Publication"].append(pub)
            cols["control_group"].append(ctrl)
            cols["sowdur"].append(seq["sowdurs"][idx])

    return cols


def regenerate_predictions_for_split(
    split_dir: Path | str,
    dataset_path: Path | str | None = None,
    device: str = "cpu",
) -> None:
    """
    为单个split重新生成预测表格

    Args:
        split_dir: split目录（如 outputs/exp_xxx/split_42）
        dataset_path: 可选的数据集路径，如果为None则使用默认路径
        device: 设备
    """
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Split目录不存在: {split_dir}")

    logger.info(f"处理split目录: {split_dir}")

    # 加载配置
    config_path = split_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    config = load_json(config_path)

    # 从父目录的summary.json获取模型类型和split seed
    summary_path = split_dir.parent / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary文件不存在: {summary_path}")
    summary = load_json(summary_path)
    model_type = summary["model_type"]

    # 从split目录名中提取seed
    split_seed = int(split_dir.name.replace("split_", ""))
    logger.info(f"模型类型: {model_type}, split_seed: {split_seed}")

    # 加载基础数据集
    if dataset_path is None:
        base_dataset = BaseN2ODataset()
    else:
        dataset_path = Path(dataset_path)
        with open(dataset_path, "rb") as f:
            sequences = pickle.load(f)
        base_dataset = BaseN2ODataset(sequences)

    # 重新划分数据集（使用相同的seed）
    set_seed(split_seed)
    n_sequences = len(base_dataset)
    indices = list(range(n_sequences))

    # 第一步：先分出测试集
    # 从summary中获取train_split，或者使用默认值0.8
    if "train_split" in summary:
        train_split = summary["train_split"]
    else:
        train_split = 0.8

    test_ratio = (1.0 - train_split) / 2
    train_val_indices, test_indices = sklearn_split(
        indices, train_size=1.0 - test_ratio, random_state=split_seed
    )

    # 第二步：从剩余的数据中分出验证集
    val_ratio = test_ratio / (1.0 - test_ratio)
    train_indices, val_indices = sklearn_split(
        train_val_indices, train_size=1.0 - val_ratio, random_state=split_seed
    )

    logger.info(f"数据集划分: {len(train_indices)} 训练, {len(val_indices)} 验证, {len(test_indices)} 测试")

    train_base = base_dataset[train_indices]
    val_base = base_dataset[val_indices]
    test_base = base_dataset[test_indices]

    # 创建输出目录
    tables_dir = split_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "rf":
        _regenerate_rf(train_base, val_base, test_base, split_dir, tables_dir)
    elif model_type == "rnn-obs":
        _regenerate_rnn_obs(train_base, val_base, test_base, split_dir, tables_dir, config, device)
    elif model_type == "rnn-daily":
        _regenerate_rnn_daily(train_base, val_base, test_base, split_dir, tables_dir, config, device)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def _regenerate_rf(
    train_base: BaseN2ODataset,
    val_base: BaseN2ODataset,
    test_base: BaseN2ODataset,
    split_dir: Path,
    tables_dir: Path,
) -> None:
    """重新生成随机森林的预测表格"""
    logger.info("重新生成随机森林预测...")

    # 加载模型
    model_path = split_dir / "model.pkl"
    model = N2OPredictorRF.load(model_path)

    # 展开数据
    train_df = train_base.flatten_to_dataframe_for_rf()
    val_df = val_base.flatten_to_dataframe_for_rf()
    test_df = test_base.flatten_to_dataframe_for_rf()

    # 预测
    train_preds = model.predict(train_df)
    val_preds = model.predict(val_df)
    test_preds = model.predict(test_df)

    from .preprocessing import LABELS

    train_targets = train_df[LABELS[0]].values
    val_targets = val_df[LABELS[0]].values
    test_targets = test_df[LABELS[0]].values

    # 保存预测结果（带定位列）
    train_cols = get_location_columns_from_df(train_df)
    val_cols = get_location_columns_from_df(val_df)
    test_cols = get_location_columns_from_df(test_df)

    save_predictions_to_csv(
        train_preds, train_targets, tables_dir / "train_predictions.csv", additional_cols=train_cols
    )
    save_predictions_to_csv(
        val_preds, val_targets, tables_dir / "val_predictions.csv", additional_cols=val_cols
    )
    save_predictions_to_csv(
        test_preds, test_targets, tables_dir / "test_predictions.csv", additional_cols=test_cols
    )

    logger.info("随机森林预测表格重新生成完成")


def _regenerate_rnn_obs(
    train_base: BaseN2ODataset,
    val_base: BaseN2ODataset,
    test_base: BaseN2ODataset,
    split_dir: Path,
    tables_dir: Path,
    config: dict,
    device: str,
) -> None:
    """重新生成RNN-Obs的预测表格"""
    logger.info("重新生成RNN-Obs预测...")

    # 加载scalers
    scalers_path = split_dir / "scalers.pkl"
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    # 创建数据集
    train_dataset = N2ODatasetForObsStepRNN(train_base, fit_scalers=False, scalers=scalers)
    val_dataset = N2ODatasetForObsStepRNN(val_base, fit_scalers=False, scalers=scalers)
    test_dataset = N2ODatasetForObsStepRNN(test_base, fit_scalers=False, scalers=scalers)

    # 加载模型
    model = _load_rnn_model(split_dir, config, "rnn-obs", device)

    # 预测
    train_preds, train_targets = _predict_rnn_obs(model, train_dataset, device)
    val_preds, val_targets = _predict_rnn_obs(model, val_dataset, device)
    test_preds, test_targets = _predict_rnn_obs(model, test_dataset, device)

    # 获取定位列
    train_cols = get_location_columns_from_sequences(train_base.sequences)
    val_cols = get_location_columns_from_sequences(val_base.sequences)
    test_cols = get_location_columns_from_sequences(test_base.sequences)

    save_predictions_to_csv(
        train_preds, train_targets, tables_dir / "train_predictions.csv", additional_cols=train_cols
    )
    save_predictions_to_csv(
        val_preds, val_targets, tables_dir / "val_predictions.csv", additional_cols=val_cols
    )
    save_predictions_to_csv(
        test_preds, test_targets, tables_dir / "test_predictions.csv", additional_cols=test_cols
    )

    logger.info("RNN-Obs预测表格重新生成完成")


def _regenerate_rnn_daily(
    train_base: BaseN2ODataset,
    val_base: BaseN2ODataset,
    test_base: BaseN2ODataset,
    split_dir: Path,
    tables_dir: Path,
    config: dict,
    device: str,
) -> None:
    """重新生成RNN-Daily的预测表格"""
    logger.info("重新生成RNN-Daily预测...")

    # 加载scalers
    scalers_path = split_dir / "scalers.pkl"
    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    # 创建数据集
    train_dataset = N2ODatasetForDailyStepRNN(train_base, fit_scalers=False, scalers=scalers)
    val_dataset = N2ODatasetForDailyStepRNN(val_base, fit_scalers=False, scalers=scalers)
    test_dataset = N2ODatasetForDailyStepRNN(test_base, fit_scalers=False, scalers=scalers)

    # 加载模型
    model = _load_rnn_model(split_dir, config, "rnn-daily", device)

    # 预测
    train_preds, train_targets, train_masks = _predict_rnn_daily(model, train_dataset, device)
    val_preds, val_targets, val_masks = _predict_rnn_daily(model, val_dataset, device)
    test_preds, test_targets, test_masks = _predict_rnn_daily(model, test_dataset, device)

    # 获取定位列（只包含真实测量点）
    train_cols = get_location_columns_from_sequences(train_base.sequences, train_masks)
    val_cols = get_location_columns_from_sequences(val_base.sequences, val_masks)
    test_cols = get_location_columns_from_sequences(test_base.sequences, test_masks)

    save_predictions_to_csv(
        train_preds, train_targets, tables_dir / "train_predictions.csv", additional_cols=train_cols
    )
    save_predictions_to_csv(
        val_preds, val_targets, tables_dir / "val_predictions.csv", additional_cols=val_cols
    )
    save_predictions_to_csv(
        test_preds, test_targets, tables_dir / "test_predictions.csv", additional_cols=test_cols
    )

    logger.info("RNN-Daily预测表格重新生成完成")


def _load_rnn_model(split_dir: Path, config: dict, model_type: str, device: str) -> N2OPredictorRNN:
    """加载RNN模型"""
    # 加载编码器以获取cardinality
    from .preprocessing import CATEGORICAL_STATIC_FEATURES, CATEGORICAL_DYNAMIC_FEATURES

    encoders_path = Path(__file__).parents[2] / "preprocessor" / "encoders.pkl"
    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)

    categorical_static_cardinalities = [len(encoders[feat].classes_) for feat in CATEGORICAL_STATIC_FEATURES]
    categorical_dynamic_cardinalities = [len(encoders[feat].classes_) for feat in CATEGORICAL_DYNAMIC_FEATURES]

    # 确定动态特征数量
    if model_type == "rnn-obs":
        num_dynamic_numeric = 7
    else:
        num_dynamic_numeric = 6

    # 创建模型
    model = N2OPredictorRNN(
        num_numeric_static=6,
        num_numeric_dynamic=num_dynamic_numeric,
        categorical_static_cardinalities=categorical_static_cardinalities,
        categorical_dynamic_cardinalities=categorical_dynamic_cardinalities,
        embedding_dim=config.get("embedding_dim", 32),
        hidden_size=config.get("hidden_size", 256),
        num_layers=config.get("num_layers", 3),
        rnn_type=config.get("rnn_type", "LSTM"),
        dropout=config.get("dropout", 0.3),
    )

    # 加载权重
    model_path = split_dir / "best_model.pt"
    if not model_path.exists():
        model_path = split_dir / "model.pt"

    if str(model_path).endswith("best_model.pt"):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    model = model.to(device)
    return model


def _predict_rnn_obs(
    model: N2OPredictorRNN, dataset: N2ODatasetForObsStepRNN, device: str
) -> tuple[np.ndarray, np.ndarray]:
    """RNN-Obs预测"""
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    device_obj = torch.device(device)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            static_numeric = batch["static_numeric"].to(device_obj)
            dynamic_numeric = batch["dynamic_numeric"].to(device_obj)
            static_categorical = batch["static_categorical"].to(device_obj)
            dynamic_categorical = batch["dynamic_categorical"].to(device_obj)
            seq_lengths = batch["seq_lengths"].to(device_obj)
            targets_original = batch["targets_original"]

            predictions = model(
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

                pred_orig = dataset.inverse_transform_targets(pred_scaled)

                all_predictions.extend(pred_orig)
                all_targets.extend(target_orig)

    return np.array(all_predictions), np.array(all_targets)


def _predict_rnn_daily(
    model: N2OPredictorRNN, dataset: N2ODatasetForDailyStepRNN, device: str
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """RNN-Daily预测，只返回真实测量点"""
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    device_obj = torch.device(device)

    all_predictions = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in loader:
            static_numeric = batch["static_numeric"].to(device_obj)
            dynamic_numeric = batch["dynamic_numeric"].to(device_obj)
            static_categorical = batch["static_categorical"].to(device_obj)
            dynamic_categorical = batch["dynamic_categorical"].to(device_obj)
            seq_lengths = batch["seq_lengths"].to(device_obj)
            targets_original = batch["targets_original"]
            masks = batch["mask"]

            predictions = model(
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
                mask_i = masks[i, :seq_len].numpy()

                pred_orig = dataset.inverse_transform_targets(pred_scaled)

                all_predictions.extend(pred_orig[mask_i])
                all_targets.extend(target_orig[mask_i])
                all_masks.append(mask_i)

    return np.array(all_predictions), np.array(all_targets), all_masks
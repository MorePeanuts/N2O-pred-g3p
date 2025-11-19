"""
评估和可视化模块
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .utils import create_logger

logger = create_logger(__name__)

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    计算回归评估指标

    Args:
        y_true: 真实值
        y_pred: 预测值

    Returns:
        指标字典
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    # 额外的统计信息
    relative_error = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8)
    mean_relative_error = np.mean(relative_error)

    return {
        "R2": float(r2),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MRE": float(mean_relative_error),
    }


def compute_sequence_metrics(
    predictions_by_seq: list[np.ndarray], targets_by_seq: list[np.ndarray]
) -> pd.DataFrame:
    """
    计算每个序列的评估指标

    Args:
        predictions_by_seq: 按序列组织的预测值列表
        targets_by_seq: 按序列组织的真实值列表

    Returns:
        包含每个序列指标的DataFrame
    """
    results = []

    for i, (preds, targets) in enumerate(zip(predictions_by_seq, targets_by_seq)):
        metrics = compute_metrics(targets, preds)
        metrics["seq_id"] = i
        metrics["seq_length"] = len(targets)
        results.append(metrics)

    return pd.DataFrame(results)


def plot_train_loss_curve(
    train_losses: list[float], val_losses: list[float], save_path: Path
) -> None:
    """
    绘制训练损失曲线

    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    plt.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Validation Loss", fontsize=14, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"训练损失曲线已保存到 {save_path}")


def plot_predictions_vs_actual(
    train_preds: np.ndarray,
    train_targets: np.ndarray,
    test_preds: np.ndarray,
    test_targets: np.ndarray,
    save_path: Path,
) -> None:
    """
    绘制预测值vs实际值散点图（训练集和测试集在同一张图上）

    Args:
        train_preds: 训练集预测值
        train_targets: 训练集真实值
        test_preds: 测试集预测值
        test_targets: 测试集真实值
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))

    # 训练集
    plt.scatter(
        train_targets,
        train_preds,
        alpha=0.5,
        s=20,
        c="blue",
        label="Train Set",
        edgecolors="none",
    )

    # 测试集
    plt.scatter(
        test_targets,
        test_preds,
        alpha=0.5,
        s=20,
        c="orange",
        label="Test Set",
        edgecolors="none",
    )

    # 计算指标
    train_r2 = r2_score(train_targets, train_preds)
    train_rmse = np.sqrt(mean_squared_error(train_targets, train_preds))
    test_r2 = r2_score(test_targets, test_preds)
    test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))

    # 添加y=x参考线
    all_vals = np.concatenate([train_targets, train_preds, test_targets, test_preds])
    min_val = all_vals.min()
    max_val = all_vals.max()
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="y=x")

    plt.xlabel("Actual N2O Flux", fontsize=12)
    plt.ylabel("Predicted N2O Flux", fontsize=12)
    plt.title(
        f"Predictions vs Actual\n"
        f"Train: R²={train_r2:.4f}, RMSE={train_rmse:.4f} | "
        f"Test: R²={test_r2:.4f}, RMSE={test_rmse:.4f}",
        fontsize=13,
        fontweight="bold",
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"预测vs实际值图已保存到 {save_path}")


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray | dict,
    save_path: Path,
    top_k: int = 20,
) -> None:
    """
    绘制特征重要性条形图

    Args:
        feature_names: 特征名称列表
        importances: 特征重要性数组或字典
        save_path: 保存路径
        top_k: 显示前k个最重要的特征
    """
    if isinstance(importances, dict):
        feature_names = list(importances.keys())
        importances = np.array(list(importances.values()))

    # 排序并选择top_k
    indices = np.argsort(importances)[::-1][:top_k]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, max(6, top_k * 0.3)))

    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    bars = plt.barh(range(len(top_features)), top_importances, color=colors)

    plt.yticks(range(len(top_features)), top_features, fontsize=10)
    plt.xlabel("Importance", fontsize=12)
    plt.title(f"Top {top_k} Feature Importances", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"特征重要性图已保存到 {save_path}")


def plot_sequence_predictions(
    seq_id: tuple,
    time_steps: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    save_path: Path,
    mask: np.ndarray | None = None,
) -> None:
    """
    绘制单个序列的时序预测图

    Args:
        seq_id: 序列ID
        time_steps: 时间步（如sowdur）
        targets: 真实值
        predictions: 预测值
        save_path: 保存路径
        mask: 掩码（用于标记真实测量点），如果提供
    """
    plt.figure(figsize=(12, 6))

    if mask is not None:
        # 区分真实测量点和插值点
        real_indices = mask == True
        interp_indices = mask == False

        plt.plot(
            time_steps[real_indices],
            targets[real_indices],
            "bo-",
            label="Actual (Measured)",
            markersize=6,
            linewidth=2,
        )
        plt.plot(
            time_steps[real_indices],
            predictions[real_indices],
            "rs-",
            label="Predicted (Measured)",
            markersize=6,
            linewidth=2,
            alpha=0.7,
        )

        if np.any(interp_indices):
            plt.plot(
                time_steps[interp_indices],
                targets[interp_indices],
                "b--",
                label="Actual (Interpolated)",
                alpha=0.3,
                linewidth=1,
            )
            plt.plot(
                time_steps[interp_indices],
                predictions[interp_indices],
                "r--",
                label="Predicted (Interpolated)",
                alpha=0.3,
                linewidth=1,
            )
    else:
        plt.plot(time_steps, targets, "bo-", label="Actual", markersize=6, linewidth=2)
        plt.plot(
            time_steps,
            predictions,
            "rs-",
            label="Predicted",
            markersize=6,
            linewidth=2,
            alpha=0.7,
        )

    # 计算该序列的指标
    if mask is not None:
        r2 = r2_score(targets[mask], predictions[mask])
    else:
        r2 = r2_score(targets, predictions)

    plt.xlabel("Time (days since sowing)", fontsize=12)
    plt.ylabel("N2O Flux", fontsize=12)
    plt.title(
        f"Sequence {seq_id} Predictions (R²={r2:.4f})", fontsize=14, fontweight="bold"
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def select_good_sequences(
    seq_metrics: pd.DataFrame, min_length: int = 15, top_n: int = 5
) -> list[int]:
    """
    选择预测效果好的长序列

    Args:
        seq_metrics: 序列指标DataFrame
        min_length: 最小序列长度
        top_n: 选择前n个

    Returns:
        序列索引列表
    """
    # 筛选长度>=min_length的序列
    long_seqs = seq_metrics[seq_metrics["seq_length"] >= min_length]

    # 按R2排序，选择top_n
    top_seqs = long_seqs.nlargest(top_n, "R2")

    return top_seqs["seq_id"].tolist()


def plot_multi_model_sequence_predictions(
    seq_id: tuple,
    time_steps: np.ndarray,
    targets: np.ndarray,
    model_predictions: dict[str, np.ndarray],
    save_path: Path,
) -> None:
    """
    绘制多个模型在同一序列上的预测对比图

    Args:
        seq_id: 序列ID
        time_steps: 时间步（如sowdur）
        targets: 真实值
        model_predictions: 模型名称到预测值的字典
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 6))

    # 绘制真实值
    plt.plot(
        time_steps,
        targets,
        "ko-",
        label="Actual",
        markersize=6,
        linewidth=2,
        alpha=0.8,
    )

    # 为每个模型使用不同的颜色和样式
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_predictions)))
    markers = ["s", "^", "D", "v", "p", "h", "*"]

    for i, (model_name, predictions) in enumerate(model_predictions.items()):
        plt.plot(
            time_steps,
            predictions,
            marker=markers[i % len(markers)],
            label=model_name,
            markersize=5,
            linewidth=1.5,
            alpha=0.7,
            color=colors[i],
        )

    plt.xlabel("Time (days since sowing)", fontsize=12)
    plt.ylabel("N2O Flux", fontsize=12)
    plt.title(
        f"Sequence {seq_id} - Model Comparison", fontsize=14, fontweight="bold"
    )
    plt.legend(fontsize=10, loc="best")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


class RNNModelWrapper:
    """
    RNN模型包装器，用于SHAP分析

    将序列数据转换为固定长度的特征向量，使得SHAP分析能够在统一的尺度上
    比较数值特征和分类特征的重要性
    """

    def __init__(self, model: Any, dataset: Any, device: str = "cpu"):
        """
        Args:
            model: RNN模型
            dataset: RNN数据集（N2ODatasetForObsStepRNN 或 N2ODatasetForDailyStepRNN）
            device: 设备
        """
        self.model = model
        self.dataset = dataset
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # 导入特征名称
        from .preprocessing import (
            CATEGORICAL_DYNAMIC_FEATURES,
            CATEGORICAL_STATIC_FEATURES,
            NUMERIC_DYNAMIC_FEATURES_RNN,
            NUMERIC_STATIC_FEATURES,
        )

        self.static_numeric_features = NUMERIC_STATIC_FEATURES
        self.static_categorical_features = CATEGORICAL_STATIC_FEATURES
        # RNN 模型只使用前 6 个动态特征（不包括 Total N amount）
        self.dynamic_numeric_features = NUMERIC_DYNAMIC_FEATURES_RNN
        self.dynamic_categorical_features = CATEGORICAL_DYNAMIC_FEATURES

        # 生成特征名称列表
        self.feature_names = self._get_feature_names()

        logger.info(f"RNN模型包装器初始化完成，设备: {device}")
        logger.info(f"特征数量: {len(self.feature_names)}")

    def _get_feature_names(self) -> list[str]:
        """生成特征名称列表"""
        names = []

        # 静态数值特征（直接使用原始名称）
        names.extend(self.static_numeric_features)

        # 静态分类特征（直接使用原始名称，因为使用了众数）
        names.extend(self.static_categorical_features)

        # 动态数值特征（使用平均值）
        for feature in self.dynamic_numeric_features:
            names.append(f"{feature}")

        # 动态分类特征（使用众数）
        for feature in self.dynamic_categorical_features:
            names.append(f"{feature}")

        return names

    def _prepare_features(self, sequence_indices: np.ndarray) -> np.ndarray:
        """
        将序列数据转换为固定长度的特征向量

        Args:
            sequence_indices: 序列索引数组

        Returns:
            特征矩阵 [n_samples, n_features]
        """
        from scipy import stats

        n_samples = len(sequence_indices)
        n_features = len(self.feature_names)
        features = np.zeros((n_samples, n_features))

        for i, seq_idx in enumerate(sequence_indices):
            # 获取原始序列数据
            seq = self.dataset.base_data.sequences[int(seq_idx)]

            feature_idx = 0

            # 静态数值特征（直接使用）
            for j in range(len(self.static_numeric_features)):
                features[i, feature_idx] = seq["numeric_static"][j]
                feature_idx += 1

            # 静态分类特征（直接使用，因为序列内是恒定的）
            for j in range(len(self.static_categorical_features)):
                features[i, feature_idx] = seq["categorical_static"][j]
                feature_idx += 1

            # 动态数值特征（计算平均值）
            for j in range(len(self.dynamic_numeric_features)):
                values = [seq["numeric_dynamic"][k][j] for k in range(seq["seq_length"])]
                features[i, feature_idx] = np.mean(values)
                feature_idx += 1

            # 动态分类特征（计算众数）
            for j in range(len(self.dynamic_categorical_features)):
                values = [seq["categorical_dynamic"][k][j] for k in range(seq["seq_length"])]
                mode_result = stats.mode(values, keepdims=True)
                features[i, feature_idx] = mode_result.mode[0]
                feature_idx += 1

        return features

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测函数，用于SHAP分析（批量优化版本）

        Args:
            X: 简化的特征数组，形状为 (n_samples, n_features)

        Returns:
            预测结果数组，形状为 (n_samples,)
        """
        # 使用批量处理优化速度
        batch_size = 32 if self.device.type == "cuda" else 16
        return self._predict_batch(X, batch_size)

    def _predict_batch(self, X: np.ndarray, batch_size: int) -> np.ndarray:
        """
        批量预测（优化速度）

        Args:
            X: 特征数组
            batch_size: 批量大小

        Returns:
            预测结果数组
        """
        from torch.utils.data import DataLoader
        from .dataset import collate_fn

        predictions = []
        n_samples = len(X)

        with torch.no_grad():
            # 按批次处理
            for batch_start in range(0, n_samples, batch_size):
                batch_end = min(batch_start + batch_size, n_samples)
                batch_size_actual = batch_end - batch_start

                # 收集批次数据
                batch_data = []
                for i in range(batch_start, batch_end):
                    seq_idx = i % len(self.dataset)
                    batch_data.append(self.dataset[seq_idx])

                # 使用 collate_fn 处理变长序列
                try:
                    batch_dict = collate_fn(batch_data)

                    # 移动到设备
                    static_numeric = batch_dict["static_numeric"].to(self.device)
                    static_categorical = batch_dict["static_categorical"].to(self.device)
                    dynamic_numeric = batch_dict["dynamic_numeric"].to(self.device)
                    dynamic_categorical = batch_dict["dynamic_categorical"].to(self.device)
                    seq_lengths = batch_dict["seq_lengths"].to(self.device)

                    # 批量预测
                    preds = self.model(
                        static_numeric,
                        static_categorical,
                        dynamic_numeric,
                        dynamic_categorical,
                        seq_lengths,
                    )

                    # 对每个序列取平均
                    for i in range(batch_size_actual):
                        seq_len = seq_lengths[i].item()
                        pred_mean = preds[i, :seq_len].mean().item()

                        # 反标准化到原始空间
                        pred_original = self.dataset.inverse_transform_targets(
                            np.array([pred_mean])
                        )[0]

                        predictions.append(pred_original)

                except Exception as e:
                    logger.warning(f"批量预测错误: {e}")
                    # 降级为逐个预测
                    for i in range(batch_start, batch_end):
                        predictions.append(0.0)

                # GPU 内存管理
                if self.device.type == "cuda" and batch_start % (batch_size * 4) == 0:
                    torch.cuda.empty_cache()

        return np.array(predictions)


def compute_shap_values(
    model: Any,
    data: Any,
    model_type: str,
    device: str = "cpu",
    max_samples: int = 1000,
    fast_mode: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    计算SHAP值

    Args:
        model: 模型（RF或RNN）
        data: 数据（DataFrame或Dataset）
        model_type: 模型类型 ('rf', 'rnn-obs', 'rnn-daily')
        device: 设备
        max_samples: 最大样本数（用于限制SHAP计算的样本数）
        fast_mode: 是否使用快速模式（RNN模型适用，默认True）

    Returns:
        (SHAP值, 特征名称列表)
    """
    try:
        import shap
    except ImportError:
        logger.error("SHAP库未安装，请运行: pip install shap")
        raise

    if model_type == "rf":
        # 随机森林使用TreeExplainer
        from .preprocessing import (
            CATEGORICAL_DYNAMIC_FEATURES,
            CATEGORICAL_STATIC_FEATURES,
            NUMERIC_DYNAMIC_FEATURES,
            NUMERIC_STATIC_FEATURES,
        )

        feature_names = (
            NUMERIC_STATIC_FEATURES
            + NUMERIC_DYNAMIC_FEATURES
            + CATEGORICAL_STATIC_FEATURES
            + CATEGORICAL_DYNAMIC_FEATURES
        )

        # 限制样本数
        if len(data) > max_samples:
            data_sample = data.sample(max_samples, random_state=42)
        else:
            data_sample = data

        X = data_sample[feature_names].values

        explainer = shap.TreeExplainer(model.model)
        shap_values = explainer.shap_values(X)

        # 计算平均绝对SHAP值作为特征重要性
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        return mean_abs_shap, feature_names

    else:
        # RNN使用KernelExplainer进行SHAP分析
        logger.info(f"RNN的SHAP分析使用KernelExplainer方法 (快速模式: {fast_mode})")

        # 创建模型包装器
        model_wrapper = RNNModelWrapper(model, data, device)

        # 根据快速模式调整样本数量
        if fast_mode:
            # 快速模式：极大减少样本数量
            n_samples = min(len(data), max_samples // 5)  # 只使用 1/5 的样本
            background_size = min(30, n_samples // 5)  # 背景数据 30 个
            n_explain = min(50, n_samples)  # 解释样本 50 个
            nsamples = 32 if device != "cpu" and torch.cuda.is_available() else 16
        else:
            # 标准模式：较少样本但更准确
            n_samples = min(len(data), max_samples // 2)
            background_size = min(50, n_samples // 4)
            n_explain = min(100, n_samples)
            nsamples = 64 if device != "cpu" and torch.cuda.is_available() else 32

        sample_indices = np.random.choice(len(data), n_samples, replace=False)

        # 准备特征矩阵
        logger.info(f"准备 {n_samples} 个样本的特征矩阵...")
        feature_matrix = model_wrapper._prepare_features(sample_indices)

        # 选择背景数据
        background_indices = np.random.choice(len(feature_matrix), background_size, replace=False)
        background_data = feature_matrix[background_indices]

        logger.info(f"背景数据样本数: {background_size}")

        # 创建SHAP解释器
        logger.info("创建SHAP解释器...")
        explainer = shap.KernelExplainer(model_wrapper.predict, background_data)

        # 选择要解释的样本
        explain_indices = np.random.choice(len(feature_matrix), n_explain, replace=False)
        explain_data = feature_matrix[explain_indices]

        logger.info(f"解释样本数: {n_explain}, SHAP采样次数: {nsamples}")

        # 计算SHAP值
        time_estimate = (background_size * n_explain * nsamples) / (32 * 100)  # 粗略估计（秒）
        logger.info(f"计算SHAP值（预计 {time_estimate:.1f} 秒）...")

        shap_values = explainer.shap_values(explain_data, nsamples=nsamples)

        logger.info(f"SHAP值计算完成，形状: {shap_values.shape}")

        # 计算平均绝对SHAP值作为特征重要性
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        return mean_abs_shap, model_wrapper.feature_names


def save_metrics_to_json(metrics: dict, save_path: Path) -> None:
    """保存指标到JSON文件"""
    import json

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info(f"指标已保存到 {save_path}")


def save_predictions_to_csv(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: Path,
    additional_cols: dict | None = None,
) -> None:
    """
    保存预测结果到CSV

    Args:
        predictions: 预测值
        targets: 真实值
        save_path: 保存路径
        additional_cols: 额外的列（如ID等）
    """
    df = pd.DataFrame(
        {
            "actual": targets,
            "predicted": predictions,
            "error": targets - predictions,
            "abs_error": np.abs(targets - predictions),
        }
    )

    if additional_cols:
        for key, value in additional_cols.items():
            df[key] = value

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)

    logger.info(f"预测结果已保存到 {save_path}")

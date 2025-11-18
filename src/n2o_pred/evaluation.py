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


def compute_shap_values(
    model: Any, data: Any, model_type: str, device: str = "cpu", max_samples: int = 1000
) -> tuple[np.ndarray, list[str]]:
    """
    计算SHAP值

    Args:
        model: 模型（RF或RNN）
        data: 数据（DataFrame或Dataset）
        model_type: 模型类型 ('rf', 'rnn-obs', 'rnn-daily')
        device: 设备
        max_samples: 最大样本数（用于限制SHAP计算的样本数）

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
        # RNN使用简化的方法：计算梯度作为重要性
        # 注：完整的DeepExplainer对于RNN较复杂，这里使用梯度近似
        logger.warning("RNN的SHAP分析使用梯度近似方法")

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

        # 收集梯度（需要在训练模式下进行反向传播）
        model.train()
        model = model.to(device)

        gradients = {name: [] for name in feature_names}

        # 采样一部分数据计算梯度
        sample_size = min(len(data), max_samples)
        indices = np.random.choice(len(data), sample_size, replace=False)

        for idx in indices[:100]:  # 进一步限制为100个样本以加快速度
            batch = data[int(idx)]

            # 准备输入
            static_numeric = (
                batch["static_numeric"].unsqueeze(0).to(device).requires_grad_(True)
            )
            dynamic_numeric = (
                batch["dynamic_numeric"].unsqueeze(0).to(device).requires_grad_(True)
            )
            static_categorical = batch["static_categorical"].unsqueeze(0).to(device)
            dynamic_categorical = batch["dynamic_categorical"].unsqueeze(0).to(device)
            seq_lengths = torch.tensor([batch["seq_length"]]).to(device)

            # 前向传播
            outputs = model(
                static_numeric,
                static_categorical,
                dynamic_numeric,
                dynamic_categorical,
                seq_lengths,
            )

            # 计算损失（预测值的和，作为标量）
            loss = outputs.sum()
            loss.backward()

            # 收集梯度
            if static_numeric.grad is not None:
                grad_static = static_numeric.grad.abs().cpu().numpy().flatten()
                for i, name in enumerate(NUMERIC_STATIC_FEATURES):
                    gradients[name].append(grad_static[i])

            if dynamic_numeric.grad is not None:
                grad_dynamic = (
                    dynamic_numeric.grad.abs().cpu().numpy().mean(axis=1).flatten()
                )
                for i, name in enumerate(NUMERIC_DYNAMIC_FEATURES):
                    gradients[name].append(grad_dynamic[i])

        # 计算平均梯度作为重要性
        importance = []
        for name in feature_names:
            if gradients[name]:
                importance.append(np.mean(gradients[name]))
            else:
                importance.append(0.0)

        return np.array(importance), feature_names


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

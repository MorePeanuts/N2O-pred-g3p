"""
实验管理器模块
"""

import json
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
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
from .evaluation import (
    compute_metrics,
    compute_sequence_metrics,
    compute_shap_values,
    plot_feature_importance,
    plot_predictions_vs_actual,
    plot_sequence_predictions,
    plot_train_loss_curve,
    save_metrics_to_json,
    save_predictions_to_csv,
    select_good_sequences,
)
from .preprocessing import preprocess_data
from .rnn import N2OPredictorRNN
from .trainer import (
    RFTrainConfig,
    RNNTrainConfig,
    train_rf_predictor,
    train_rnn_predictor,
)
from .utils import create_logger, get_device, load_json, save_json, set_seed

logger = create_logger(__name__)


def _parallel_train_worker(args_tuple):
    """
    Worker function for parallel split training

    This function is designed to be called by ProcessPoolExecutor.
    It recreates necessary objects in the subprocess and executes training.

    Args:
        args_tuple: (model_type, config_dict, seed, train_split, output_dir, device)

    Returns:
        dict with seed and metrics
    """
    model_type, config_dict, seed, train_split, output_dir, device = args_tuple

    try:
        # Create split-specific directory
        split_dir = Path(output_dir) / f"split_{seed}"
        split_dir.mkdir(parents=True, exist_ok=True)

        # Create split-specific logger to avoid conflicts
        split_logger = create_logger(f"split_{seed}", split_dir / "train.log")

        # Set random seed
        set_seed(seed)

        # Reconstruct config
        if model_type.startswith("rnn"):
            config = RNNTrainConfig(**config_dict)
            config.device = device
        else:
            config = RFTrainConfig(**config_dict)

        split_logger.info(f"开始训练 split_seed={seed} on {device}")

        # Load dataset (each worker loads its own copy)
        base_dataset = BaseN2ODataset()

        # Split data
        n_sequences = len(base_dataset)
        indices = list(range(n_sequences))

        test_ratio = (1.0 - train_split) / 2
        train_val_indices, test_indices = sklearn_split(
            indices, train_size=1.0-test_ratio, random_state=seed
        )

        val_ratio = test_ratio / (1.0 - test_ratio)
        train_indices, val_indices = sklearn_split(
            train_val_indices, train_size=1.0-val_ratio, random_state=seed
        )

        split_logger.info(
            f"数据集划分: {len(train_indices)} 训练, "
            f"{len(val_indices)} 验证, {len(test_indices)} 测试"
        )

        train_base = base_dataset[train_indices]
        val_base = base_dataset[val_indices]
        test_base = base_dataset[test_indices]

        # Execute training based on model type
        if model_type == "rf":
            result = _train_rf_worker(
                train_base, val_base, test_base, config, split_dir, split_logger
            )
        elif model_type == "rnn-obs":
            result = _train_rnn_obs_worker(
                train_base, val_base, test_base, config, split_dir, device, split_logger
            )
        elif model_type == "rnn-daily":
            result = _train_rnn_daily_worker(
                train_base, val_base, test_base, config, split_dir, device, split_logger
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")

        # Clean up GPU memory
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        split_logger.info(f"Split {seed} 完成")

        return {
            "seed": seed,
            "metrics": result["metrics"],
        }

    except Exception as e:
        split_logger.error(f"Split {seed} 失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def _train_rf_worker(train_base, val_base, test_base, config, save_dir, logger):
    """RF training worker - mirrors ExperimentRunner._train_rf logic"""
    # Flatten to DataFrame
    train_df = train_base.flatten_to_dataframe_for_rf()
    val_df = val_base.flatten_to_dataframe_for_rf()
    test_df = test_base.flatten_to_dataframe_for_rf()

    logger.info(f"训练数据: {len(train_df)} 个样本")
    logger.info(f"验证数据: {len(val_df)} 个样本")
    logger.info(f"测试数据: {len(test_df)} 个样本")

    # Train model
    model, train_result = train_rf_predictor(
        train_df=train_df, val_df=val_df, test_df=test_df,
        config=config, save_dir=save_dir
    )

    # Save config
    save_json(config.to_dict(), save_dir / "config.json")

    # Generate outputs (full version with plots)
    figs_dir = save_dir / "figs"
    tables_dir = save_dir / "tables"
    figs_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    # 1. 预测vs实际值图（使用测试集）
    plot_predictions_vs_actual(
        train_result["train_predictions"],
        train_result["train_targets"],
        train_result["test_predictions"],
        train_result["test_targets"],
        figs_dir / "predictions_vs_actual.png",
    )

    # 2. 特征重要性图
    plot_feature_importance(
        list(train_result["feature_importances"].keys()),
        train_result["feature_importances"],
        figs_dir / "feature_importance.png",
    )

    # 3. 保存预测结果到CSV
    save_predictions_to_csv(
        train_result["train_predictions"],
        train_result["train_targets"],
        tables_dir / "train_predictions.csv",
    )
    save_predictions_to_csv(
        train_result["val_predictions"],
        train_result["val_targets"],
        tables_dir / "val_predictions.csv",
    )
    save_predictions_to_csv(
        train_result["test_predictions"],
        train_result["test_targets"],
        tables_dir / "test_predictions.csv",
    )

    # 4. 保存特征重要性到CSV
    importance_df = pd.DataFrame(
        {
            "feature": list(train_result["feature_importances"].keys()),
            "importance": list(train_result["feature_importances"].values()),
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(tables_dir / "feature_importance.csv", index=False)

    # 5. 序列预测图（使用测试集将DataFrame重构为序列）
    logger.info("生成序列预测图...")
    test_df_with_pred = test_df.copy()
    test_df_with_pred["predicted_daily_fluxes"] = train_result["test_predictions"]

    test_base_reconstructed = BaseN2ODataset.from_dataframe(test_df_with_pred)

    test_preds_by_seq = []
    test_targets_by_seq = []
    test_seq_ids = []

    for seq in test_base_reconstructed.sequences:
        test_preds_by_seq.append(
            test_df_with_pred[
                (test_df_with_pred["Publication"] == seq["seq_id"][0])
                & (test_df_with_pred["control_group"] == seq["seq_id"][1])
            ]["predicted_daily_fluxes"].values
        )
        test_targets_by_seq.append(np.array(seq["targets"]))
        test_seq_ids.append(tuple(seq["seq_id"]))

    seq_metrics = compute_sequence_metrics(test_preds_by_seq, test_targets_by_seq)
    seq_metrics["seq_id_tuple"] = test_seq_ids

    good_seq_indices = select_good_sequences(seq_metrics, min_length=15, top_n=5)

    logger.info(f"绘制 {len(good_seq_indices)} 个序列的预测图...")
    for idx in good_seq_indices:
        seq = test_base_reconstructed.sequences[idx]
        seq_id = tuple(seq["seq_id"])
        time_steps = np.array(seq["sowdurs"])
        targets = np.array(seq["targets"])
        predictions = test_preds_by_seq[idx]

        plot_sequence_predictions(
            seq_id,
            time_steps,
            targets,
            predictions,
            figs_dir / f"sequence_predictions_{seq_id[0]}_{seq_id[1]}.png",
            mask=None,
        )

    logger.info(f"输出已保存到 {save_dir}")

    metrics = {
        "train": train_result["train_metrics"],
        "val": train_result["val_metrics"],
        "test": train_result["test_metrics"],
        "n_parameters": train_result["n_parameters"],
    }
    save_metrics_to_json(metrics, save_dir / "metrics.json")

    return {"metrics": metrics}


def _train_rnn_obs_worker(train_base, val_base, test_base, config, save_dir, device, logger):
    """RNN-Obs training worker"""
    # Create RNN datasets
    train_dataset = N2ODatasetForObsStepRNN(train_base, fit_scalers=True)
    val_dataset = N2ODatasetForObsStepRNN(val_base, fit_scalers=False, scalers=train_dataset.scalers)
    test_dataset = N2ODatasetForObsStepRNN(test_base, fit_scalers=False, scalers=train_dataset.scalers)

    logger.info(f"训练数据: {len(train_dataset)} 个序列")
    logger.info(f"验证数据: {len(val_dataset)} 个序列")
    logger.info(f"测试数据: {len(test_dataset)} 个序列")

    # Save scalers
    with open(save_dir / "scalers.pkl", "wb") as f:
        pickle.dump(train_dataset.scalers, f)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    # Get categorical cardinalities
    from .preprocessing import CATEGORICAL_STATIC_FEATURES, CATEGORICAL_DYNAMIC_FEATURES

    encoders_path = Path(__file__).parents[2] / "preprocessor" / "encoders.pkl"
    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)

    categorical_static_cardinalities = [len(encoders[feat].classes_) for feat in CATEGORICAL_STATIC_FEATURES]
    categorical_dynamic_cardinalities = [len(encoders[feat].classes_) for feat in CATEGORICAL_DYNAMIC_FEATURES]

    # Create model
    model = N2OPredictorRNN(
        num_numeric_static=6,
        num_numeric_dynamic=7,  # 原始6个 + time_delta
        categorical_static_cardinalities=categorical_static_cardinalities,
        categorical_dynamic_cardinalities=categorical_dynamic_cardinalities,
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        rnn_type=config.rnn_type,
        dropout=config.dropout,
    )

    # Train model
    train_result = train_rnn_predictor(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        save_dir=save_dir,
        use_mask=False,
    )

    # Save config and model
    save_json(config.to_dict(), save_dir / "config.json")
    torch.save(model.state_dict(), save_dir / "model.pt")

    # Generate outputs (full version with plots)
    figs_dir = save_dir / "figs"
    tables_dir = save_dir / "tables"
    figs_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    # 1. 训练损失曲线
    plot_train_loss_curve(
        train_result["train_losses"],
        train_result["val_losses"],
        figs_dir / "train_loss_curve.png",
    )

    # 2. 预测vs实际值图（使用测试集）
    plot_predictions_vs_actual(
        train_result["train_predictions"],
        train_result["train_targets"],
        train_result["test_predictions"],
        train_result["test_targets"],
        figs_dir / "predictions_vs_actual.png",
    )

    # 3. 保存预测结果到CSV
    save_predictions_to_csv(
        train_result["train_predictions"],
        train_result["train_targets"],
        tables_dir / "train_predictions.csv",
    )
    save_predictions_to_csv(
        train_result["val_predictions"],
        train_result["val_targets"],
        tables_dir / "val_predictions.csv",
    )
    save_predictions_to_csv(
        train_result["test_predictions"],
        train_result["test_targets"],
        tables_dir / "test_predictions.csv",
    )

    # 4. SHAP分析
    try:
        logger.info("计算特征重要性（使用梯度近似）...")
        shap_values, feature_names = compute_shap_values(
            model, test_dataset, "rnn-obs", device, max_samples=100
        )
        plot_feature_importance(
            feature_names, shap_values, figs_dir / "feature_importance.png"
        )
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": shap_values}
        ).sort_values("importance", ascending=False)
        importance_df.to_csv(tables_dir / "feature_importance.csv", index=False)
    except Exception as e:
        logger.warning(f"SHAP分析失败: {e}")

    # 5. 序列预测图
    logger.info("生成序列预测图...")
    test_preds_by_seq = []
    test_targets_by_seq = []
    test_seq_ids = []

    for i in range(len(test_dataset)):
        seq = test_dataset[i]
        test_seq_ids.append(tuple(test_dataset.processed_sequences[i]["seq_id"]))

        model.eval()
        with torch.no_grad():
            static_numeric = seq["static_numeric"].unsqueeze(0).to(device)
            dynamic_numeric = seq["dynamic_numeric"].unsqueeze(0).to(device)
            static_categorical = seq["static_categorical"].unsqueeze(0).to(device)
            dynamic_categorical = seq["dynamic_categorical"].unsqueeze(0).to(device)
            seq_lengths = torch.tensor([seq["seq_length"]]).to(device)

            pred_scaled = model(
                static_numeric, static_categorical,
                dynamic_numeric, dynamic_categorical, seq_lengths,
            )
            pred_scaled = pred_scaled.cpu().numpy()[0, : seq["seq_length"]]
            pred_orig = test_dataset.inverse_transform_targets(pred_scaled)

        target_orig = seq["targets_original"].numpy()
        test_preds_by_seq.append(pred_orig)
        test_targets_by_seq.append(target_orig)

    seq_metrics = compute_sequence_metrics(test_preds_by_seq, test_targets_by_seq)
    seq_metrics["seq_id_tuple"] = test_seq_ids

    good_seq_indices = select_good_sequences(seq_metrics, min_length=15, top_n=5)

    logger.info(f"绘制 {len(good_seq_indices)} 个序列的预测图...")
    for idx in good_seq_indices:
        seq = test_dataset.processed_sequences[idx]
        seq_id = tuple(seq["seq_id"])

        model.eval()
        with torch.no_grad():
            seq_data = test_dataset[idx]
            static_numeric = seq_data["static_numeric"].unsqueeze(0).to(device)
            dynamic_numeric = seq_data["dynamic_numeric"].unsqueeze(0).to(device)
            static_categorical = seq_data["static_categorical"].unsqueeze(0).to(device)
            dynamic_categorical = seq_data["dynamic_categorical"].unsqueeze(0).to(device)
            seq_lengths = torch.tensor([seq_data["seq_length"]]).to(device)

            pred_scaled = model(
                static_numeric, static_categorical,
                dynamic_numeric, dynamic_categorical, seq_lengths,
            )
            pred_scaled = pred_scaled.cpu().numpy()[0, : seq["seq_length"]]
            pred_orig = test_dataset.inverse_transform_targets(pred_scaled)

        target_orig = seq_data["targets_original"].numpy()
        time_steps = np.array(test_dataset.sequences[idx]["sowdurs"])

        plot_sequence_predictions(
            seq_id, time_steps, target_orig, pred_orig,
            figs_dir / f"sequence_predictions_{seq_id[0]}_{seq_id[1]}.png",
            mask=None,
        )

    logger.info(f"输出已保存到 {save_dir}")

    # Save metrics
    metrics = {
        "train": train_result["train_metrics"],
        "val": train_result["val_metrics"],
        "test": train_result["test_metrics"],
        "n_parameters": train_result["n_parameters"],
    }
    save_metrics_to_json(metrics, save_dir / "metrics.json")

    return {"metrics": metrics}


def _train_rnn_daily_worker(train_base, val_base, test_base, config, save_dir, device, logger):
    """RNN-Daily training worker"""
    # Create RNN datasets
    train_dataset = N2ODatasetForDailyStepRNN(train_base, fit_scalers=True)
    val_dataset = N2ODatasetForDailyStepRNN(val_base, fit_scalers=False, scalers=train_dataset.scalers)
    test_dataset = N2ODatasetForDailyStepRNN(test_base, fit_scalers=False, scalers=train_dataset.scalers)

    logger.info(f"训练数据: {len(train_dataset)} 个序列")
    logger.info(f"验证数据: {len(val_dataset)} 个序列")
    logger.info(f"测试数据: {len(test_dataset)} 个序列")

    # Save scalers
    with open(save_dir / "scalers.pkl", "wb") as f:
        pickle.dump(train_dataset.scalers, f)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    # Get categorical cardinalities
    from .preprocessing import CATEGORICAL_STATIC_FEATURES, CATEGORICAL_DYNAMIC_FEATURES

    encoders_path = Path(__file__).parents[2] / "preprocessor" / "encoders.pkl"
    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)

    categorical_static_cardinalities = [len(encoders[feat].classes_) for feat in CATEGORICAL_STATIC_FEATURES]
    categorical_dynamic_cardinalities = [len(encoders[feat].classes_) for feat in CATEGORICAL_DYNAMIC_FEATURES]

    # Create model
    model = N2OPredictorRNN(
        num_numeric_static=6,
        num_numeric_dynamic=6,
        categorical_static_cardinalities=categorical_static_cardinalities,
        categorical_dynamic_cardinalities=categorical_dynamic_cardinalities,
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        rnn_type=config.rnn_type,
        dropout=config.dropout,
    )

    # Train model
    train_result = train_rnn_predictor(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        save_dir=save_dir,
        use_mask=True,
    )

    # Save config and model
    save_json(config.to_dict(), save_dir / "config.json")
    torch.save(model.state_dict(), save_dir / "model.pt")

    # Generate outputs (full version with plots)
    figs_dir = save_dir / "figs"
    tables_dir = save_dir / "tables"
    figs_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    # 1. 训练损失曲线
    plot_train_loss_curve(
        train_result["train_losses"],
        train_result["val_losses"],
        figs_dir / "train_loss_curve.png",
    )

    # 2. 预测vs实际值图（使用测试集）
    plot_predictions_vs_actual(
        train_result["train_predictions"],
        train_result["train_targets"],
        train_result["test_predictions"],
        train_result["test_targets"],
        figs_dir / "predictions_vs_actual.png",
    )

    # 3. 保存预测结果到CSV
    save_predictions_to_csv(
        train_result["train_predictions"],
        train_result["train_targets"],
        tables_dir / "train_predictions.csv",
    )
    save_predictions_to_csv(
        train_result["val_predictions"],
        train_result["val_targets"],
        tables_dir / "val_predictions.csv",
    )
    save_predictions_to_csv(
        train_result["test_predictions"],
        train_result["test_targets"],
        tables_dir / "test_predictions.csv",
    )

    # 4. SHAP分析
    try:
        logger.info("计算特征重要性（使用梯度近似）...")
        shap_values, feature_names = compute_shap_values(
            model, test_dataset, "rnn-daily", device, max_samples=100
        )
        plot_feature_importance(
            feature_names, shap_values, figs_dir / "feature_importance.png"
        )
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": shap_values}
        ).sort_values("importance", ascending=False)
        importance_df.to_csv(tables_dir / "feature_importance.csv", index=False)
    except Exception as e:
        logger.warning(f"SHAP分析失败: {e}")

    # 5. 序列预测图（使用测试集）
    logger.info("生成序列预测图...")
    test_preds_by_seq = []
    test_targets_by_seq = []
    test_seq_ids = []

    for i in range(len(test_dataset)):
        seq = test_dataset[i]
        test_seq_ids.append(tuple(test_dataset.processed_sequences[i]["seq_id"]))

        model.eval()
        with torch.no_grad():
            static_numeric = seq["static_numeric"].unsqueeze(0).to(device)
            dynamic_numeric = seq["dynamic_numeric"].unsqueeze(0).to(device)
            static_categorical = seq["static_categorical"].unsqueeze(0).to(device)
            dynamic_categorical = seq["dynamic_categorical"].unsqueeze(0).to(device)
            seq_lengths = torch.tensor([seq["seq_length"]]).to(device)

            pred_scaled = model(
                static_numeric, static_categorical,
                dynamic_numeric, dynamic_categorical, seq_lengths,
            )
            pred_scaled = pred_scaled.cpu().numpy()[0, : seq["seq_length"]]
            pred_orig = test_dataset.inverse_transform_targets(pred_scaled)

        target_orig = seq["targets_original"].numpy()

        # For DailyStepRNN, only keep real measurement points
        mask = seq["mask"].numpy()
        test_preds_by_seq.append(pred_orig[mask])
        test_targets_by_seq.append(target_orig[mask])

    seq_metrics = compute_sequence_metrics(test_preds_by_seq, test_targets_by_seq)
    seq_metrics["seq_id_tuple"] = test_seq_ids

    good_seq_indices = select_good_sequences(seq_metrics, min_length=15, top_n=5)

    logger.info(f"绘制 {len(good_seq_indices)} 个序列的预测图...")
    for idx in good_seq_indices:
        seq = test_dataset.processed_sequences[idx]
        seq_id = tuple(seq["seq_id"])

        model.eval()
        with torch.no_grad():
            seq_data = test_dataset[idx]
            static_numeric = seq_data["static_numeric"].unsqueeze(0).to(device)
            dynamic_numeric = seq_data["dynamic_numeric"].unsqueeze(0).to(device)
            static_categorical = seq_data["static_categorical"].unsqueeze(0).to(device)
            dynamic_categorical = seq_data["dynamic_categorical"].unsqueeze(0).to(device)
            seq_lengths = torch.tensor([seq_data["seq_length"]]).to(device)

            pred_scaled = model(
                static_numeric, static_categorical,
                dynamic_numeric, dynamic_categorical, seq_lengths,
            )
            pred_scaled = pred_scaled.cpu().numpy()[0, : seq["seq_length"]]
            pred_orig = test_dataset.inverse_transform_targets(pred_scaled)

        target_orig = seq_data["targets_original"].numpy()

        # For DailyStepRNN, only plot real measurement points
        mask_seq = seq_data["mask"].numpy()
        time_steps = np.array(test_dataset.sequences[idx]["sowdurs"])
        pred_orig = pred_orig[mask_seq]
        target_orig = target_orig[mask_seq]

        plot_sequence_predictions(
            seq_id, time_steps, target_orig, pred_orig,
            figs_dir / f"sequence_predictions_{seq_id[0]}_{seq_id[1]}.png",
            mask=None,
        )

    logger.info(f"输出已保存到 {save_dir}")

    # Save metrics
    metrics = {
        "train": train_result["train_metrics"],
        "val": train_result["val_metrics"],
        "test": train_result["test_metrics"],
        "n_parameters": train_result["n_parameters"],
    }
    save_metrics_to_json(metrics, save_dir / "metrics.json")

    return {"metrics": metrics}


class ExperimentRunner:
    """实验管理器"""

    def __init__(
        self,
        model_type: str,
        output_dir: Path | str | None = None,
        device: str = "cuda:0",
        rnn_config: RNNTrainConfig | None = None,
        rf_config: RFTrainConfig | None = None,
    ):
        """
        Args:
            model_type: 模型类型 ('rf', 'rnn-obs', 'rnn-daily')
            output_dir: 输出目录
            device: 设备
            rnn_config: RNN训练配置
            rf_config: RF训练配置
        """
        self.model_type = model_type
        self.device = device

        # 设置输出目录
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(__file__).parents[2] / "outputs" / f"exp_{model_type}_{timestamp}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志
        global logger
        logger = create_logger(__name__, self.output_dir / "experiment.log")

        # 配置
        if model_type.startswith("rnn"):
            self.config = rnn_config or RNNTrainConfig()
            self.config.device = device
        else:
            self.config = rf_config or RFTrainConfig()

        logger.info(f"实验管理器初始化完成")
        logger.info(f"模型类型: {model_type}")
        logger.info(f"输出目录: {self.output_dir}")

    def run(
        self,
        total_splits: int = 1,
        split_seeds: list[int] | None = None,
        split_seed: int = 42,
        train_split: float = 0.9,
        max_workers: int = 1,
    ) -> dict[str, Any]:
        """
        运行实验

        Args:
            total_splits: 总划分数
            split_seeds: 指定的种子列表（如果提供，则忽略total_splits和split_seed）
            split_seed: 生成随机种子的种子
            train_split: 训练集比例
            max_workers: 最大并行worker数（默认1，串行执行）

        Returns:
            实验总结
        """
        # 确定种子列表
        if split_seeds is not None:
            seeds = split_seeds
        else:
            # 生成随机种子
            np.random.seed(split_seed)
            seeds = np.random.randint(0, 10000, total_splits).tolist()

        logger.info(f"将运行 {len(seeds)} 个划分")
        logger.info(f"种子列表: {seeds}")
        if max_workers > 1:
            logger.info(f"使用 {max_workers} 个并行worker")

        # 存储每个split的结果
        split_results = []
        best_split = None
        best_metric = -float("inf")  # 使用测试集R2作为最佳指标

        if max_workers == 1:
            # Serial execution (original behavior)
            logger.info("使用串行模式执行")

            # 加载基础数据集
            base_dataset = BaseN2ODataset()
            logger.info(f"加载数据集，共 {len(base_dataset)} 个序列")

            for seed in seeds:
                logger.info(f"\n{'=' * 80}")
                logger.info(f"开始运行 split_seed={seed}")
                logger.info(f"{'=' * 80}")

                split_dir = self.output_dir / f"split_{seed}"
                split_dir.mkdir(parents=True, exist_ok=True)

                try:
                    result = self._run_single_split(
                        base_dataset=base_dataset,
                        seed=seed,
                        train_split=train_split,
                        save_dir=split_dir,
                    )

                    split_results.append(
                        {
                            "seed": seed,
                            "metrics": result["metrics"],
                        }
                    )

                    # 更新最佳split（使用测试集R2）
                    test_r2 = result["metrics"]["test"]["R2"]
                    if test_r2 > best_metric:
                        best_metric = test_r2
                        best_split = seed

                    logger.info(f"Split {seed} 完成")

                except Exception as e:
                    logger.error(f"Split {seed} 失败: {e}")
                    import traceback
                    traceback.print_exc()

        else:
            # Parallel execution
            logger.info("使用并行模式执行")

            # Detect available GPUs
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                logger.warning("未检测到GPU，将使用CPU训练")
                devices = ["cpu"] * max_workers
            else:
                logger.info(f"检测到 {num_gpus} 个GPU")
                # Round-robin GPU assignment
                devices = [f"cuda:{i % num_gpus}" for i in range(max_workers)]

            # Prepare tasks
            config_dict = self.config.to_dict()
            tasks = [
                (self.model_type, config_dict, seed, train_split, str(self.output_dir), devices[i % max_workers])
                for i, seed in enumerate(seeds)
            ]

            # Execute in parallel
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_seed = {
                    executor.submit(_parallel_train_worker, task): task[2]
                    for task in tasks
                }

                # Collect results as they complete
                for future in as_completed(future_to_seed):
                    seed = future_to_seed[future]
                    try:
                        result = future.result()
                        split_results.append(result)

                        # 更新最佳split（使用测试集R2）
                        test_r2 = result["metrics"]["test"]["R2"]
                        if test_r2 > best_metric:
                            best_metric = test_r2
                            best_split = seed

                        logger.info(f"Split {seed} 完成 (并行)")

                    except Exception as e:
                        logger.error(f"Split {seed} 失败: {e}")
                        import traceback
                        traceback.print_exc()

        # 生成实验总结
        summary = self._generate_summary(split_results, seeds, best_split)

        # 保存总结
        save_json(summary, self.output_dir / "summary.json")
        logger.info(f"\n实验完成！总结已保存到 {self.output_dir / 'summary.json'}")
        logger.info(f"最佳split: {best_split} (测试集R2={best_metric:.4f})")

        return summary

    def _run_single_split(
        self,
        base_dataset: BaseN2ODataset,
        seed: int,
        train_split: float,
        save_dir: Path,
    ) -> dict[str, Any]:
        """运行单个划分的实验"""
        # 设置随机种子
        set_seed(seed)

        # 划分数据集：train:val:test
        # train_split 表示训练集占总数据的比例（默认0.8，即80%）
        # 剩余的 (1-train_split) 平分给验证集和测试集
        n_sequences = len(base_dataset)
        indices = list(range(n_sequences))

        # 第一步：先分出测试集
        test_ratio = (1.0 - train_split) / 2  # 例如：(1-0.8)/2 = 0.1 (10%)
        train_val_indices, test_indices = sklearn_split(
            indices, train_size=1.0-test_ratio, random_state=seed
        )

        # 第二步：从剩余的数据中分出验证集
        # 验证集占剩余数据的比例 = test_ratio / (1 - test_ratio)
        # 例如：0.1 / 0.9 = 1/9，即从90%中分出10%
        val_ratio = test_ratio / (1.0 - test_ratio)
        train_indices, val_indices = sklearn_split(
            train_val_indices, train_size=1.0-val_ratio, random_state=seed
        )

        logger.info(f"数据集划分: {len(train_indices)} 训练, {len(val_indices)} 验证, {len(test_indices)} 测试")

        # 创建子数据集
        train_base = base_dataset[train_indices]
        val_base = base_dataset[val_indices]
        test_base = base_dataset[test_indices]

        # 根据模型类型准备数据和训练
        if self.model_type == "rf":
            result = self._train_rf(train_base, val_base, test_base, save_dir)
        elif self.model_type == "rnn-obs":
            result = self._train_rnn_obs(train_base, val_base, test_base, save_dir)
        elif self.model_type == "rnn-daily":
            result = self._train_rnn_daily(train_base, val_base, test_base, save_dir)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        return result

    def _train_rf(
        self, train_base: BaseN2ODataset, val_base: BaseN2ODataset, test_base: BaseN2ODataset, save_dir: Path
    ) -> dict[str, Any]:
        """训练随机森林模型"""
        # 展开数据为DataFrame（使用RF专用方法，包含Total N amount，不包含Split N amount和ferdur）
        train_df = train_base.flatten_to_dataframe_for_rf()
        val_df = val_base.flatten_to_dataframe_for_rf()
        test_df = test_base.flatten_to_dataframe_for_rf()

        logger.info(f"训练数据: {len(train_df)} 个样本")
        logger.info(f"验证数据: {len(val_df)} 个样本")
        logger.info(f"测试数据: {len(test_df)} 个样本")

        # 训练模型（在验证集上评估）
        model, train_result = train_rf_predictor(
            train_df=train_df, val_df=val_df, test_df=test_df, config=self.config, save_dir=save_dir
        )

        # 保存配置
        save_json(self.config.to_dict(), save_dir / "config.json")

        # 生成可视化和表格（包含测试集）
        self._generate_outputs_rf(
            train_result=train_result,
            model=model,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            save_dir=save_dir,
        )

        # 返回结果（包含测试集指标）
        metrics = {
            "train": train_result["train_metrics"],
            "val": train_result["val_metrics"],
            "test": train_result["test_metrics"],
            "n_parameters": train_result["n_parameters"],
        }

        save_metrics_to_json(metrics, save_dir / "metrics.json")

        return {
            "model": model,
            "metrics": metrics,
            "train_result": train_result,
        }

    def _train_rnn_obs(
        self, train_base: BaseN2ODataset, val_base: BaseN2ODataset, test_base: BaseN2ODataset, save_dir: Path
    ) -> dict[str, Any]:
        """训练观测步长RNN模型"""
        # 创建RNN数据集
        train_dataset = N2ODatasetForObsStepRNN(train_base, fit_scalers=True)
        val_dataset = N2ODatasetForObsStepRNN(
            val_base, fit_scalers=False, scalers=train_dataset.scalers
        )
        test_dataset = N2ODatasetForObsStepRNN(
            test_base, fit_scalers=False, scalers=train_dataset.scalers
        )

        logger.info(f"训练数据: {len(train_dataset)} 个序列")
        logger.info(f"验证数据: {len(val_dataset)} 个序列")
        logger.info(f"测试数据: {len(test_dataset)} 个序列")

        # 保存scalers
        with open(save_dir / "scalers.pkl", "wb") as f:
            pickle.dump(train_dataset.scalers, f)

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # 创建模型
        # 获取分类特征的cardinality
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

        # ObsStepRNN的动态特征多了time_delta
        num_dynamic_numeric = 7  # 原始6个 + time_delta

        model = N2OPredictorRNN(
            num_numeric_static=6,
            num_numeric_dynamic=num_dynamic_numeric,
            categorical_static_cardinalities=categorical_static_cardinalities,
            categorical_dynamic_cardinalities=categorical_dynamic_cardinalities,
            embedding_dim=self.config.embedding_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            rnn_type=self.config.rnn_type,
            dropout=self.config.dropout,
        )

        # 训练模型（在验证集上评估，最后在测试集上测试）
        train_result = train_rnn_predictor(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=self.config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            save_dir=save_dir,
            use_mask=False,
        )

        # 保存配置
        save_json(self.config.to_dict(), save_dir / "config.json")

        # 保存最终模型
        torch.save(model.state_dict(), save_dir / "model.pt")

        # 生成可视化和表格（包含测试集）
        self._generate_outputs_rnn(
            train_result=train_result,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            save_dir=save_dir,
            use_mask=False,
        )

        # 返回结果（包含测试集指标）
        metrics = {
            "train": train_result["train_metrics"],
            "val": train_result["val_metrics"],
            "test": train_result["test_metrics"],
            "n_parameters": train_result["n_parameters"],
        }

        save_metrics_to_json(metrics, save_dir / "metrics.json")

        return {
            "model": model,
            "metrics": metrics,
            "train_result": train_result,
        }

    def _train_rnn_daily(
        self, train_base: BaseN2ODataset, val_base: BaseN2ODataset, test_base: BaseN2ODataset, save_dir: Path
    ) -> dict[str, Any]:
        """训练每日步长RNN模型"""
        # 创建RNN数据集
        train_dataset = N2ODatasetForDailyStepRNN(train_base, fit_scalers=True)
        val_dataset = N2ODatasetForDailyStepRNN(
            val_base, fit_scalers=False, scalers=train_dataset.scalers
        )
        test_dataset = N2ODatasetForDailyStepRNN(
            test_base, fit_scalers=False, scalers=train_dataset.scalers
        )

        logger.info(f"训练数据: {len(train_dataset)} 个序列")
        logger.info(f"验证数据: {len(val_dataset)} 个序列")
        logger.info(f"测试数据: {len(test_dataset)} 个序列")

        # 保存scalers
        with open(save_dir / "scalers.pkl", "wb") as f:
            pickle.dump(train_dataset.scalers, f)

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # 创建模型
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

        model = N2OPredictorRNN(
            num_numeric_static=6,
            num_numeric_dynamic=6,
            categorical_static_cardinalities=categorical_static_cardinalities,
            categorical_dynamic_cardinalities=categorical_dynamic_cardinalities,
            embedding_dim=self.config.embedding_dim,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            rnn_type=self.config.rnn_type,
            dropout=self.config.dropout,
        )

        # 训练模型（在验证集上评估，最后在测试集上测试）
        train_result = train_rnn_predictor(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=self.config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            save_dir=save_dir,
            use_mask=True,  # DailyStepRNN使用掩码
        )

        # 保存配置
        save_json(self.config.to_dict(), save_dir / "config.json")

        # 保存最终模型
        torch.save(model.state_dict(), save_dir / "model.pt")

        # 生成可视化和表格（包含测试集）
        self._generate_outputs_rnn(
            train_result=train_result,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            save_dir=save_dir,
            use_mask=True,
        )

        # 返回结果（包含测试集指标）
        metrics = {
            "train": train_result["train_metrics"],
            "val": train_result["val_metrics"],
            "test": train_result["test_metrics"],
            "n_parameters": train_result["n_parameters"],
        }

        save_metrics_to_json(metrics, save_dir / "metrics.json")

        return {
            "model": model,
            "metrics": metrics,
            "train_result": train_result,
        }

    def _generate_outputs_rf(
        self,
        train_result: dict,
        model: Any,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        save_dir: Path,
    ):
        """生成随机森林模型的输出"""
        figs_dir = save_dir / "figs"
        tables_dir = save_dir / "tables"
        figs_dir.mkdir(exist_ok=True)
        tables_dir.mkdir(exist_ok=True)

        # 1. 预测vs实际值图（使用测试集）
        plot_predictions_vs_actual(
            train_result["train_predictions"],
            train_result["train_targets"],
            train_result["test_predictions"],
            train_result["test_targets"],
            figs_dir / "predictions_vs_actual.png",
        )

        # 2. 特征重要性图
        plot_feature_importance(
            list(train_result["feature_importances"].keys()),
            train_result["feature_importances"],
            figs_dir / "feature_importance.png",
        )

        # 3. 保存预测结果到CSV
        save_predictions_to_csv(
            train_result["train_predictions"],
            train_result["train_targets"],
            tables_dir / "train_predictions.csv",
        )
        save_predictions_to_csv(
            train_result["val_predictions"],
            train_result["val_targets"],
            tables_dir / "val_predictions.csv",
        )
        save_predictions_to_csv(
            train_result["test_predictions"],
            train_result["test_targets"],
            tables_dir / "test_predictions.csv",
        )

        # 4. 保存特征重要性到CSV
        importance_df = pd.DataFrame(
            {
                "feature": list(train_result["feature_importances"].keys()),
                "importance": list(train_result["feature_importances"].values()),
            }
        ).sort_values("importance", ascending=False)
        importance_df.to_csv(tables_dir / "feature_importance.csv", index=False)

        # 5. 序列预测图（使用测试集将DataFrame重构为序列）
        logger.info("生成序列预测图...")

        # 为test_df添加预测值
        test_df_with_pred = test_df.copy()
        test_df_with_pred["predicted_daily_fluxes"] = train_result["test_predictions"]

        # 重构为序列
        test_base_reconstructed = BaseN2ODataset.from_dataframe(test_df_with_pred)

        # 收集每个序列的预测和真实值
        test_preds_by_seq = []
        test_targets_by_seq = []
        test_seq_ids = []

        for seq in test_base_reconstructed.sequences:
            test_preds_by_seq.append(
                test_df_with_pred[
                    (test_df_with_pred["Publication"] == seq["seq_id"][0])
                    & (test_df_with_pred["control_group"] == seq["seq_id"][1])
                ]["predicted_daily_fluxes"].values
            )
            test_targets_by_seq.append(np.array(seq["targets"]))
            test_seq_ids.append(tuple(seq["seq_id"]))

        # 计算每个序列的指标
        seq_metrics = compute_sequence_metrics(test_preds_by_seq, test_targets_by_seq)
        seq_metrics["seq_id_tuple"] = test_seq_ids

        # 选择好的长序列
        good_seq_indices = select_good_sequences(seq_metrics, min_length=15, top_n=5)

        logger.info(f"绘制 {len(good_seq_indices)} 个序列的预测图...")
        for idx in good_seq_indices:
            seq = test_base_reconstructed.sequences[idx]
            seq_id = tuple(seq["seq_id"])
            time_steps = np.array(seq["sowdurs"])
            targets = np.array(seq["targets"])
            predictions = test_preds_by_seq[idx]

            plot_sequence_predictions(
                seq_id,
                time_steps,
                targets,
                predictions,
                figs_dir / f"sequence_predictions_{seq_id[0]}_{seq_id[1]}.png",
                mask=None,
            )

        logger.info(f"输出已保存到 {save_dir}")

    def _generate_outputs_rnn(
        self,
        train_result: dict,
        model: Any,
        train_dataset: Any,
        val_dataset: Any,
        test_dataset: Any,
        save_dir: Path,
        use_mask: bool,
    ):
        """生成RNN模型的输出"""
        figs_dir = save_dir / "figs"
        tables_dir = save_dir / "tables"
        figs_dir.mkdir(exist_ok=True)
        tables_dir.mkdir(exist_ok=True)

        # 1. 训练损失曲线
        plot_train_loss_curve(
            train_result["train_losses"],
            train_result["val_losses"],
            figs_dir / "train_loss_curve.png",
        )

        # 2. 预测vs实际值图（使用测试集）
        plot_predictions_vs_actual(
            train_result["train_predictions"],
            train_result["train_targets"],
            train_result["test_predictions"],
            train_result["test_targets"],
            figs_dir / "predictions_vs_actual.png",
        )

        # 3. 保存预测结果到CSV
        save_predictions_to_csv(
            train_result["train_predictions"],
            train_result["train_targets"],
            tables_dir / "train_predictions.csv",
        )
        save_predictions_to_csv(
            train_result["val_predictions"],
            train_result["val_targets"],
            tables_dir / "val_predictions.csv",
        )
        save_predictions_to_csv(
            train_result["test_predictions"],
            train_result["test_targets"],
            tables_dir / "test_predictions.csv",
        )

        # 4. SHAP分析（简化版，使用梯度近似）
        try:
            logger.info("计算特征重要性（使用梯度近似）...")
            shap_values, feature_names = compute_shap_values(
                model, test_dataset, self.model_type, self.device, max_samples=100
            )

            plot_feature_importance(
                feature_names, shap_values, figs_dir / "feature_importance.png"
            )

            importance_df = pd.DataFrame(
                {"feature": feature_names, "importance": shap_values}
            ).sort_values("importance", ascending=False)
            importance_df.to_csv(tables_dir / "feature_importance.csv", index=False)
        except Exception as e:
            logger.warning(f"SHAP分析失败: {e}")

        # 5. 序列预测图（使用测试集，选择几个表现好的长序列）
        # 收集按序列的预测结果
        test_preds_by_seq = []
        test_targets_by_seq = []
        test_seq_ids = []
        test_masks = []

        for i in range(len(test_dataset)):
            seq = test_dataset[i]
            test_seq_ids.append(tuple(test_dataset.processed_sequences[i]["seq_id"]))

            # 获取预测（需要重新推理）
            model.eval()
            with torch.no_grad():
                static_numeric = seq["static_numeric"].unsqueeze(0).to(self.device)
                dynamic_numeric = seq["dynamic_numeric"].unsqueeze(0).to(self.device)
                static_categorical = (
                    seq["static_categorical"].unsqueeze(0).to(self.device)
                )
                dynamic_categorical = (
                    seq["dynamic_categorical"].unsqueeze(0).to(self.device)
                )
                seq_lengths = torch.tensor([seq["seq_length"]]).to(self.device)

                pred_scaled = model(
                    static_numeric,
                    static_categorical,
                    dynamic_numeric,
                    dynamic_categorical,
                    seq_lengths,
                )
                pred_scaled = pred_scaled.cpu().numpy()[0, : seq["seq_length"]]
                pred_orig = test_dataset.inverse_transform_targets(pred_scaled)

            target_orig = seq["targets_original"].numpy()

            if use_mask:
                mask = seq["mask"].numpy()
                test_preds_by_seq.append(pred_orig[mask])
                test_targets_by_seq.append(target_orig[mask])
                test_masks.append(mask)
            else:
                test_preds_by_seq.append(pred_orig)
                test_targets_by_seq.append(target_orig)
                test_masks.append(None)

        # 计算每个序列的指标
        seq_metrics = compute_sequence_metrics(test_preds_by_seq, test_targets_by_seq)
        seq_metrics["seq_id_tuple"] = test_seq_ids

        # 选择好的长序列
        good_seq_indices = select_good_sequences(seq_metrics, min_length=15, top_n=5)

        logger.info(f"绘制 {len(good_seq_indices)} 个序列的预测图...")
        for idx in good_seq_indices:
            seq = test_dataset.processed_sequences[idx]
            seq_id = tuple(seq["seq_id"])

            # 获取预测和真实值
            model.eval()
            with torch.no_grad():
                seq_data = test_dataset[idx]
                static_numeric = seq_data["static_numeric"].unsqueeze(0).to(self.device)
                dynamic_numeric = (
                    seq_data["dynamic_numeric"].unsqueeze(0).to(self.device)
                )
                static_categorical = (
                    seq_data["static_categorical"].unsqueeze(0).to(self.device)
                )
                dynamic_categorical = (
                    seq_data["dynamic_categorical"].unsqueeze(0).to(self.device)
                )
                seq_lengths = torch.tensor([seq_data["seq_length"]]).to(self.device)

                pred_scaled = model(
                    static_numeric,
                    static_categorical,
                    dynamic_numeric,
                    dynamic_categorical,
                    seq_lengths,
                )
                pred_scaled = pred_scaled.cpu().numpy()[0, : seq["seq_length"]]
                pred_orig = test_dataset.inverse_transform_targets(pred_scaled)

            target_orig = seq_data["targets_original"].numpy()

            # 对于RNN-Daily，只绘制真实测量点（与RNN-Obs保持一致）
            if use_mask:
                mask_seq = seq_data["mask"].numpy()
                # 使用原始的sowdurs（只取真实测量点）
                time_steps = np.array(test_dataset.sequences[idx]["sowdurs"])
                # 只保留真实测量点的数据
                pred_orig = pred_orig[mask_seq]
                target_orig = target_orig[mask_seq]
                mask_for_plot = None  # 不需要在图中区分插值点
            else:
                # ObsStepRNN: 使用原始的sowdur
                time_steps = np.array(test_dataset.sequences[idx]["sowdurs"])
                mask_for_plot = None

            plot_sequence_predictions(
                seq_id,
                time_steps,
                target_orig,
                pred_orig,
                figs_dir / f"sequence_predictions_{seq_id[0]}_{seq_id[1]}.png",
                mask=mask_for_plot,
            )

        logger.info(f"输出已保存到 {save_dir}")

    def _generate_summary(
        self, split_results: list[dict], seeds: list[int], best_split: int
    ) -> dict[str, Any]:
        """生成实验总结"""
        # 收集所有指标
        train_r2_list = []
        train_rmse_list = []
        val_r2_list = []
        val_rmse_list = []
        test_r2_list = []
        test_rmse_list = []

        for result in split_results:
            metrics = result["metrics"]
            train_r2_list.append(metrics["train"]["R2"])
            train_rmse_list.append(metrics["train"]["RMSE"])
            val_r2_list.append(metrics["val"]["R2"])
            val_rmse_list.append(metrics["val"]["RMSE"])
            test_r2_list.append(metrics["test"]["R2"])
            test_rmse_list.append(metrics["test"]["RMSE"])

        summary = {
            "model_type": self.model_type,
            "n_splits": len(seeds),
            "seeds": seeds,
            "best_seed": best_split,  # 最佳split的种子（基于测试集R2，用于compare命令）
            "best_split_seed": best_split,  # 保持向后兼容
            "config": self.config.to_dict(),
            "metrics_summary": {
                "train": {
                    "R2_mean": float(np.mean(train_r2_list)),
                    "R2_std": float(np.std(train_r2_list)),
                    "RMSE_mean": float(np.mean(train_rmse_list)),
                    "RMSE_std": float(np.std(train_rmse_list)),
                },
                "val": {
                    "R2_mean": float(np.mean(val_r2_list)),
                    "R2_std": float(np.std(val_r2_list)),
                    "RMSE_mean": float(np.mean(val_rmse_list)),
                    "RMSE_std": float(np.std(val_rmse_list)),
                },
                "test": {
                    "R2_mean": float(np.mean(test_r2_list)),
                    "R2_std": float(np.std(test_r2_list)),
                    "RMSE_mean": float(np.mean(test_rmse_list)),
                    "RMSE_std": float(np.std(test_rmse_list)),
                },
            },
            "split_results": split_results,
        }

        return summary

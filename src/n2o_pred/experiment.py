"""
实验管理器模块
"""

import json
import pickle
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
            output_dir = Path(__file__).parents[2] / "outputs" / f"exp_{timestamp}"
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
    ) -> dict[str, Any]:
        """
        运行实验

        Args:
            total_splits: 总划分数
            split_seeds: 指定的种子列表（如果提供，则忽略total_splits和split_seed）
            split_seed: 生成随机种子的种子
            train_split: 训练集比例

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

        # 加载基础数据集
        base_dataset = BaseN2ODataset()
        logger.info(f"加载数据集，共 {len(base_dataset)} 个序列")

        # 存储每个split的结果
        split_results = []
        best_split = None
        best_metric = -float("inf")  # 使用R2作为最佳指标

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

                # 更新最佳split
                val_r2 = result["metrics"]["val"]["R2"]
                if val_r2 > best_metric:
                    best_metric = val_r2
                    best_split = seed

                logger.info(f"Split {seed} 完成")

            except Exception as e:
                logger.error(f"Split {seed} 失败: {e}")
                import traceback

                traceback.print_exc()

        # 生成实验总结
        summary = self._generate_summary(split_results, seeds, best_split)

        # 保存总结
        save_json(summary, self.output_dir / "summary.json")
        logger.info(f"\n实验完成！总结已保存到 {self.output_dir / 'summary.json'}")
        logger.info(f"最佳split: {best_split} (R2={best_metric:.4f})")

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

        # 划分数据集
        n_sequences = len(base_dataset)
        indices = list(range(n_sequences))
        train_indices, val_indices = sklearn_split(
            indices, train_size=train_split, random_state=seed
        )

        logger.info(f"数据集划分: {len(train_indices)} 训练, {len(val_indices)} 验证")

        # 创建子数据集
        train_base = base_dataset[train_indices]
        val_base = base_dataset[val_indices]

        # 根据模型类型准备数据和训练
        if self.model_type == "rf":
            result = self._train_rf(train_base, val_base, save_dir)
        elif self.model_type == "rnn-obs":
            result = self._train_rnn_obs(train_base, val_base, save_dir)
        elif self.model_type == "rnn-daily":
            result = self._train_rnn_daily(train_base, val_base, save_dir)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        return result

    def _train_rf(
        self, train_base: BaseN2ODataset, val_base: BaseN2ODataset, save_dir: Path
    ) -> dict[str, Any]:
        """训练随机森林模型"""
        # 展开数据为DataFrame
        train_df = train_base.flatten_to_dataframe()
        val_df = val_base.flatten_to_dataframe()

        logger.info(f"训练数据: {len(train_df)} 个样本")
        logger.info(f"验证数据: {len(val_df)} 个样本")

        # 训练模型
        model, train_result = train_rf_predictor(
            train_df=train_df, val_df=val_df, config=self.config, save_dir=save_dir
        )

        # 保存配置
        save_json(self.config.to_dict(), save_dir / "config.json")

        # 生成可视化和表格
        self._generate_outputs_rf(
            train_result=train_result,
            model=model,
            train_df=train_df,
            val_df=val_df,
            save_dir=save_dir,
        )

        # 返回结果
        metrics = {
            "train": train_result["train_metrics"],
            "val": train_result["val_metrics"],
            "n_parameters": train_result["n_parameters"],
        }

        save_metrics_to_json(metrics, save_dir / "metrics.json")

        return {
            "model": model,
            "metrics": metrics,
            "train_result": train_result,
        }

    def _train_rnn_obs(
        self, train_base: BaseN2ODataset, val_base: BaseN2ODataset, save_dir: Path
    ) -> dict[str, Any]:
        """训练观测步长RNN模型"""
        # 创建RNN数据集
        train_dataset = N2ODatasetForObsStepRNN(train_base, fit_scalers=True)
        val_dataset = N2ODatasetForObsStepRNN(
            val_base, fit_scalers=False, scalers=train_dataset.scalers
        )

        logger.info(f"训练数据: {len(train_dataset)} 个序列")
        logger.info(f"验证数据: {len(val_dataset)} 个序列")

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

        # 训练模型
        train_result = train_rnn_predictor(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            save_dir=save_dir,
            use_mask=False,
        )

        # 保存配置
        save_json(self.config.to_dict(), save_dir / "config.json")

        # 保存最终模型
        torch.save(model.state_dict(), save_dir / "model.pt")

        # 生成可视化和表格
        self._generate_outputs_rnn(
            train_result=train_result,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            save_dir=save_dir,
            use_mask=False,
        )

        # 返回结果
        metrics = {
            "train": train_result["train_metrics"],
            "val": train_result["val_metrics"],
            "n_parameters": train_result["n_parameters"],
        }

        save_metrics_to_json(metrics, save_dir / "metrics.json")

        return {
            "model": model,
            "metrics": metrics,
            "train_result": train_result,
        }

    def _train_rnn_daily(
        self, train_base: BaseN2ODataset, val_base: BaseN2ODataset, save_dir: Path
    ) -> dict[str, Any]:
        """训练每日步长RNN模型"""
        # 创建RNN数据集
        train_dataset = N2ODatasetForDailyStepRNN(train_base, fit_scalers=True)
        val_dataset = N2ODatasetForDailyStepRNN(
            val_base, fit_scalers=False, scalers=train_dataset.scalers
        )

        logger.info(f"训练数据: {len(train_dataset)} 个序列")
        logger.info(f"验证数据: {len(val_dataset)} 个序列")

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

        # 训练模型
        train_result = train_rnn_predictor(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            save_dir=save_dir,
            use_mask=True,  # DailyStepRNN使用掩码
        )

        # 保存配置
        save_json(self.config.to_dict(), save_dir / "config.json")

        # 保存最终模型
        torch.save(model.state_dict(), save_dir / "model.pt")

        # 生成可视化和表格
        self._generate_outputs_rnn(
            train_result=train_result,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            save_dir=save_dir,
            use_mask=True,
        )

        # 返回结果
        metrics = {
            "train": train_result["train_metrics"],
            "val": train_result["val_metrics"],
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
        save_dir: Path,
    ):
        """生成随机森林模型的输出"""
        figs_dir = save_dir / "figs"
        tables_dir = save_dir / "tables"
        figs_dir.mkdir(exist_ok=True)
        tables_dir.mkdir(exist_ok=True)

        # 1. 预测vs实际值图
        plot_predictions_vs_actual(
            train_result["train_predictions"],
            train_result["train_targets"],
            train_result["val_predictions"],
            train_result["val_targets"],
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

        # 4. 保存特征重要性到CSV
        importance_df = pd.DataFrame(
            {
                "feature": list(train_result["feature_importances"].keys()),
                "importance": list(train_result["feature_importances"].values()),
            }
        ).sort_values("importance", ascending=False)
        importance_df.to_csv(tables_dir / "feature_importance.csv", index=False)

        logger.info(f"输出已保存到 {save_dir}")

    def _generate_outputs_rnn(
        self,
        train_result: dict,
        model: Any,
        train_dataset: Any,
        val_dataset: Any,
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

        # 2. 预测vs实际值图
        plot_predictions_vs_actual(
            train_result["train_predictions"],
            train_result["train_targets"],
            train_result["val_predictions"],
            train_result["val_targets"],
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

        # 4. SHAP分析（简化版，使用梯度近似）
        try:
            logger.info("计算特征重要性（使用梯度近似）...")
            shap_values, feature_names = compute_shap_values(
                model, val_dataset, self.model_type, self.device, max_samples=100
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

        # 5. 序列预测图（选择几个表现好的长序列）
        # 收集按序列的预测结果
        val_preds_by_seq = []
        val_targets_by_seq = []
        val_seq_ids = []
        val_masks = []

        for i in range(len(val_dataset)):
            seq = val_dataset[i]
            val_seq_ids.append(tuple(val_dataset.processed_sequences[i]["seq_id"]))

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
                pred_orig = val_dataset.inverse_transform_targets(pred_scaled)

            target_orig = seq["targets_original"].numpy()

            if use_mask:
                mask = seq["mask"].numpy()
                val_preds_by_seq.append(pred_orig[mask])
                val_targets_by_seq.append(target_orig[mask])
                val_masks.append(mask)
            else:
                val_preds_by_seq.append(pred_orig)
                val_targets_by_seq.append(target_orig)
                val_masks.append(None)

        # 计算每个序列的指标
        seq_metrics = compute_sequence_metrics(val_preds_by_seq, val_targets_by_seq)
        seq_metrics["seq_id_tuple"] = val_seq_ids

        # 选择好的长序列
        good_seq_indices = select_good_sequences(seq_metrics, min_length=15, top_n=5)

        logger.info(f"绘制 {len(good_seq_indices)} 个序列的预测图...")
        for idx in good_seq_indices:
            seq = val_dataset.processed_sequences[idx]
            seq_id = tuple(seq["seq_id"])

            # 获取时间步
            if use_mask:
                # DailyStepRNN: 时间步是从start_day开始的每一天
                time_steps = np.arange(
                    val_dataset.daily_sequences[idx]["start_day"],
                    val_dataset.daily_sequences[idx]["start_day"] + seq["seq_length"],
                )
            else:
                # ObsStepRNN: 使用原始的sowdur
                time_steps = np.array(val_dataset.sequences[idx]["sowdurs"])

            # 获取预测和真实值
            model.eval()
            with torch.no_grad():
                seq_data = val_dataset[idx]
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
                pred_orig = val_dataset.inverse_transform_targets(pred_scaled)

            target_orig = seq_data["targets_original"].numpy()
            mask_seq = seq_data["mask"].numpy() if use_mask else None

            plot_sequence_predictions(
                seq_id,
                time_steps,
                target_orig,
                pred_orig,
                figs_dir / f"sequence_predictions_{seq_id[0]}_{seq_id[1]}.png",
                mask=mask_seq,
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

        for result in split_results:
            metrics = result["metrics"]
            train_r2_list.append(metrics["train"]["R2"])
            train_rmse_list.append(metrics["train"]["RMSE"])
            val_r2_list.append(metrics["val"]["R2"])
            val_rmse_list.append(metrics["val"]["RMSE"])

        summary = {
            "model_type": self.model_type,
            "n_splits": len(seeds),
            "seeds": seeds,
            "best_split_seed": best_split,
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
            },
            "split_results": split_results,
        }

        return summary

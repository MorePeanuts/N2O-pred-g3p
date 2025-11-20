"""
模型对比工具模块
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .dataset import BaseN2ODataset
from .predict import N2OPredictor
from .evaluation import (
    compute_sequence_metrics,
    plot_multi_model_sequence_predictions,
    select_good_sequences,
)
from .utils import create_logger, load_json, save_json

logger = create_logger(__name__)


class ModelComparator:
    """模型对比器"""

    def __init__(
        self, exp_dirs: list[Path | str], output_dir: Path | str | None = None
    ):
        """
        Args:
            exp_dirs: 实验目录列表
            output_dir: 输出目录
        """
        self.exp_dirs = [Path(d) for d in exp_dirs]

        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = (
                Path(__file__).parents[2] / "outputs" / f"comparison_{timestamp}"
            )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载实验总结
        self.summaries = []
        self.exp_names = []

        for exp_dir in self.exp_dirs:
            summary_path = exp_dir / "summary.json"
            if not summary_path.exists():
                logger.warning(f"未找到 {summary_path}，跳过该实验")
                continue

            summary = load_json(summary_path)
            self.summaries.append(summary)
            self.exp_names.append(exp_dir.name)

        logger.info(f"加载了 {len(self.summaries)} 个实验")

    def compare(self) -> dict[str, Any]:
        """
        比较模型

        Returns:
            对比结果字典
        """
        if len(self.summaries) < 2:
            logger.error("需要至少2个实验进行对比")
            return {}

        # 找到所有实验的种子交集
        common_seeds = self._find_common_seeds()

        if not common_seeds:
            logger.error("实验之间没有共同的种子，无法进行对比")
            return {}

        logger.info(f"找到 {len(common_seeds)} 个共同的种子: {common_seeds}")

        # 对每个共同的种子进行对比
        all_split_results = []
        for seed in common_seeds:
            logger.info(f"\n{'='*80}")
            logger.info(f"比较种子 {seed} 的结果")
            logger.info(f"{'='*80}")

            split_output_dir = self.output_dir / f"split_{seed}"
            split_output_dir.mkdir(parents=True, exist_ok=True)

            split_result = self._compare_single_split(seed, split_output_dir)
            all_split_results.append(split_result)

        # 生成总体总结
        comparison_summary = self._generate_overall_summary(
            common_seeds, all_split_results
        )
        save_json(comparison_summary, self.output_dir / "comparison_summary.json")

        logger.info(f"\n对比完成！结果已保存到 {self.output_dir}")

        return comparison_summary

    def _find_common_seeds(self) -> list[int]:
        """找到所有实验的种子交集"""
        if len(self.summaries) == 0:
            return []

        # 获取第一个实验的种子
        common_seeds = set(self.summaries[0]["seeds"])

        # 与其他实验的种子求交集
        for summary in self.summaries[1:]:
            common_seeds &= set(summary["seeds"])

        # 转换为排序的列表
        return sorted(list(common_seeds))

    def _compare_single_split(
        self, seed: int, output_dir: Path
    ) -> dict[str, Any]:
        """
        对单个种子下的所有模型进行比较

        Args:
            seed: 种子值
            output_dir: 输出目录

        Returns:
            该种子下的对比结果
        """
        # 收集该种子下所有模型的指标
        model_results = []

        for exp_dir, summary in zip(self.exp_dirs, self.summaries):
            model_type = summary["model_type"]
            exp_name = exp_dir.name

            # 找到该种子对应的split结果
            split_result = None
            for result in summary["split_results"]:
                if result["seed"] == seed:
                    split_result = result
                    break

            if split_result is None:
                logger.warning(f"实验 {exp_name} 没有种子 {seed} 的结果")
                continue

            model_results.append(
                {
                    "exp_name": exp_name,
                    "model_type": model_type,
                    "metrics": split_result["metrics"],
                }
            )

        if len(model_results) < 2:
            logger.warning(f"种子 {seed} 下少于2个模型，跳过对比")
            return {"seed": seed, "models": model_results}

        # 生成对比表格和图表
        figs_dir = output_dir / "figs"
        tables_dir = output_dir / "tables"
        figs_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        # 1. 生成对比表格
        self._generate_split_comparison_table(model_results, tables_dir)

        # 2. 生成对比图表
        self._generate_split_comparison_plots(model_results, figs_dir)

        # 3. 生成序列预测对比图
        self._generate_split_sequence_comparison(seed, model_results, figs_dir)

        return {
            "seed": seed,
            "models": model_results,
        }

    def _generate_split_comparison_table(
        self, model_results: list[dict], tables_dir: Path
    ):
        """为单个split生成对比表格"""
        # 准备数据
        rows = []
        for result in model_results:
            row = {
                "Experiment": result["exp_name"],
                "Model Type": result["model_type"],
                "Train R2": result["metrics"]["train"]["R2"],
                "Train RMSE": result["metrics"]["train"]["RMSE"],
                "Train MAE": result["metrics"]["train"]["MAE"],
                "Test R2": result["metrics"]["test"]["R2"],
                "Test RMSE": result["metrics"]["test"]["RMSE"],
                "Test MAE": result["metrics"]["test"]["MAE"],
            }
            if "n_parameters" in result["metrics"]:
                row["N Parameters"] = result["metrics"]["n_parameters"]
            rows.append(row)

        # 保存为CSV
        df = pd.DataFrame(rows)
        df.to_csv(tables_dir / "comparison.csv", index=False)
        logger.info(f"  对比表格已保存到 {tables_dir / 'comparison.csv'}")

    def _generate_split_comparison_plots(self, model_results: list[dict], figs_dir: Path):
        """为单个split生成对比图表 (美化版)"""
        
        # 设置全局风格（可选，如果想保持和其他图一致可去掉）
        # plt.style.use('seaborn-v0_8-whitegrid') 
        
        # 1. 验证集指标对比条形图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300) # 提高DPI使图像更清晰

        metrics_to_plot = ["R2", "RMSE", "MAE"]
        exp_names = [r["exp_name"] for r in model_results]
        
        # 定义更柔和的颜色方案 (例如使用 tab10 或 set2)
        # 也可以继续使用 viridis，但建议稍微透明一点
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(exp_names))) 

        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            values = [r["metrics"]["test"][metric] for r in model_results]

            # --- 核心美化修改 ---
            # 1. width: 设置柱子宽度 (0.6 比较适中)
            # 2. zorder: 设为3，确保柱子在网格线上面
            # 3. edgecolor/linewidth: 给柱子加个细白边，更有质感
            bars = ax.bar(
                range(len(exp_names)),
                values,
                color=colors,
                width=0.5,         # [关键] 调窄柱子
                edgecolor='white', # [美化] 白色描边
                linewidth=1.2,     # [美化] 描边宽度
                zorder=3,          # [美化] 图层置于网格之上
                alpha=0.9          # [美化] 轻微透明度
            )

            # --- 坐标轴美化 ---
            ax.set_xticks(range(len(exp_names)))
            ax.set_xticklabels(exp_names, rotation=30, ha="right", fontsize=11) # 角度改小一点，易读
            
            # 设置标题和字号
            ax.set_title(f"Test {metric}", fontsize=14, fontweight="bold", pad=15)
            
            # --- 网格与边框 ---
            ax.grid(axis="y", linestyle='--', alpha=0.5, zorder=0) # 网格虚线，置于底层
            
            # 去掉上方和右侧的边框 (Spines)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # 加粗左侧和底部边框
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)

            # --- 数值标签 ---
            # 动态计算Y轴上限，留出 15% 的空间给文字
            if values:
                max_val = max(values)
                min_val = min(values)
                # 如果有负数(如R2)，需要特殊处理ylim，这里简单处理正数场景为主
                range_val = max_val - min(0, min_val)
                ax.set_ylim(bottom=min(0, min_val) * 1.1, top=max_val + range_val * 0.15)

            for bar, val in zip(bars, values):
                height = bar.get_height()
                # 处理负值情况下的标签位置
                y_pos = height if height >= 0 else 0
                va_align = 'bottom' if height >= 0 else 'bottom' 
                
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_pos,
                    f"{val:.4f}",
                    ha="center",
                    va=va_align,
                    fontsize=10,
                    fontweight='medium',
                    color='#333333', # 深灰色文字比纯黑更柔和
                    zorder=4 # 文字在最上层
                )

        plt.tight_layout()
        save_path = figs_dir / "metrics_comparison.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        logger.info(f"  指标对比图已保存到 {save_path}")

    def _generate_overall_summary(
        self, common_seeds: list[int], split_results: list[dict]
    ) -> dict[str, Any]:
        """生成总体总结"""
        # 收集各模型在所有种子上的平均表现
        model_performance = {}

        for exp_dir, summary in zip(self.exp_dirs, self.summaries):
            exp_name = exp_dir.name
            model_type = summary["model_type"]

            # 收集该模型在所有共同种子上的指标
            r2_list = []
            rmse_list = []
            mae_list = []

            for seed in common_seeds:
                for result in summary["split_results"]:
                    if result["seed"] == seed:
                        r2_list.append(result["metrics"]["test"]["R2"])
                        rmse_list.append(result["metrics"]["test"]["RMSE"])
                        mae_list.append(result["metrics"]["test"]["MAE"])
                        break

            model_performance[exp_name] = {
                "model_type": model_type,
                "test_R2_mean": float(np.mean(r2_list)) if r2_list else 0,
                "test_R2_std": float(np.std(r2_list)) if r2_list else 0,
                "test_RMSE_mean": float(np.mean(rmse_list)) if rmse_list else 0,
                "test_RMSE_std": float(np.std(rmse_list)) if rmse_list else 0,
                "test_MAE_mean": float(np.mean(mae_list)) if mae_list else 0,
                "test_MAE_std": float(np.std(mae_list)) if mae_list else 0,
            }

        # 找出最佳模型
        best_by_r2 = max(
            model_performance.items(), key=lambda x: x[1]["test_R2_mean"]
        )
        best_by_rmse = min(
            model_performance.items(), key=lambda x: x[1]["test_RMSE_mean"]
        )

        summary = {
            "n_experiments": len(self.summaries),
            "n_common_seeds": len(common_seeds),
            "common_seeds": common_seeds,
            "experiments": list(model_performance.keys()),
            "model_performance": model_performance,
            "best_model_by_R2": {
                "experiment": best_by_r2[0],
                "model_type": best_by_r2[1]["model_type"],
                "test_R2": best_by_r2[1]["test_R2_mean"],
            },
            "best_model_by_RMSE": {
                "experiment": best_by_rmse[0],
                "model_type": best_by_rmse[1]["model_type"],
                "test_RMSE": best_by_rmse[1]["test_RMSE_mean"],
            },
        }

        return summary

    def _generate_split_sequence_comparison(
        self, seed: int, model_results: list[dict], figs_dir: Path
    ):
        """为单个split生成序列预测对比图"""
        logger.info("  生成序列预测对比图...")

        # 加载基础数据集
        try:
            base_dataset = BaseN2ODataset()
        except Exception as e:
            logger.warning(f"  无法加载基础数据集: {e}，跳过序列预测对比")
            return

        # 使用指定种子划分测试集
        from sklearn.model_selection import train_test_split as sklearn_split

        n_sequences = len(base_dataset)
        indices = list(range(n_sequences))
        _, test_indices = sklearn_split(indices, train_size=0.9, random_state=seed)
        test_base = base_dataset[test_indices]

        # 为每个模型加载预测器并进行预测
        model_predictions_by_seq = {}

        for result in model_results:
            exp_name = result["exp_name"]
            model_type = result["model_type"]

            # 找到对应的实验目录
            exp_dir = None
            for d, s in zip(self.exp_dirs, self.summaries):
                if d.name == exp_name:
                    exp_dir = d
                    break

            if exp_dir is None:
                logger.warning(f"  未找到实验目录: {exp_name}")
                continue

            split_dir = exp_dir / f"split_{seed}"
            if not split_dir.exists():
                logger.warning(f"  未找到split目录: {split_dir}")
                continue

            try:
                # 加载预测器并预测
                predictor = N2OPredictor(split_dir)
                pred_results = predictor.predict(test_base, device="cpu", batch_size=32)
                predictions_flat = pred_results["predictions"]

                # 重构为序列
                if model_type == "rf":
                    test_df = test_base.flatten_to_dataframe_for_rf()
                    test_df["predicted"] = predictions_flat
                    preds_by_seq = []
                    for seq in test_base.sequences:
                        seq_pred = test_df[
                            (test_df["Publication"] == seq["seq_id"][0])
                            & (test_df["control_group"] == seq["seq_id"][1])
                        ]["predicted"].values
                        # 确保长度一致（可能因为排序或其他原因导致长度不匹配）
                        expected_len = seq["seq_length"]
                        if len(seq_pred) != expected_len:
                            logger.warning(
                                f"  序列 {seq['seq_id']} 预测长度 {len(seq_pred)} 与期望长度 {expected_len} 不匹配，跳过该序列"
                            )
                            preds_by_seq.append(None)  # 标记为无效
                        else:
                            preds_by_seq.append(seq_pred)
                else:
                    preds_by_seq = []
                    idx = 0
                    for seq in test_base.sequences:
                        seq_len = seq["seq_length"]
                        preds_by_seq.append(predictions_flat[idx : idx + seq_len])
                        idx += seq_len

                model_predictions_by_seq[exp_name] = preds_by_seq

            except Exception as e:
                logger.warning(f"  加载模型 {exp_name} 失败: {e}")
                continue

        if len(model_predictions_by_seq) < 2:
            logger.warning("  没有足够的模型预测结果，跳过序列预测对比")
            return

        # 计算序列指标并选择好的长序列
        targets_by_seq = [np.array(seq["targets"]) for seq in test_base.sequences]
        n_sequences = len(targets_by_seq)

        # 找到所有模型中都有效的序列（交集）
        # 一个序列有效的条件：
        # 1. 所有模型对该序列的预测都不为 None
        # 2. 所有模型的预测长度都等于目标长度
        valid_indices = []
        for i in range(n_sequences):
            target_len = len(targets_by_seq[i])
            is_valid = True
            for model_name, preds_by_seq in model_predictions_by_seq.items():
                pred = preds_by_seq[i]
                if pred is None:
                    is_valid = False
                    break
                if len(pred) != target_len:
                    logger.warning(
                        f"  序列 {i} 模型 {model_name} 预测长度 {len(pred)} 与目标长度 {target_len} 不匹配，跳过"
                    )
                    is_valid = False
                    break
            if is_valid:
                valid_indices.append(i)

        if len(valid_indices) == 0:
            logger.warning("  没有有效的序列预测结果，跳过序列预测对比")
            return

        logger.info(f"  找到 {len(valid_indices)}/{n_sequences} 个有效序列")

        # 只保留有效序列
        first_model_name = list(model_predictions_by_seq.keys())[0]
        first_model_preds_valid = [model_predictions_by_seq[first_model_name][i] for i in valid_indices]
        targets_by_seq_valid = [targets_by_seq[i] for i in valid_indices]
        valid_sequences = [test_base.sequences[i] for i in valid_indices]

        # 对所有模型的预测也进行过滤
        valid_model_predictions = {}
        for model_name, preds_by_seq in model_predictions_by_seq.items():
            valid_model_predictions[model_name] = [preds_by_seq[i] for i in valid_indices]

        seq_metrics = compute_sequence_metrics(first_model_preds_valid, targets_by_seq_valid)
        seq_ids = [tuple(seq["seq_id"]) for seq in valid_sequences]
        seq_metrics["seq_id_tuple"] = seq_ids

        good_seq_indices = select_good_sequences(seq_metrics, min_length=15, top_n=5)

        if len(good_seq_indices) == 0:
            logger.warning("  没有找到符合条件的长序列")
            return

        logger.info(f"  绘制 {len(good_seq_indices)} 个序列的对比图...")
        for idx in good_seq_indices:
            seq = valid_sequences[idx]
            seq_id = tuple(seq["seq_id"])
            time_steps = np.array(seq["sowdurs"])
            targets = np.array(seq["targets"])

            # 收集所有模型的预测
            model_preds = {}
            for model_name, valid_preds in valid_model_predictions.items():
                model_preds[model_name] = valid_preds[idx]

            plot_multi_model_sequence_predictions(
                seq_id,
                time_steps,
                targets,
                model_preds,
                figs_dir / f"sequence_comparison_{seq_id[0]}_{seq_id[1]}.png",
            )

        logger.info(f"  序列预测对比图已保存")


def compare_experiments(
    exp_dirs: list[Path | str], output_dir: Path | str | None = None
) -> dict[str, Any]:
    """
    比较多个实验的便捷函数

    Args:
        exp_dirs: 实验目录列表
        output_dir: 输出目录

    Returns:
        对比总结
    """
    comparator = ModelComparator(exp_dirs, output_dir)
    return comparator.compare()

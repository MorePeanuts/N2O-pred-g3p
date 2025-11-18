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

        # 检查种子是否一致
        seeds_match = self._check_seeds_consistency()
        if not seeds_match:
            logger.warning("警告：实验使用的种子不完全一致，对比结果可能不公平")

        # 收集对比数据
        comparison_data = self._collect_comparison_data()

        # 生成对比表格
        self._generate_comparison_tables(comparison_data)

        # 生成对比图表
        self._generate_comparison_plots(comparison_data)

        # 生成总结
        comparison_summary = self._generate_comparison_summary(comparison_data)
        save_json(comparison_summary, self.output_dir / "comparison_summary.json")

        logger.info(f"对比结果已保存到 {self.output_dir}")

        return comparison_summary

    def _check_seeds_consistency(self) -> bool:
        """检查种子一致性"""
        if len(self.summaries) == 0:
            return True

        reference_seeds = set(self.summaries[0]["seeds"])

        for summary in self.summaries[1:]:
            if set(summary["seeds"]) != reference_seeds:
                return False

        return True

    def _collect_comparison_data(self) -> dict[str, Any]:
        """收集对比数据"""
        data = {
            "exp_names": self.exp_names,
            "model_types": [s["model_type"] for s in self.summaries],
            "seeds": [s["seeds"] for s in self.summaries],
            "n_parameters": [],
            "train_metrics": {
                "R2": [],
                "RMSE": [],
                "MAE": [],
            },
            "val_metrics": {
                "R2": [],
                "RMSE": [],
                "MAE": [],
            },
            "by_seed": {},  # 按种子组织的数据
        }

        # 收集各实验的指标
        for summary in self.summaries:
            # 获取参数数量（从第一个split获取）
            if summary["split_results"]:
                n_params = summary["split_results"][0]["metrics"].get("n_parameters", 0)
                data["n_parameters"].append(n_params)
            else:
                data["n_parameters"].append(0)

            # 收集平均指标（包括MAE）
            for metric in ["R2", "RMSE"]:
                metric_key = f"{metric}_mean"
                if metric_key in summary["metrics_summary"]["train"]:
                    data["train_metrics"][metric].append(
                        summary["metrics_summary"]["train"][metric_key]
                    )
                else:
                    data["train_metrics"][metric].append(0)
                if metric_key in summary["metrics_summary"]["val"]:
                    data["val_metrics"][metric].append(
                        summary["metrics_summary"]["val"][metric_key]
                    )
                else:
                    data["val_metrics"][metric].append(0)

            # MAE可能不在summary中，尝试从split_results获取
            mae_train_list = []
            mae_val_list = []
            for split_result in summary["split_results"]:
                if "MAE" in split_result["metrics"]["train"]:
                    mae_train_list.append(split_result["metrics"]["train"]["MAE"])
                if "MAE" in split_result["metrics"]["val"]:
                    mae_val_list.append(split_result["metrics"]["val"]["MAE"])

            data["train_metrics"]["MAE"].append(
                np.mean(mae_train_list) if mae_train_list else 0
            )
            data["val_metrics"]["MAE"].append(
                np.mean(mae_val_list) if mae_val_list else 0
            )

        # 按种子组织数据（用于绘制箱线图）
        common_seeds = set(self.summaries[0]["seeds"])
        for summary in self.summaries[1:]:
            common_seeds &= set(summary["seeds"])
        common_seeds = sorted(list(common_seeds))

        for seed in common_seeds:
            data["by_seed"][seed] = {
                "train_R2": [],
                "val_R2": [],
                "train_RMSE": [],
                "val_RMSE": [],
            }

            for summary in self.summaries:
                # 找到该seed的结果
                seed_result = None
                for split_result in summary["split_results"]:
                    if split_result["seed"] == seed:
                        seed_result = split_result
                        break

                if seed_result:
                    data["by_seed"][seed]["train_R2"].append(
                        seed_result["metrics"]["train"]["R2"]
                    )
                    data["by_seed"][seed]["val_R2"].append(
                        seed_result["metrics"]["val"]["R2"]
                    )
                    data["by_seed"][seed]["train_RMSE"].append(
                        seed_result["metrics"]["train"]["RMSE"]
                    )
                    data["by_seed"][seed]["val_RMSE"].append(
                        seed_result["metrics"]["val"]["RMSE"]
                    )

        return data

    def _generate_comparison_tables(self, data: dict[str, Any]):
        """生成对比表格"""
        tables_dir = self.output_dir / "tables"
        tables_dir.mkdir(exist_ok=True)

        # 1. 总体对比表
        summary_df = pd.DataFrame(
            {
                "Experiment": data["exp_names"],
                "Model Type": data["model_types"],
                "N Parameters": data["n_parameters"],
                "Train R2": data["train_metrics"]["R2"],
                "Val R2": data["val_metrics"]["R2"],
                "Train RMSE": data["train_metrics"]["RMSE"],
                "Val RMSE": data["val_metrics"]["RMSE"],
                "Train MAE": data["train_metrics"]["MAE"],
                "Val MAE": data["val_metrics"]["MAE"],
            }
        )
        summary_df.to_csv(tables_dir / "overall_comparison.csv", index=False)
        logger.info(f"总体对比表已保存")

        # 2. 按种子对比表
        if data["by_seed"]:
            seed_comparison = []
            for seed, metrics in data["by_seed"].items():
                row = {"Seed": seed}
                for i, exp_name in enumerate(data["exp_names"]):
                    row[f"{exp_name}_train_R2"] = metrics["train_R2"][i]
                    row[f"{exp_name}_val_R2"] = metrics["val_R2"][i]
                    row[f"{exp_name}_train_RMSE"] = metrics["train_RMSE"][i]
                    row[f"{exp_name}_val_RMSE"] = metrics["val_RMSE"][i]
                seed_comparison.append(row)

            seed_df = pd.DataFrame(seed_comparison)
            seed_df.to_csv(tables_dir / "by_seed_comparison.csv", index=False)
            logger.info(f"按种子对比表已保存")

    def _generate_comparison_plots(self, data: dict[str, Any]):
        """生成对比图表"""
        figs_dir = self.output_dir / "figs"
        figs_dir.mkdir(exist_ok=True)

        # 1. 验证集指标对比条形图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        metrics_to_plot = ["R2", "RMSE", "MAE"]
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            values = data["val_metrics"][metric]

            bars = ax.bar(
                range(len(data["exp_names"])),
                values,
                color=plt.cm.viridis(np.linspace(0, 1, len(data["exp_names"]))),
            )
            ax.set_xticks(range(len(data["exp_names"])))
            ax.set_xticklabels(data["exp_names"], rotation=45, ha="right")
            ax.set_ylabel(metric, fontsize=12)
            ax.set_title(f"Validation {metric}", fontsize=13, fontweight="bold")
            ax.grid(axis="y", alpha=0.3)

            # 添加数值标签
            for j, (bar, val) in enumerate(zip(bars, values)):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        plt.tight_layout()
        plt.savefig(figs_dir / "metrics_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("指标对比条形图已保存")

        # 2. 按种子的箱线图（如果有多个种子）
        if len(data["by_seed"]) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # 准备箱线图数据
            metrics_for_box = [
                ("train_R2", "Train R2"),
                ("val_R2", "Val R2"),
                ("train_RMSE", "Train RMSE"),
                ("val_RMSE", "Val RMSE"),
            ]

            for idx, (metric_key, metric_label) in enumerate(metrics_for_box):
                ax = axes[idx // 2, idx % 2]

                # 收集数据
                box_data = []
                for exp_idx in range(len(data["exp_names"])):
                    exp_values = [
                        data["by_seed"][seed][metric_key][exp_idx]
                        for seed in sorted(data["by_seed"].keys())
                    ]
                    box_data.append(exp_values)

                bp = ax.boxplot(box_data, labels=data["exp_names"], patch_artist=True)

                # 设置颜色
                colors = plt.cm.viridis(np.linspace(0, 1, len(data["exp_names"])))
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                ax.set_xticklabels(data["exp_names"], rotation=45, ha="right")
                ax.set_ylabel(metric_label, fontsize=12)
                ax.set_title(
                    f"{metric_label} Distribution Across Seeds",
                    fontsize=13,
                    fontweight="bold",
                )
                ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                figs_dir / "metrics_distribution.png", dpi=300, bbox_inches="tight"
            )
            plt.close()
            logger.info("指标分布箱线图已保存")

        # 3. 模型复杂度vs性能图
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.cm.viridis(np.linspace(0, 1, len(data["exp_names"])))
        for i, exp_name in enumerate(data["exp_names"]):
            ax.scatter(
                data["n_parameters"][i],
                data["val_metrics"]["R2"][i],
                s=200,
                c=[colors[i]],
                alpha=0.7,
                label=exp_name,
            )

        ax.set_xlabel("Number of Parameters", fontsize=12)
        ax.set_ylabel("Validation R2", fontsize=12)
        ax.set_title("Model Complexity vs Performance", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            figs_dir / "complexity_vs_performance.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        logger.info("复杂度vs性能图已保存")

    def _generate_comparison_summary(self, data: dict[str, Any]) -> dict[str, Any]:
        """生成对比总结"""
        # 找出最佳模型
        best_val_r2_idx = np.argmax(data["val_metrics"]["R2"])
        best_val_rmse_idx = np.argmin(data["val_metrics"]["RMSE"])

        summary = {
            "n_experiments": len(self.summaries),
            "exp_names": data["exp_names"],
            "model_types": data["model_types"],
            "best_model_by_R2": {
                "experiment": data["exp_names"][best_val_r2_idx],
                "model_type": data["model_types"][best_val_r2_idx],
                "val_R2": data["val_metrics"]["R2"][best_val_r2_idx],
            },
            "best_model_by_RMSE": {
                "experiment": data["exp_names"][best_val_rmse_idx],
                "model_type": data["model_types"][best_val_rmse_idx],
                "val_RMSE": data["val_metrics"]["RMSE"][best_val_rmse_idx],
            },
            "overall_metrics": {
                "train": data["train_metrics"],
                "val": data["val_metrics"],
            },
            "n_parameters": data["n_parameters"],
        }

        return summary


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

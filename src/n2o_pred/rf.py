"""
随机森林模型模块
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .preprocessing import (
    CATEGORICAL_DYNAMIC_FEATURES,
    CATEGORICAL_STATIC_FEATURES,
    LABELS,
    NUMERIC_DYNAMIC_FEATURES,
    NUMERIC_STATIC_FEATURES,
)


class N2OPredictorRF:
    """N2O排放预测随机森林模型包装类"""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | int | float = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ):
        """
        Args:
            n_estimators: 树的数量
            max_depth: 树的最大深度
            min_samples_split: 分裂内部节点所需的最小样本数
            min_samples_leaf: 叶节点所需的最小样本数
            max_features: 寻找最佳分裂时考虑的特征数
            random_state: 随机种子
            n_jobs: 并行作业数
            **kwargs: 其他RandomForestRegressor参数
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs,
        )

        self.feature_names = None
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "N2OPredictorRF":
        """
        训练模型

        Args:
            train_df: 训练数据DataFrame

        Returns:
            self
        """
        # 准备特征和标签
        feature_cols = (
            NUMERIC_STATIC_FEATURES
            + NUMERIC_DYNAMIC_FEATURES
            + CATEGORICAL_STATIC_FEATURES
            + CATEGORICAL_DYNAMIC_FEATURES
        )

        X = train_df[feature_cols].values
        y = train_df[LABELS[0]].values

        self.feature_names = feature_cols

        # 训练模型
        self.model.fit(X, y)
        self.is_fitted = True

        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        预测

        Args:
            test_df: 测试数据DataFrame

        Returns:
            预测值数组
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit 方法")

        X = test_df[self.feature_names].values
        return self.model.predict(X)

    def get_feature_importances(self) -> dict[str, float]:
        """
        获取特征重要性

        Returns:
            特征名到重要性的字典
        """
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit 方法")

        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances))

    def save(self, path: Path) -> None:
        """保存模型"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "N2OPredictorRF":
        """加载模型"""
        with open(path, "rb") as f:
            return pickle.load(f)

    def count_parameters(self) -> int:
        """
        估算模型"参数"数量（树的节点数总和）
        注：这不是真正的参数数量，而是模型复杂度的一个指标
        """
        if not self.is_fitted:
            return 0

        total_nodes = sum(tree.tree_.node_count for tree in self.model.estimators_)
        return total_nodes


def train_rf_predictor(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_estimators: int = 100,
    max_depth: int | None = None,
    random_state: int = 42,
    **kwargs,
) -> tuple[N2OPredictorRF, dict[str, Any]]:
    """
    训练随机森林模型的便捷函数

    Args:
        train_df: 训练数据
        test_df: 测试数据
        n_estimators: 树的数量
        max_depth: 树的最大深度
        random_state: 随机种子
        **kwargs: 其他模型参数

    Returns:
        (模型, 预测结果字典)
    """
    # 创建并训练模型
    model = N2OPredictorRF(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        **kwargs,
    )
    model.fit(train_df)

    # 在训练集和测试集上预测
    train_preds = model.predict(train_df)
    test_preds = model.predict(test_df)

    train_targets = train_df[LABELS[0]].values
    test_targets = test_df[LABELS[0]].values

    results = {
        "train_predictions": train_preds,
        "test_predictions": test_preds,
        "train_targets": train_targets,
        "test_targets": test_targets,
        "feature_importances": model.get_feature_importances(),
        "n_parameters": model.count_parameters(),
    }

    return model, results

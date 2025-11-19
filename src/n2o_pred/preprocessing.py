"""
数据预处理模块
"""

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .utils import create_logger

logger = create_logger(__name__)


# 定义字段分组
NUMERIC_STATIC_FEATURES = ["Clay", "CEC", "BD", "pH", "SOC", "TN"]
# 注意：Total N amount 在序列数据中保存，但只有RF模型会使用它
# RNN模型仍然使用前6个特征（不包括Total N amount）
NUMERIC_DYNAMIC_FEATURES = ["Temp", "Prec", "ST", "WFPS", "Split N amount", "ferdur", "Total N amount"]
CATEGORICAL_STATIC_FEATURES = ["crop_class"]
CATEGORICAL_DYNAMIC_FEATURES = ["fertilization_class", "appl_class"]
GROUP_VARIABLES = ["No. of obs", "Publication", "control_group", "sowdur"]
DROP_VARIABLES = ["NH4+-N", "NO3_-N", "MN", "C/N"]
LABELS = ["Daily fluxes"]

# RF模型专用的特征列表（去掉Split N amount和ferdur，使用Total N amount）
NUMERIC_DYNAMIC_FEATURES_RF = ["Temp", "Prec", "ST", "WFPS", "Total N amount"]


def preprocess_data(
    raw_data_path: Path | str = None,
    output_path: Path | str = None,
    encoders_path: Path | str = None,
) -> list[dict[str, Any]]:
    """
    预处理原始数据集

    Args:
        raw_data_path: 原始数据路径，默认为 datasets/data_EUR_raw.csv
        output_path: 输出路径，默认为 datasets/data_EUR_processed.pkl
        encoders_path: 编码器保存路径，默认为 preprocessor/encoders.pkl

    Returns:
        处理后的序列列表
    """
    # 设置默认路径
    if raw_data_path is None:
        raw_data_path = Path(__file__).parents[2] / "datasets" / "data_EUR_raw.csv"
    if output_path is None:
        output_path = Path(__file__).parents[2] / "datasets" / "data_EUR_processed.pkl"
    if encoders_path is None:
        encoders_path = Path(__file__).parents[2] / "preprocessor" / "encoders.pkl"

    raw_data_path = Path(raw_data_path)
    output_path = Path(output_path)
    encoders_path = Path(encoders_path)

    logger.info(f"从 {raw_data_path} 加载原始数据")
    df = pd.read_csv(raw_data_path, index_col=0)
    # 将索引转为列（No. of obs）
    df = df.reset_index()

    logger.info(f"原始数据形状: {df.shape}")

    # 1. 去除冗余字段
    logger.info(f"去除字段: {DROP_VARIABLES}")
    df = df.drop(columns=DROP_VARIABLES, errors="ignore")

    # 2. 将 ferdur=-1 修改为 0
    logger.info("将 ferdur=-1 修改为 0")
    df.loc[df["ferdur"] == -1, "ferdur"] = 0

    # 3. 按照 (Publication, control_group, sowdur) 排序
    logger.info("按照 (Publication, control_group, sowdur) 排序")
    df = df.sort_values(["Publication", "control_group", "sowdur"]).reset_index(
        drop=True
    )

    # 4. 分类变量编码
    logger.info("对分类变量进行编码")
    categorical_features = CATEGORICAL_STATIC_FEATURES + CATEGORICAL_DYNAMIC_FEATURES
    encoders = {}

    for col in categorical_features:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder
        logger.info(f"  {col}: {len(encoder.classes_)} 个类别")

    # 保存编码器
    encoders_path.parent.mkdir(parents=True, exist_ok=True)
    with open(encoders_path, "wb") as f:
        pickle.dump(encoders, f)
    logger.info(f"编码器已保存到 {encoders_path}")

    # 5. 构建时间序列数据
    logger.info("构建时间序列数据")
    sequences = []

    grouped = df.groupby(["Publication", "control_group"])

    for (publication, control_group), group in grouped:
        # 按 sowdur 排序（应该已经排序过，但再次确认）
        group = group.sort_values("sowdur").reset_index(drop=True)

        # TN字段前向填充（同序列内）
        group["TN"] = group["TN"].ffill()
        # 如果仍有NaN（第一个值就是NaN），用后向填充
        group["TN"] = group["TN"].bfill()
        # 如果还有NaN，用该列的均值填充（极端情况）
        if group["TN"].isna().any():
            mean_tn = df["TN"].mean()
            group["TN"] = group["TN"].fillna(mean_tn if not pd.isna(mean_tn) else 1.0)

        # 构建序列数据结构
        seq_data = {
            "seq_id": [int(publication), int(control_group)],
            "seq_length": len(group),
            "No. of obs": group["No. of obs"].tolist(),
            "sowdurs": group["sowdur"].tolist(),
            "numeric_static": group[NUMERIC_STATIC_FEATURES].iloc[0].values.tolist(),
            "numeric_dynamic": group[NUMERIC_DYNAMIC_FEATURES].values.tolist(),
            "categorical_static": group[CATEGORICAL_STATIC_FEATURES]
            .iloc[0]
            .values.tolist(),
            "categorical_dynamic": group[CATEGORICAL_DYNAMIC_FEATURES].values.tolist(),
            "targets": group[LABELS].values.flatten().tolist(),
        }

        sequences.append(seq_data)

    logger.info(f"共构建 {len(sequences)} 个序列")

    # 统计序列长度分布
    seq_lengths = [seq["seq_length"] for seq in sequences]
    logger.info(
        f"序列长度统计: min={min(seq_lengths)}, max={max(seq_lengths)}, "
        f"mean={sum(seq_lengths) / len(seq_lengths):.2f}, "
        f"median={sorted(seq_lengths)[len(seq_lengths) // 2]}"
    )

    # 6. 保存处理后的数据
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(sequences, f)
    logger.info(f"处理后的数据已保存到 {output_path}")

    return sequences


if __name__ == "__main__":
    # 运行数据预处理
    sequences = preprocess_data()
    print(f"数据预处理完成，共 {len(sequences)} 个序列")

"""
数据集类模块
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .preprocessing import (
    CATEGORICAL_DYNAMIC_FEATURES,
    CATEGORICAL_STATIC_FEATURES,
    NUMERIC_DYNAMIC_FEATURES,
    NUMERIC_STATIC_FEATURES,
)
from .utils import SymlogTransformer


class BaseN2ODataset(Dataset):
    """基础N2O数据集，用于加载处理后的序列数据"""

    seq_data_path = Path(__file__).parents[2] / "datasets" / "data_EUR_processed.pkl"

    def __init__(self, sequences: list[dict[str, Any]] | None = None):
        """
        Args:
            sequences: 序列数据列表，如果为None则从默认路径加载
        """
        if sequences is None:
            with open(self.seq_data_path, "rb") as f:
                self.sequences = pickle.load(f)
        else:
            self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int | list[int]) -> dict[str, Any] | "BaseN2ODataset":
        """
        支持单个索引或索引列表

        如果是单个索引，返回序列字典
        如果是索引列表，返回包含这些序列的新数据集
        """
        if isinstance(idx, (list, np.ndarray)):
            subset_sequences = [self.sequences[i] for i in idx]
            return BaseN2ODataset(sequences=subset_sequences)
        return self.sequences[idx]

    def flatten_to_dataframe(self) -> pd.DataFrame:
        """
        将序列数据展开为DataFrame格式，供随机森林使用

        Returns:
            展开后的DataFrame，每行对应一个观测点
        """
        rows = []

        for seq in self.sequences:
            seq_length = seq["seq_length"]

            for i in range(seq_length):
                row = {}

                # 添加ID信息
                row["No. of obs"] = seq["No. of obs"][i]
                row["Publication"] = seq["seq_id"][0]
                row["control_group"] = seq["seq_id"][1]
                row["sowdur"] = seq["sowdurs"][i]

                # 添加静态数值特征
                for j, feat_name in enumerate(NUMERIC_STATIC_FEATURES):
                    row[feat_name] = seq["numeric_static"][j]

                # 添加动态数值特征
                for j, feat_name in enumerate(NUMERIC_DYNAMIC_FEATURES):
                    row[feat_name] = seq["numeric_dynamic"][i][j]

                # 添加静态分类特征
                for j, feat_name in enumerate(CATEGORICAL_STATIC_FEATURES):
                    row[feat_name] = seq["categorical_static"][j]

                # 添加动态分类特征
                for j, feat_name in enumerate(CATEGORICAL_DYNAMIC_FEATURES):
                    row[feat_name] = seq["categorical_dynamic"][i][j]

                # 添加目标值（如果存在）
                if "targets" in seq:
                    row["Daily fluxes"] = seq["targets"][i]

                rows.append(row)

        return pd.DataFrame(rows)

    def flatten_to_dataframe_for_rf(self) -> pd.DataFrame:
        """
        将序列数据展开为DataFrame格式，专门供随机森林使用

        与 flatten_to_dataframe() 的区别：
        - 使用 Total N amount（索引6）而不是 Split N amount（索引4）和 ferdur（索引5）

        Returns:
            展开后的DataFrame，每行对应一个观测点
        """
        from .preprocessing import NUMERIC_DYNAMIC_FEATURES_RF

        rows = []

        for seq in self.sequences:
            seq_length = seq["seq_length"]

            for i in range(seq_length):
                row = {}

                # 添加ID信息
                row["No. of obs"] = seq["No. of obs"][i]
                row["Publication"] = seq["seq_id"][0]
                row["control_group"] = seq["seq_id"][1]
                row["sowdur"] = seq["sowdurs"][i]

                # 添加静态数值特征
                for j, feat_name in enumerate(NUMERIC_STATIC_FEATURES):
                    row[feat_name] = seq["numeric_static"][j]

                # 添加动态数值特征（RF专用）
                # 索引映射：0=Temp, 1=Prec, 2=ST, 3=WFPS, 4=Split N amount, 5=ferdur, 6=Total N amount
                dynamic_feat = seq["numeric_dynamic"][i]
                row["Temp"] = dynamic_feat[0]
                row["Prec"] = dynamic_feat[1]
                row["ST"] = dynamic_feat[2]
                row["WFPS"] = dynamic_feat[3]
                row["Total N amount"] = dynamic_feat[6]  # 使用Total N amount而不是Split N amount和ferdur

                # 添加静态分类特征
                for j, feat_name in enumerate(CATEGORICAL_STATIC_FEATURES):
                    row[feat_name] = seq["categorical_static"][j]

                # 添加动态分类特征
                for j, feat_name in enumerate(CATEGORICAL_DYNAMIC_FEATURES):
                    row[feat_name] = seq["categorical_dynamic"][i][j]

                # 添加目标值（如果存在）
                if "targets" in seq:
                    row["Daily fluxes"] = seq["targets"][i]

                rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> "BaseN2ODataset":
        """
        从DataFrame重构序列数据

        Args:
            df: 展开的DataFrame，包含Publication, control_group, sowdur等字段

        Returns:
            BaseN2ODataset实例
        """
        from .preprocessing import NUMERIC_DYNAMIC_FEATURES_RF

        # 按(Publication, control_group)分组，并按sowdur排序
        df = df.sort_values(["Publication", "control_group", "sowdur"])

        # 检测 DataFrame 是来自 RF 还是 RNN
        # RF 的 DataFrame 不包含 'Split N amount' 和 'ferdur'，但包含 'Total N amount'
        is_rf_dataframe = (
            "Total N amount" in df.columns
            and "Split N amount" not in df.columns
            and "ferdur" not in df.columns
        )

        sequences = []
        for (pub, ctrl_grp), group_df in df.groupby(
            ["Publication", "control_group"], sort=False
        ):
            seq_length = len(group_df)

            # 提取静态数值特征（取第一行即可）
            numeric_static = [
                group_df.iloc[0][feat] for feat in NUMERIC_STATIC_FEATURES
            ]

            # 提取动态数值特征
            if is_rf_dataframe:
                # RF DataFrame: 需要将5个特征扩展为7个特征
                # RF特征: Temp, Prec, ST, WFPS, Total N amount
                # 完整特征: Temp, Prec, ST, WFPS, Split N amount, ferdur, Total N amount
                numeric_dynamic = []
                for i in range(seq_length):
                    row = group_df.iloc[i]
                    # 按照 NUMERIC_DYNAMIC_FEATURES 的顺序构建完整的特征列表
                    full_features = [
                        row["Temp"],           # 索引 0
                        row["Prec"],           # 索引 1
                        row["ST"],             # 索引 2
                        row["WFPS"],           # 索引 3
                        0.0,                   # 索引 4: Split N amount (RF不使用，填0)
                        0.0,                   # 索引 5: ferdur (RF不使用，填0)
                        row["Total N amount"], # 索引 6
                    ]
                    numeric_dynamic.append(full_features)
            else:
                # RNN DataFrame 或完整的 DataFrame
                numeric_dynamic = [
                    [group_df.iloc[i][feat] for feat in NUMERIC_DYNAMIC_FEATURES]
                    for i in range(seq_length)
                ]

            # 提取静态分类特征
            categorical_static = [
                group_df.iloc[0][feat] for feat in CATEGORICAL_STATIC_FEATURES
            ]

            # 提取动态分类特征
            categorical_dynamic = [
                [group_df.iloc[i][feat] for feat in CATEGORICAL_DYNAMIC_FEATURES]
                for i in range(seq_length)
            ]

            # 提取目标值（如果存在）
            targets = (
                group_df["Daily fluxes"].tolist()
                if "Daily fluxes" in group_df.columns
                else [0.0] * seq_length
            )

            # 提取其他信息
            no_of_obs = group_df["No. of obs"].tolist()
            sowdurs = group_df["sowdur"].tolist()

            seq = {
                "seq_id": [pub, ctrl_grp],
                "seq_length": seq_length,
                "No. of obs": no_of_obs,
                "sowdurs": sowdurs,
                "numeric_static": numeric_static,
                "numeric_dynamic": numeric_dynamic,
                "categorical_static": categorical_static,
                "categorical_dynamic": categorical_dynamic,
                "targets": targets,
            }

            sequences.append(seq)

        return BaseN2ODataset(sequences=sequences)


class N2ODatasetForObsStepRNN(Dataset):
    """观测步长RNN的数据集（每个观测点作为一个时间步）"""

    def __init__(
        self,
        base_data: BaseN2ODataset,
        fit_scalers: bool = True,
        scalers: dict | None = None,
    ):
        """
        Args:
            base_data: 基础数据集
            fit_scalers: 是否拟合缩放器（训练集为True，测试集为False）
            scalers: 已拟合的缩放器（如果fit_scalers=False，必须提供）
        """
        self.base_data = base_data
        self.sequences = base_data.sequences

        # 特征工程
        if fit_scalers:
            self.scalers = self._fit_scalers()
        else:
            if scalers is None:
                raise ValueError("当 fit_scalers=False 时，必须提供 scalers 参数")
            self.scalers = scalers

        # 应用特征工程
        self._apply_feature_engineering()

    def _fit_scalers(self) -> dict:
        """拟合缩放器"""
        scalers = {}

        # 收集所有数据用于拟合
        all_targets = []
        all_temp = []
        all_st = []
        all_wfps = []
        all_clay = []
        all_cec = []
        all_bd = []
        all_ph = []
        all_soc = []
        all_tn = []
        all_prec = []
        all_split_n = []
        all_ferdur = []

        for seq in self.sequences:
            all_targets.extend(seq["targets"])

            # 静态数值特征
            clay, cec, bd, ph, soc, tn = seq["numeric_static"]
            seq_len = seq["seq_length"]
            all_clay.extend([clay] * seq_len)
            all_cec.extend([cec] * seq_len)
            all_bd.extend([bd] * seq_len)
            all_ph.extend([ph] * seq_len)
            all_soc.extend([soc] * seq_len)
            all_tn.extend([tn] * seq_len)

            # 动态数值特征（RNN只使用前6个，索引0-5）
            for i in range(seq_len):
                dynamic_feat = seq["numeric_dynamic"][i]
                temp = dynamic_feat[0]  # Temp
                prec = dynamic_feat[1]  # Prec
                st = dynamic_feat[2]    # ST
                wfps = dynamic_feat[3]  # WFPS
                split_n = dynamic_feat[4]  # Split N amount
                ferdur = dynamic_feat[5]   # ferdur
                all_temp.append(temp)
                all_prec.append(prec)
                all_st.append(st)
                all_wfps.append(wfps)
                all_split_n.append(split_n)
                all_ferdur.append(ferdur)

        # Daily fluxes: Symlog + StandardScaler
        scalers["target_symlog"] = SymlogTransformer(C=1.0)
        target_symlog = scalers["target_symlog"].fit_transform(
            np.array(all_targets).reshape(-1, 1)
        )
        scalers["target_scaler"] = StandardScaler()
        scalers["target_scaler"].fit(target_symlog)

        # Prec, Split N amount, ferdur: log(x+1) + StandardScaler
        prec_log = np.log1p(np.array(all_prec).reshape(-1, 1))
        scalers["prec_scaler"] = StandardScaler()
        scalers["prec_scaler"].fit(prec_log)

        split_n_log = np.log1p(np.array(all_split_n).reshape(-1, 1))
        scalers["split_n_scaler"] = StandardScaler()
        scalers["split_n_scaler"].fit(split_n_log)

        ferdur_log = np.log1p(np.array(all_ferdur).reshape(-1, 1))
        scalers["ferdur_scaler"] = StandardScaler()
        scalers["ferdur_scaler"].fit(ferdur_log)

        # 其他数值特征: StandardScaler
        scalers["temp_scaler"] = StandardScaler()
        scalers["temp_scaler"].fit(np.array(all_temp).reshape(-1, 1))

        scalers["st_scaler"] = StandardScaler()
        scalers["st_scaler"].fit(np.array(all_st).reshape(-1, 1))

        scalers["wfps_scaler"] = StandardScaler()
        scalers["wfps_scaler"].fit(np.array(all_wfps).reshape(-1, 1))

        scalers["clay_scaler"] = StandardScaler()
        scalers["clay_scaler"].fit(np.array(all_clay).reshape(-1, 1))

        scalers["cec_scaler"] = StandardScaler()
        scalers["cec_scaler"].fit(np.array(all_cec).reshape(-1, 1))

        scalers["bd_scaler"] = StandardScaler()
        scalers["bd_scaler"].fit(np.array(all_bd).reshape(-1, 1))

        scalers["ph_scaler"] = StandardScaler()
        scalers["ph_scaler"].fit(np.array(all_ph).reshape(-1, 1))

        scalers["soc_scaler"] = StandardScaler()
        scalers["soc_scaler"].fit(np.array(all_soc).reshape(-1, 1))

        scalers["tn_scaler"] = StandardScaler()
        scalers["tn_scaler"].fit(np.array(all_tn).reshape(-1, 1))

        return scalers

    def _apply_feature_engineering(self):
        """应用特征工程并保存处理后的数据"""
        self.processed_sequences = []

        for seq in self.sequences:
            seq_len = seq["seq_length"]

            # 提取特征
            clay, cec, bd, ph, soc, tn = seq["numeric_static"]
            sowdurs = np.array(seq["sowdurs"])

            # 计算 time_delta
            time_delta = np.zeros(seq_len)
            if seq_len > 1:
                time_delta[1:] = sowdurs[1:] - sowdurs[:-1]

            # 处理静态数值特征
            static_numeric = []
            static_numeric.append(self.scalers["clay_scaler"].transform([[clay]])[0, 0])
            static_numeric.append(self.scalers["cec_scaler"].transform([[cec]])[0, 0])
            static_numeric.append(self.scalers["bd_scaler"].transform([[bd]])[0, 0])
            static_numeric.append(self.scalers["ph_scaler"].transform([[ph]])[0, 0])
            static_numeric.append(self.scalers["soc_scaler"].transform([[soc]])[0, 0])
            static_numeric.append(self.scalers["tn_scaler"].transform([[tn]])[0, 0])

            # 处理动态数值特征（RNN只使用前6个，索引0-5）
            dynamic_numeric = []
            for i in range(seq_len):
                dynamic_feat = seq["numeric_dynamic"][i]
                temp = dynamic_feat[0]  # Temp
                prec = dynamic_feat[1]  # Prec
                st = dynamic_feat[2]    # ST
                wfps = dynamic_feat[3]  # WFPS
                split_n = dynamic_feat[4]  # Split N amount
                ferdur = dynamic_feat[5]   # ferdur

                feat = []
                feat.append(self.scalers["temp_scaler"].transform([[temp]])[0, 0])
                feat.append(
                    self.scalers["prec_scaler"].transform([[np.log1p(prec)]])[0, 0]
                )
                feat.append(self.scalers["st_scaler"].transform([[st]])[0, 0])
                feat.append(self.scalers["wfps_scaler"].transform([[wfps]])[0, 0])
                feat.append(
                    self.scalers["split_n_scaler"].transform([[np.log1p(split_n)]])[
                        0, 0
                    ]
                )
                feat.append(
                    self.scalers["ferdur_scaler"].transform([[np.log1p(ferdur)]])[0, 0]
                )
                feat.append(time_delta[i])  # 添加 time_delta

                dynamic_numeric.append(feat)

            # 处理目标值
            targets = np.array(seq["targets"]).reshape(-1, 1)
            targets_symlog = self.scalers["target_symlog"].transform(targets)
            targets_scaled = (
                self.scalers["target_scaler"].transform(targets_symlog).flatten()
            )

            processed_seq = {
                "seq_id": seq["seq_id"],
                "seq_length": seq_len,
                "static_numeric": np.array(static_numeric, dtype=np.float32),
                "dynamic_numeric": np.array(dynamic_numeric, dtype=np.float32),
                "static_categorical": np.array(
                    seq["categorical_static"], dtype=np.int64
                ),
                "dynamic_categorical": np.array(
                    seq["categorical_dynamic"], dtype=np.int64
                ),
                "targets": np.array(targets_scaled, dtype=np.float32),
                "targets_original": np.array(seq["targets"], dtype=np.float32),
            }

            self.processed_sequences.append(processed_seq)

    def __len__(self) -> int:
        return len(self.processed_sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """返回一个序列的张量"""
        seq = self.processed_sequences[idx]
        return {
            "static_numeric": torch.from_numpy(seq["static_numeric"]),
            "dynamic_numeric": torch.from_numpy(seq["dynamic_numeric"]),
            "static_categorical": torch.from_numpy(seq["static_categorical"]),
            "dynamic_categorical": torch.from_numpy(seq["dynamic_categorical"]),
            "targets": torch.from_numpy(seq["targets"]),
            "targets_original": torch.from_numpy(seq["targets_original"]),
            "seq_length": seq["seq_length"],
        }

    def inverse_transform_targets(self, targets_scaled: np.ndarray) -> np.ndarray:
        """将缩放后的目标值转换回原始空间"""
        targets_symlog = self.scalers["target_scaler"].inverse_transform(
            targets_scaled.reshape(-1, 1)
        )
        targets_original = self.scalers["target_symlog"].inverse_transform(
            targets_symlog
        )
        return targets_original.flatten()


class N2ODatasetForDailyStepRNN(Dataset):
    """每日步长RNN的数据集（每天作为一个时间步）"""

    def __init__(
        self,
        base_data: BaseN2ODataset,
        fit_scalers: bool = True,
        scalers: dict | None = None,
    ):
        """
        Args:
            base_data: 基础数据集
            fit_scalers: 是否拟合缩放器（训练集为True，测试集为False）
            scalers: 已拟合的缩放器（如果fit_scalers=False，必须提供）
        """
        self.base_data = base_data
        self.sequences = base_data.sequences

        # 先展开序列为每日步长
        self._expand_sequences_to_daily()

        # 特征工程（只在真实测量点上拟合）
        if fit_scalers:
            self.scalers = self._fit_scalers()
        else:
            if scalers is None:
                raise ValueError("当 fit_scalers=False 时，必须提供 scalers 参数")
            self.scalers = scalers

        # 应用特征工程
        self._apply_feature_engineering()

    def _expand_sequences_to_daily(self):
        """将序列展开为每日步长"""
        self.daily_sequences = []

        for seq in self.sequences:
            sowdurs = np.array(seq["sowdurs"])
            seq_len = seq["seq_length"]

            # 确定序列的总天数范围
            start_day = int(sowdurs[0])
            end_day = int(sowdurs[-1])
            total_days = end_day - start_day + 1

            # 初始化每日数据
            daily_data = {
                "seq_id": seq["seq_id"],
                "start_day": start_day,
                "total_days": total_days,
                "numeric_static": seq["numeric_static"],
                "categorical_static": seq["categorical_static"],
                "daily_numeric": [],
                "daily_categorical": [],
                "daily_targets": [],
                "mask": [],  # True表示真实测量点，False表示插值点
                "original_indices": [],  # 保存mask=True对应的原始索引，-1表示插值点
                "original_sowdurs": [],  # 保存原始sowdur值
            }

            # 创建观测索引映射 - 处理可能的int重复问题
            # 使用列表存储所有观测点信息
            obs_list = [(int(sowdurs[i]), i, sowdurs[i]) for i in range(seq_len)]
            obs_ptr = 0  # 当前待匹配的观测点指针

            # 同时创建一个辅助映射用于插值（取首次出现的索引）
            # 这仅用于插值，不会影响真实测量点的匹配
            obs_map_for_interp = {}
            for int_day, orig_idx, _ in obs_list:
                if int_day not in obs_map_for_interp:
                    obs_map_for_interp[int_day] = orig_idx

            # 提取原始动态数据用于插值
            temp_vals = [seq["numeric_dynamic"][i][0] for i in range(seq_len)]
            st_vals = [seq["numeric_dynamic"][i][2] for i in range(seq_len)]
            wfps_vals = [seq["numeric_dynamic"][i][3] for i in range(seq_len)]

            # 用于记录施肥信息
            last_fert_day = -1
            last_fert_amount = 0

            for day in range(start_day, end_day + 1):
                # 检查是否是真实测量点 - 按顺序匹配
                is_obs = False
                idx = -1
                orig_sowdur = float(day)  # 默认使用day作为sowdur

                # 先检查当前指针位置的观测点
                if obs_ptr < len(obs_list):
                    obs_int_day, obs_idx, obs_sowdur_val = obs_list[obs_ptr]
                    if obs_int_day == day:
                        # 找到匹配的观测点
                        is_obs = True
                        idx = obs_idx
                        orig_sowdur = obs_sowdur_val
                        obs_ptr += 1
                    elif obs_int_day < day:
                        # 跳过一些天？尝试在剩余的观测点中查找
                        for i in range(obs_ptr, len(obs_list)):
                            obs_int_day_i, obs_idx_i, obs_sowdur_val_i = obs_list[i]
                            if obs_int_day_i == day:
                                is_obs = True
                                idx = obs_idx_i
                                orig_sowdur = obs_sowdur_val_i
                                obs_ptr = i + 1
                                break
                            elif obs_int_day_i > day:
                                break
                else:
                    # 已经处理完所有观测点，回退查找
                    for obs_int_day_i, obs_idx_i, obs_sowdur_val_i in obs_list:
                        if obs_int_day_i == day:
                            is_obs = True
                            idx = obs_idx_i
                            orig_sowdur = obs_sowdur_val_i
                            break

                if is_obs and idx >= 0:
                    # 真实测量点
                    # RNN 只使用前 6 个动态特征（索引 0-5），不包括 Total N amount（索引 6）
                    # 特征顺序：Temp, Prec, ST, WFPS, Split N amount, ferdur
                    daily_numeric = list(seq["numeric_dynamic"][idx][:6])
                    daily_categorical = list(seq["categorical_dynamic"][idx])
                    target = seq["targets"][idx]

                    # 更新施肥信息
                    if daily_numeric[4] > 0:  # Split N amount
                        last_fert_day = day
                        last_fert_amount = daily_numeric[4]

                    daily_data["daily_numeric"].append(daily_numeric)
                    daily_data["daily_categorical"].append(daily_categorical)
                    daily_data["daily_targets"].append(target)
                    daily_data["mask"].append(True)
                    daily_data["original_indices"].append(idx)
                    daily_data["original_sowdurs"].append(orig_sowdur)
                else:
                    # 插值点
                    # 线性插值数值特征（除了Prec和Split N amount）
                    day_idx = day - start_day

                    # 找到前后最近的观测点用于插值 - 从obs_list获取所有int_day
                    obs_int_days = [od for od, _, _ in obs_list]
                    before_days = [d for d in obs_int_days if d < day]
                    after_days = [d for d in obs_int_days if d > day]

                    if before_days and after_days:
                        before_day = max(before_days)
                        after_day = min(after_days)
                        before_idx = obs_map_for_interp[before_day]
                        after_idx = obs_map_for_interp[after_day]

                        # 插值比例
                        alpha = (day - before_day) / (after_day - before_day)

                        # Temp, ST, WFPS 线性插值
                        temp = (
                            seq["numeric_dynamic"][before_idx][0] * (1 - alpha)
                            + seq["numeric_dynamic"][after_idx][0] * alpha
                        )
                        st = (
                            seq["numeric_dynamic"][before_idx][2] * (1 - alpha)
                            + seq["numeric_dynamic"][after_idx][2] * alpha
                        )
                        wfps = (
                            seq["numeric_dynamic"][before_idx][3] * (1 - alpha)
                            + seq["numeric_dynamic"][after_idx][3] * alpha
                        )
                    elif before_days:
                        # 只有之前的点，前向填充
                        before_day = max(before_days)
                        before_idx = obs_map_for_interp[before_day]
                        temp = seq["numeric_dynamic"][before_idx][0]
                        st = seq["numeric_dynamic"][before_idx][2]
                        wfps = seq["numeric_dynamic"][before_idx][3]
                    else:
                        # 只有之后的点，后向填充
                        after_day = min(after_days)
                        after_idx = obs_map_for_interp[after_day]
                        temp = seq["numeric_dynamic"][after_idx][0]
                        st = seq["numeric_dynamic"][after_idx][2]
                        wfps = seq["numeric_dynamic"][after_idx][3]

                    # Prec填充为0（非观测日无降水记录）
                    prec = 0.0

                    # Split N amount: 使用前向填充，保持为"上次施肥量"
                    if before_days:
                        before_day = max(before_days)
                        before_idx = obs_map_for_interp[before_day]
                        split_n = seq["numeric_dynamic"][before_idx][4]
                    else:
                        # 后向填充
                        after_day = min(after_days)
                        after_idx = obs_map_for_interp[after_day]
                        split_n = seq["numeric_dynamic"][after_idx][4]

                    # ferdur: 距离上次施肥的天数
                    if last_fert_day >= 0:
                        ferdur = day - last_fert_day
                    else:
                        ferdur = 0.0

                    daily_numeric = [temp, prec, st, wfps, split_n, ferdur]

                    # 分类特征前向填充
                    if before_days:
                        before_day = max(before_days)
                        before_idx = obs_map_for_interp[before_day]
                        daily_categorical = list(seq["categorical_dynamic"][before_idx])
                    else:
                        # 后向填充
                        after_day = min(after_days)
                        after_idx = obs_map_for_interp[after_day]
                        daily_categorical = list(seq["categorical_dynamic"][after_idx])

                    # 目标值插值（但不会用于损失计算）
                    if before_days and after_days:
                        before_day = max(before_days)
                        after_day = min(after_days)
                        before_idx = obs_map_for_interp[before_day]
                        after_idx = obs_map_for_interp[after_day]
                        alpha = (day - before_day) / (after_day - before_day)
                        target = (
                            seq["targets"][before_idx] * (1 - alpha)
                            + seq["targets"][after_idx] * alpha
                        )
                    elif before_days:
                        before_day = max(before_days)
                        before_idx = obs_map_for_interp[before_day]
                        target = seq["targets"][before_idx]
                    else:
                        after_day = min(after_days)
                        after_idx = obs_map_for_interp[after_day]
                        target = seq["targets"][after_idx]

                    daily_data["daily_numeric"].append(daily_numeric)
                    daily_data["daily_categorical"].append(daily_categorical)
                    daily_data["daily_targets"].append(target)
                    daily_data["mask"].append(False)
                    daily_data["original_indices"].append(-1)
                    daily_data["original_sowdurs"].append(float(day))

            self.daily_sequences.append(daily_data)

    def _fit_scalers(self) -> dict:
        """拟合缩放器（只在真实测量点上拟合）"""
        scalers = {}

        # 收集所有真实测量点的数据
        all_targets = []
        all_temp = []
        all_st = []
        all_wfps = []
        all_clay = []
        all_cec = []
        all_bd = []
        all_ph = []
        all_soc = []
        all_tn = []
        all_prec = []
        all_split_n = []
        all_ferdur = []

        for seq in self.daily_sequences:
            clay, cec, bd, ph, soc, tn = seq["numeric_static"]

            for i in range(seq["total_days"]):
                if seq["mask"][i]:  # 只收集真实测量点
                    all_targets.append(seq["daily_targets"][i])

                    # 静态特征
                    all_clay.append(clay)
                    all_cec.append(cec)
                    all_bd.append(bd)
                    all_ph.append(ph)
                    all_soc.append(soc)
                    all_tn.append(tn)

                    # 动态特征（RNN只使用前6个，索引0-5）
                    daily_feat = seq["daily_numeric"][i]
                    temp = daily_feat[0]  # Temp
                    prec = daily_feat[1]  # Prec
                    st = daily_feat[2]    # ST
                    wfps = daily_feat[3]  # WFPS
                    split_n = daily_feat[4]  # Split N amount
                    ferdur = daily_feat[5]   # ferdur
                    all_temp.append(temp)
                    all_prec.append(prec)
                    all_st.append(st)
                    all_wfps.append(wfps)
                    all_split_n.append(split_n)
                    all_ferdur.append(ferdur)

        # Daily fluxes: Symlog + StandardScaler
        scalers["target_symlog"] = SymlogTransformer(C=1.0)
        target_symlog = scalers["target_symlog"].fit_transform(
            np.array(all_targets).reshape(-1, 1)
        )
        scalers["target_scaler"] = StandardScaler()
        scalers["target_scaler"].fit(target_symlog)

        # Prec, Split N amount, ferdur: log(x+1) + StandardScaler
        prec_log = np.log1p(np.array(all_prec).reshape(-1, 1))
        scalers["prec_scaler"] = StandardScaler()
        scalers["prec_scaler"].fit(prec_log)

        split_n_log = np.log1p(np.array(all_split_n).reshape(-1, 1))
        scalers["split_n_scaler"] = StandardScaler()
        scalers["split_n_scaler"].fit(split_n_log)

        ferdur_log = np.log1p(np.array(all_ferdur).reshape(-1, 1))
        scalers["ferdur_scaler"] = StandardScaler()
        scalers["ferdur_scaler"].fit(ferdur_log)

        # 其他数值特征: StandardScaler
        scalers["temp_scaler"] = StandardScaler()
        scalers["temp_scaler"].fit(np.array(all_temp).reshape(-1, 1))

        scalers["st_scaler"] = StandardScaler()
        scalers["st_scaler"].fit(np.array(all_st).reshape(-1, 1))

        scalers["wfps_scaler"] = StandardScaler()
        scalers["wfps_scaler"].fit(np.array(all_wfps).reshape(-1, 1))

        scalers["clay_scaler"] = StandardScaler()
        scalers["clay_scaler"].fit(np.array(all_clay).reshape(-1, 1))

        scalers["cec_scaler"] = StandardScaler()
        scalers["cec_scaler"].fit(np.array(all_cec).reshape(-1, 1))

        scalers["bd_scaler"] = StandardScaler()
        scalers["bd_scaler"].fit(np.array(all_bd).reshape(-1, 1))

        scalers["ph_scaler"] = StandardScaler()
        scalers["ph_scaler"].fit(np.array(all_ph).reshape(-1, 1))

        scalers["soc_scaler"] = StandardScaler()
        scalers["soc_scaler"].fit(np.array(all_soc).reshape(-1, 1))

        scalers["tn_scaler"] = StandardScaler()
        scalers["tn_scaler"].fit(np.array(all_tn).reshape(-1, 1))

        return scalers

    def _apply_feature_engineering(self):
        """应用特征工程"""
        self.processed_sequences = []

        for seq in self.daily_sequences:
            # 处理静态数值特征
            clay, cec, bd, ph, soc, tn = seq["numeric_static"]
            static_numeric = []
            static_numeric.append(self.scalers["clay_scaler"].transform([[clay]])[0, 0])
            static_numeric.append(self.scalers["cec_scaler"].transform([[cec]])[0, 0])
            static_numeric.append(self.scalers["bd_scaler"].transform([[bd]])[0, 0])
            static_numeric.append(self.scalers["ph_scaler"].transform([[ph]])[0, 0])
            static_numeric.append(self.scalers["soc_scaler"].transform([[soc]])[0, 0])
            static_numeric.append(self.scalers["tn_scaler"].transform([[tn]])[0, 0])

            # 处理动态数值特征（RNN只使用前6个，索引0-5）
            dynamic_numeric = []
            for i in range(seq["total_days"]):
                daily_feat = seq["daily_numeric"][i]
                temp = daily_feat[0]  # Temp
                prec = daily_feat[1]  # Prec
                st = daily_feat[2]    # ST
                wfps = daily_feat[3]  # WFPS
                split_n = daily_feat[4]  # Split N amount
                ferdur = daily_feat[5]   # ferdur

                feat = []
                feat.append(self.scalers["temp_scaler"].transform([[temp]])[0, 0])
                feat.append(
                    self.scalers["prec_scaler"].transform([[np.log1p(prec)]])[0, 0]
                )
                feat.append(self.scalers["st_scaler"].transform([[st]])[0, 0])
                feat.append(self.scalers["wfps_scaler"].transform([[wfps]])[0, 0])
                feat.append(
                    self.scalers["split_n_scaler"].transform([[np.log1p(split_n)]])[
                        0, 0
                    ]
                )
                feat.append(
                    self.scalers["ferdur_scaler"].transform([[np.log1p(ferdur)]])[0, 0]
                )

                dynamic_numeric.append(feat)

            # 处理目标值
            targets = np.array(seq["daily_targets"]).reshape(-1, 1)
            targets_symlog = self.scalers["target_symlog"].transform(targets)
            targets_scaled = (
                self.scalers["target_scaler"].transform(targets_symlog).flatten()
            )

            processed_seq = {
                "seq_id": seq["seq_id"],
                "seq_length": seq["total_days"],
                "static_numeric": np.array(static_numeric, dtype=np.float32),
                "dynamic_numeric": np.array(dynamic_numeric, dtype=np.float32),
                "static_categorical": np.array(
                    seq["categorical_static"], dtype=np.int64
                ),
                "dynamic_categorical": np.array(
                    seq["daily_categorical"], dtype=np.int64
                ),
                "targets": np.array(targets_scaled, dtype=np.float32),
                "targets_original": np.array(seq["daily_targets"], dtype=np.float32),
                "mask": np.array(seq["mask"], dtype=bool),
            }

            self.processed_sequences.append(processed_seq)

    def __len__(self) -> int:
        return len(self.processed_sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """返回一个序列的张量"""
        seq = self.processed_sequences[idx]
        return {
            "static_numeric": torch.from_numpy(seq["static_numeric"]),
            "dynamic_numeric": torch.from_numpy(seq["dynamic_numeric"]),
            "static_categorical": torch.from_numpy(seq["static_categorical"]),
            "dynamic_categorical": torch.from_numpy(seq["dynamic_categorical"]),
            "targets": torch.from_numpy(seq["targets"]),
            "targets_original": torch.from_numpy(seq["targets_original"]),
            "mask": torch.from_numpy(seq["mask"]),
            "seq_length": seq["seq_length"],
        }

    def inverse_transform_targets(self, targets_scaled: np.ndarray) -> np.ndarray:
        """将缩放后的目标值转换回原始空间"""
        targets_symlog = self.scalers["target_scaler"].inverse_transform(
            targets_scaled.reshape(-1, 1)
        )
        targets_original = self.scalers["target_symlog"].inverse_transform(
            targets_symlog
        )
        return targets_original.flatten()


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    自定义collate函数，用于处理变长序列
    """
    # 获取批次中的最大序列长度
    max_len = max(item["seq_length"] for item in batch)
    batch_size = len(batch)

    # 获取特征维度
    num_static_numeric = batch[0]["static_numeric"].shape[0]
    num_dynamic_numeric = batch[0]["dynamic_numeric"].shape[1]
    num_static_categorical = batch[0]["static_categorical"].shape[0]
    num_dynamic_categorical = batch[0]["dynamic_categorical"].shape[1]

    # 初始化填充后的张量
    static_numeric = torch.zeros(batch_size, num_static_numeric)
    dynamic_numeric = torch.zeros(batch_size, max_len, num_dynamic_numeric)
    static_categorical = torch.zeros(
        batch_size, num_static_categorical, dtype=torch.long
    )
    dynamic_categorical = torch.zeros(
        batch_size, max_len, num_dynamic_categorical, dtype=torch.long
    )
    targets = torch.zeros(batch_size, max_len)
    targets_original = torch.zeros(batch_size, max_len)
    seq_lengths = torch.zeros(batch_size, dtype=torch.long)

    # 检查是否有mask（DailyStepRNN才有）
    has_mask = "mask" in batch[0]
    if has_mask:
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    # 填充数据
    for i, item in enumerate(batch):
        seq_len = item["seq_length"]
        seq_lengths[i] = seq_len

        static_numeric[i] = item["static_numeric"]
        dynamic_numeric[i, :seq_len] = item["dynamic_numeric"]
        static_categorical[i] = item["static_categorical"]
        dynamic_categorical[i, :seq_len] = item["dynamic_categorical"]
        targets[i, :seq_len] = item["targets"]
        targets_original[i, :seq_len] = item["targets_original"]

        if has_mask:
            mask[i, :seq_len] = item["mask"]

    result = {
        "static_numeric": static_numeric,
        "dynamic_numeric": dynamic_numeric,
        "static_categorical": static_categorical,
        "dynamic_categorical": dynamic_categorical,
        "targets": targets,
        "targets_original": targets_original,
        "seq_lengths": seq_lengths,
    }

    if has_mask:
        result["mask"] = mask

    return result


class TifDataLoader:
    """TIF格式数据加载器，用于在TIF和RNN输入格式之间转换"""

    # 作物目录名到编码器类别名的映射
    DIR_TO_CROP = {
        "barley": "barley",
        "barley2": "barley",
        "fruit": "fruit",
        "legume": "legume",
        "maize": "maize",
        "oilplant": "oilplant",
        "oilplant2": "oilplant",
        "other_cereal": "other_cereal",
        "other_cereal2": "other_cereal",
        "potato": "potato",
        "rice": "rice",
        "sugar": "sugarbeet",  # 特殊映射
        "vegetables": "vegetables",
        "wheat": "wheat",
        "wheat2": "wheat",
    }


    # 排除的分类值（无施肥情况）
    EXCLUDED_FERT = ["NO"]  # fertilization_class 中排除
    EXCLUDED_APPL = ["no"]  # appl_class 中排除

    def __init__(
        self,
        input_dir: Path | str,
        encoders_path: Path | str | None = None,
    ):
        """
        初始化TIF数据加载器

        Args:
            input_dir: TIF文件所在目录 (如 input_2020)
            encoders_path: 编码器文件路径，默认为 preprocessor/encoders.pkl
        """
        import pickle
        import rasterio

        self.input_dir = Path(input_dir)

        # 加载编码器
        if encoders_path is None:
            encoders_path = Path(__file__).parents[2] / "preprocessor" / "encoders.pkl"
        else:
            encoders_path = Path(encoders_path)

        with open(encoders_path, "rb") as f:
            self.encoders = pickle.load(f)

        # 加载所有TIF文件
        self._load_tif_files()

        # 获取有效的分类组合
        self._init_valid_combinations()

    def _load_tif_files(self):
        """加载所有TIF文件到内存"""
        import rasterio

        # 静态特征文件映射
        static_file_map = {
            "Clay": "CLAY.tif",
            "CEC": "CEC.tif",
            "BD": "BD.tif",
            "pH": "PH.tif",
            "SOC": "SOC.tif",
            "TN": "TN.tif",
        }

        # 动态特征文件映射
        dynamic_file_map = {
            "Temp": "T.tif",
            "Prec": "P.tif",
            "ST": "ST.tif",
            "WFPS": "WFPS.tif",
        }

        # 加载静态特征
        self.static_data = {}
        for feat_name, filename in static_file_map.items():
            filepath = self.input_dir / filename
            with rasterio.open(filepath) as src:
                self.static_data[feat_name] = src.read(1)  # shape: (160, 280)
                # 保存地理参考信息（用于输出）
                if not hasattr(self, "profile"):
                    self.profile = src.profile.copy()
                    self.transform = src.transform
                    self.crs = src.crs

        # 加载动态特征
        self.dynamic_data = {}
        for feat_name, filename in dynamic_file_map.items():
            filepath = self.input_dir / filename
            with rasterio.open(filepath) as src:
                self.dynamic_data[feat_name] = src.read()  # shape: (366, 160, 280)
                self.n_days = src.count

        # 加载每个作物的特定数据
        # 结构: {crop_name: {source_name: {"N_amount": ..., "ferdur": ..., "sowdur": ...}}}
        self.crop_data = {}
        for dir_name in self.DIR_TO_CROP.keys():
            crop_dir = self.input_dir / dir_name
            if crop_dir.exists():
                crop_name = self.DIR_TO_CROP[dir_name]
                source_name = dir_name  # 使用目录名作为来源标识

                if crop_name not in self.crop_data:
                    self.crop_data[crop_name] = {}

                self.crop_data[crop_name][source_name] = {}

                # N_amount (静态，单波段)
                n_amount_path = crop_dir / "N_amount.tif"
                with rasterio.open(n_amount_path) as src:
                    self.crop_data[crop_name][source_name]["N_amount"] = src.read(1)  # shape: (160, 280)

                # ferdur (动态，多波段)
                ferdur_path = crop_dir / "ferdur.tif"
                with rasterio.open(ferdur_path) as src:
                    self.crop_data[crop_name][source_name]["ferdur"] = src.read()  # shape: (366, 160, 280)

                # sowdur (动态，多波段) - 可能需要用于确定有效区域
                sowdur_path = crop_dir / "sowdur.tif"
                with rasterio.open(sowdur_path) as src:
                    self.crop_data[crop_name][source_name]["sowdur"] = src.read()  # shape: (366, 160, 280)

        # 获取空间维度
        self.height, self.width = self.static_data["Clay"].shape

    def _init_valid_combinations(self):
        """初始化有效的分类特征组合"""
        # 获取所有类别
        crop_classes = list(self.encoders["crop_class"].classes_)
        fert_classes = list(self.encoders["fertilization_class"].classes_)
        appl_classes = list(self.encoders["appl_class"].classes_)

        # 过滤排除的类别
        valid_fert = [f for f in fert_classes if f not in self.EXCLUDED_FERT]
        valid_appl = [a for a in appl_classes if a not in self.EXCLUDED_APPL]

        # 生成所有有效组合 (crop, fert, appl, source)
        self.valid_combinations = []
        for crop in crop_classes:
            # 检查该作物是否有对应的数据
            if crop in self.crop_data:
                # 遍历该作物的所有数据源
                for source_name in self.crop_data[crop].keys():
                    for fert in valid_fert:
                        for appl in valid_appl:
                            self.valid_combinations.append((crop, fert, appl, source_name))

    def get_prediction_combinations(self) -> list[tuple[str, str, str, str]]:
        """
        返回所有有效的 (crop, fert, appl, source) 组合

        Returns:
            组合列表，每个元素为 (crop_name, fert_name, appl_name, source_name)
        """
        return self.valid_combinations

    def create_rnn_dataset(
        self,
        crop: str,
        fert: str,
        appl: str,
        source_name: str,
        scalers: dict,
        model_type: str = "rnn-obs",
    ) -> "TifPixelDataset":
        """
        为特定的分类组合创建可用于DataLoader的数据集

        Args:
            crop: 作物类别名称
            fert: 施肥类型名称
            appl: 施肥方式名称
            source_name: 数据源名称（目录名）
            scalers: 训练时使用的特征缩放器
            model_type: 模型类型 ("rnn-obs" 或 "rnn-daily")

        Returns:
            TifPixelDataset 实例
        """
        # 获取编码后的分类特征值
        crop_idx = self.encoders["crop_class"].transform([crop])[0]
        fert_idx = self.encoders["fertilization_class"].transform([fert])[0]
        appl_idx = self.encoders["appl_class"].transform([appl])[0]

        # 获取该作物特定来源的数据
        crop_n_amount = self.crop_data[crop][source_name]["N_amount"]  # (160, 280)
        crop_ferdur = self.crop_data[crop][source_name]["ferdur"]  # (366, 160, 280)
        crop_sowdur = self.crop_data[crop][source_name]["sowdur"]  # (366, 160, 280)

        # 确定有效像素（需要所有特征都非NaN）
        valid_mask = ~np.isnan(crop_sowdur[0])
        # 检查静态特征
        for feat_name in ["Clay", "CEC", "BD", "pH", "SOC", "TN"]:
            valid_mask = valid_mask & ~np.isnan(self.static_data[feat_name])
        # 检查动态特征（使用第一天）
        for feat_name in ["Temp", "Prec", "ST", "WFPS"]:
            valid_mask = valid_mask & ~np.isnan(self.dynamic_data[feat_name][0])
        # 检查作物特定数据
        valid_mask = valid_mask & ~np.isnan(crop_n_amount)
        valid_mask = valid_mask & ~np.isnan(crop_ferdur[0])

        # 收集有效像素的数据
        valid_indices = np.where(valid_mask)
        n_pixels = len(valid_indices[0])

        # 构建每个像素的序列数据
        pixel_sequences = []

        for i in range(n_pixels):
            row, col = valid_indices[0][i], valid_indices[1][i]

            # 静态数值特征: Clay, CEC, BD, pH, SOC, TN
            static_numeric = np.array([
                self.static_data["Clay"][row, col],
                self.static_data["CEC"][row, col],
                self.static_data["BD"][row, col],
                self.static_data["pH"][row, col],
                self.static_data["SOC"][row, col],
                self.static_data["TN"][row, col],
            ], dtype=np.float32)

            # 动态数值特征: Temp, Prec, ST, WFPS, Split N amount, ferdur[, time_delta]
            # rnn-daily: 6个特征 (不含time_delta)
            # rnn-obs: 7个特征 (含time_delta)
            num_dynamic_features = 7 if model_type == "rnn-obs" else 6
            dynamic_numeric = np.zeros((self.n_days, num_dynamic_features), dtype=np.float32)

            # 创建有效天掩码：sowdur >= 0 表示播种后（有效），sowdur < 0 表示休耕期（无效）
            valid_days_mask = np.zeros(self.n_days, dtype=bool)

            for day in range(self.n_days):
                sowdur_val = crop_sowdur[day, row, col]
                is_valid = sowdur_val >= 0  # 播种后为有效天
                valid_days_mask[day] = is_valid

                dynamic_numeric[day, 0] = self.dynamic_data["Temp"][day, row, col]
                dynamic_numeric[day, 1] = self.dynamic_data["Prec"][day, row, col]
                dynamic_numeric[day, 2] = self.dynamic_data["ST"][day, row, col]
                dynamic_numeric[day, 3] = self.dynamic_data["WFPS"][day, row, col]
                dynamic_numeric[day, 4] = crop_n_amount[row, col]  # Split N amount (静态值复制到每天)
                # ferdur: -1 表示尚未施肥，转换为 0
                ferdur_val = crop_ferdur[day, row, col]
                dynamic_numeric[day, 5] = max(0.0, ferdur_val)
                # time_delta: 仅对 rnn-obs 模型添加
                if model_type == "rnn-obs":
                    dynamic_numeric[day, 6] = 1.0 if day > 0 else 0.0

            # 静态分类特征
            static_categorical = np.array([crop_idx], dtype=np.int64)

            # 动态分类特征 (每天都相同)
            dynamic_categorical = np.zeros((self.n_days, 2), dtype=np.int64)
            dynamic_categorical[:, 0] = fert_idx
            dynamic_categorical[:, 1] = appl_idx

            pixel_sequences.append({
                "pixel_idx": (row, col),
                "static_numeric": static_numeric,
                "dynamic_numeric": dynamic_numeric,
                "static_categorical": static_categorical,
                "dynamic_categorical": dynamic_categorical,
                "valid_days_mask": valid_days_mask,  # 添加有效天掩码
            })

        return TifPixelDataset(pixel_sequences, scalers, self.n_days, model_type)

    def save_predictions(
        self,
        predictions: np.ndarray,
        pixel_indices: list[tuple[int, int]],
        crop: str,
        fert: str,
        appl: str,
        source_name: str,
        output_dir: Path | str,
    ):
        """
        将预测结果保存为TIF文件

        Args:
            predictions: 预测结果，shape (n_pixels, n_days)
            pixel_indices: 像素索引列表
            crop: 作物类别名称
            fert: 施肥类型名称
            appl: 施肥方式名称
            source_name: 数据源名称（目录名）
            output_dir: 输出目录
        """
        import rasterio

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建输出数组
        output_data = np.full((self.n_days, self.height, self.width), np.nan, dtype=np.float32)

        # 填充预测结果
        for i, (row, col) in enumerate(pixel_indices):
            output_data[:, row, col] = predictions[i]

        # 更新profile用于多波段输出
        output_profile = self.profile.copy()
        output_profile.update(
            count=self.n_days,
            dtype="float32",
        )

        # 保存TIF文件，文件名包含来源标识
        filename = f"{crop}_{source_name}_{fert}_{appl}.tif"
        output_path = output_dir / filename

        with rasterio.open(output_path, "w", **output_profile) as dst:
            dst.write(output_data)

        return output_path


class TifPixelDataset(Dataset):
    """TIF像素数据集，用于批量预测"""

    def __init__(
        self,
        pixel_sequences: list[dict],
        scalers: dict,
        n_days: int,
        model_type: str = "rnn-obs",
    ):
        """
        Args:
            pixel_sequences: 像素序列数据列表
            scalers: 特征缩放器
            n_days: 时间步数
            model_type: 模型类型 ("rnn-obs" 或 "rnn-daily")
        """
        self.pixel_sequences = pixel_sequences
        self.scalers = scalers
        self.n_days = n_days
        self.model_type = model_type

        # 应用特征缩放
        self._apply_feature_engineering()

    def _apply_feature_engineering(self):
        """应用特征工程（向量化版本，大幅提升性能）"""
        n_pixels = len(self.pixel_sequences)

        if n_pixels == 0:
            self.processed_sequences = []
            return

        # 收集所有像素的原始数据到数组中
        all_static = np.stack([seq["static_numeric"] for seq in self.pixel_sequences])  # (n_pixels, 6)
        all_dynamic = np.stack([seq["dynamic_numeric"] for seq in self.pixel_sequences])  # (n_pixels, n_days, 6 or 7)
        all_valid_masks = np.stack([seq["valid_days_mask"] for seq in self.pixel_sequences])  # (n_pixels, n_days)

        # 检测动态特征数量 (6 for rnn-daily, 7 for rnn-obs)
        num_dynamic_features = all_dynamic.shape[2]

        # 批量处理静态数值特征
        # static_numeric: Clay, CEC, BD, pH, SOC, TN
        static_scaled = np.zeros((n_pixels, 6), dtype=np.float32)
        static_scaled[:, 0] = self.scalers["clay_scaler"].transform(all_static[:, 0:1]).flatten()
        static_scaled[:, 1] = self.scalers["cec_scaler"].transform(all_static[:, 1:2]).flatten()
        static_scaled[:, 2] = self.scalers["bd_scaler"].transform(all_static[:, 2:3]).flatten()
        static_scaled[:, 3] = self.scalers["ph_scaler"].transform(all_static[:, 3:4]).flatten()
        static_scaled[:, 4] = self.scalers["soc_scaler"].transform(all_static[:, 4:5]).flatten()
        static_scaled[:, 5] = self.scalers["tn_scaler"].transform(all_static[:, 5:6]).flatten()

        # 裁剪静态特征以防止溢出 (float32 范围约 ±3.4e38，但实际值不应超过 ±10 标准差)
        static_scaled = np.clip(static_scaled, -10, 10)

        # 批量处理动态数值特征
        # dynamic_numeric: Temp, Prec, ST, WFPS, Split N amount, ferdur[, time_delta]
        # 先重塑为 (n_pixels * n_days, 1) 进行批量变换
        n_total = n_pixels * self.n_days

        dynamic_scaled = np.zeros((n_pixels, self.n_days, num_dynamic_features), dtype=np.float32)

        # Temp
        temp_flat = all_dynamic[:, :, 0].reshape(n_total, 1)
        dynamic_scaled[:, :, 0] = self.scalers["temp_scaler"].transform(temp_flat).reshape(n_pixels, self.n_days)

        # Prec (需要 log1p 变换)
        prec_flat = np.log1p(all_dynamic[:, :, 1]).reshape(n_total, 1)
        dynamic_scaled[:, :, 1] = self.scalers["prec_scaler"].transform(prec_flat).reshape(n_pixels, self.n_days)

        # ST
        st_flat = all_dynamic[:, :, 2].reshape(n_total, 1)
        dynamic_scaled[:, :, 2] = self.scalers["st_scaler"].transform(st_flat).reshape(n_pixels, self.n_days)

        # WFPS
        wfps_flat = all_dynamic[:, :, 3].reshape(n_total, 1)
        dynamic_scaled[:, :, 3] = self.scalers["wfps_scaler"].transform(wfps_flat).reshape(n_pixels, self.n_days)

        # Split N amount (需要 log1p 变换)
        split_n_flat = np.log1p(all_dynamic[:, :, 4]).reshape(n_total, 1)
        dynamic_scaled[:, :, 4] = self.scalers["split_n_scaler"].transform(split_n_flat).reshape(n_pixels, self.n_days)

        # ferdur (需要 log1p 变换)
        ferdur_flat = np.log1p(all_dynamic[:, :, 5]).reshape(n_total, 1)
        dynamic_scaled[:, :, 5] = self.scalers["ferdur_scaler"].transform(ferdur_flat).reshape(n_pixels, self.n_days)

        # time_delta (仅对 rnn-obs，不需要缩放，直接复制)
        if num_dynamic_features == 7:
            dynamic_scaled[:, :, 6] = all_dynamic[:, :, 6]

        # 裁剪动态特征以防止溢出（除了time_delta）
        dynamic_scaled[:, :, :6] = np.clip(dynamic_scaled[:, :, :6], -10, 10)

        # 构建处理后的序列列表
        self.processed_sequences = []
        for i in range(n_pixels):
            processed_seq = {
                "pixel_idx": self.pixel_sequences[i]["pixel_idx"],
                "static_numeric": static_scaled[i],
                "dynamic_numeric": dynamic_scaled[i],
                "static_categorical": self.pixel_sequences[i]["static_categorical"],
                "dynamic_categorical": self.pixel_sequences[i]["dynamic_categorical"],
                "valid_days_mask": all_valid_masks[i],  # 保存有效天掩码
                "seq_length": self.n_days,
            }
            self.processed_sequences.append(processed_seq)

    def __len__(self) -> int:
        return len(self.processed_sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """返回一个像素序列的张量"""
        seq = self.processed_sequences[idx]
        return {
            "static_numeric": torch.from_numpy(seq["static_numeric"]),
            "dynamic_numeric": torch.from_numpy(seq["dynamic_numeric"]),
            "static_categorical": torch.from_numpy(seq["static_categorical"]),
            "dynamic_categorical": torch.from_numpy(seq["dynamic_categorical"]),
            "valid_days_mask": torch.from_numpy(seq["valid_days_mask"]),  # 返回有效天掩码
            "seq_length": seq["seq_length"],
            "pixel_idx": seq["pixel_idx"],
        }

    def get_pixel_indices(self) -> list[tuple[int, int]]:
        """返回所有像素的索引"""
        return [seq["pixel_idx"] for seq in self.processed_sequences]

    def get_valid_masks(self) -> np.ndarray:
        """返回所有像素的有效天掩码，shape: (n_pixels, n_days)"""
        return np.array([seq["valid_days_mask"] for seq in self.processed_sequences], dtype=bool)

    def inverse_transform_targets(self, targets_scaled: np.ndarray) -> np.ndarray:
        """将缩放后的目标值转换回原始空间"""
        targets_symlog = self.scalers["target_scaler"].inverse_transform(
            targets_scaled.reshape(-1, 1)
        )
        targets_original = self.scalers["target_symlog"].inverse_transform(
            targets_symlog
        )
        return targets_original.flatten()

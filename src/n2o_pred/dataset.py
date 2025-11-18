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
    
    seq_data_path = Path(__file__).parents[2] / 'datasets' / 'data_EUR_processed.pkl'
    
    def __init__(self, sequences: list[dict[str, Any]] | None = None):
        """
        Args:
            sequences: 序列数据列表，如果为None则从默认路径加载
        """
        if sequences is None:
            with open(self.seq_data_path, 'rb') as f:
                self.sequences = pickle.load(f)
        else:
            self.sequences = sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int | list[int]) -> dict[str, Any] | 'BaseN2ODataset':
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
            seq_length = seq['seq_length']
            
            for i in range(seq_length):
                row = {}
                
                # 添加ID信息
                row['No. of obs'] = seq['No. of obs'][i]
                row['Publication'] = seq['seq_id'][0]
                row['control_group'] = seq['seq_id'][1]
                row['sowdur'] = seq['sowdurs'][i]
                
                # 添加静态数值特征
                for j, feat_name in enumerate(NUMERIC_STATIC_FEATURES):
                    row[feat_name] = seq['numeric_static'][j]
                
                # 添加动态数值特征
                for j, feat_name in enumerate(NUMERIC_DYNAMIC_FEATURES):
                    row[feat_name] = seq['numeric_dynamic'][i][j]
                
                # 添加静态分类特征
                for j, feat_name in enumerate(CATEGORICAL_STATIC_FEATURES):
                    row[feat_name] = seq['categorical_static'][j]
                
                # 添加动态分类特征
                for j, feat_name in enumerate(CATEGORICAL_DYNAMIC_FEATURES):
                    row[feat_name] = seq['categorical_dynamic'][i][j]
                
                # 添加目标值
                row['Daily fluxes'] = seq['targets'][i]
                
                rows.append(row)
        
        return pd.DataFrame(rows)


class N2ODatasetForObsStepRNN(Dataset):
    """观测步长RNN的数据集（每个观测点作为一个时间步）"""
    
    def __init__(
        self,
        base_data: BaseN2ODataset,
        fit_scalers: bool = True,
        scalers: dict | None = None
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
            all_targets.extend(seq['targets'])
            
            # 静态数值特征
            clay, cec, bd, ph, soc, tn = seq['numeric_static']
            seq_len = seq['seq_length']
            all_clay.extend([clay] * seq_len)
            all_cec.extend([cec] * seq_len)
            all_bd.extend([bd] * seq_len)
            all_ph.extend([ph] * seq_len)
            all_soc.extend([soc] * seq_len)
            all_tn.extend([tn] * seq_len)
            
            # 动态数值特征
            for i in range(seq_len):
                temp, prec, st, wfps, split_n, ferdur = seq['numeric_dynamic'][i]
                all_temp.append(temp)
                all_prec.append(prec)
                all_st.append(st)
                all_wfps.append(wfps)
                all_split_n.append(split_n)
                all_ferdur.append(ferdur)
        
        # Daily fluxes: Symlog + StandardScaler
        scalers['target_symlog'] = SymlogTransformer(C=1.0)
        target_symlog = scalers['target_symlog'].fit_transform(np.array(all_targets).reshape(-1, 1))
        scalers['target_scaler'] = StandardScaler()
        scalers['target_scaler'].fit(target_symlog)
        
        # Prec, Split N amount, ferdur: log(x+1) + StandardScaler
        prec_log = np.log1p(np.array(all_prec).reshape(-1, 1))
        scalers['prec_scaler'] = StandardScaler()
        scalers['prec_scaler'].fit(prec_log)
        
        split_n_log = np.log1p(np.array(all_split_n).reshape(-1, 1))
        scalers['split_n_scaler'] = StandardScaler()
        scalers['split_n_scaler'].fit(split_n_log)
        
        ferdur_log = np.log1p(np.array(all_ferdur).reshape(-1, 1))
        scalers['ferdur_scaler'] = StandardScaler()
        scalers['ferdur_scaler'].fit(ferdur_log)
        
        # 其他数值特征: StandardScaler
        scalers['temp_scaler'] = StandardScaler()
        scalers['temp_scaler'].fit(np.array(all_temp).reshape(-1, 1))
        
        scalers['st_scaler'] = StandardScaler()
        scalers['st_scaler'].fit(np.array(all_st).reshape(-1, 1))
        
        scalers['wfps_scaler'] = StandardScaler()
        scalers['wfps_scaler'].fit(np.array(all_wfps).reshape(-1, 1))
        
        scalers['clay_scaler'] = StandardScaler()
        scalers['clay_scaler'].fit(np.array(all_clay).reshape(-1, 1))
        
        scalers['cec_scaler'] = StandardScaler()
        scalers['cec_scaler'].fit(np.array(all_cec).reshape(-1, 1))
        
        scalers['bd_scaler'] = StandardScaler()
        scalers['bd_scaler'].fit(np.array(all_bd).reshape(-1, 1))
        
        scalers['ph_scaler'] = StandardScaler()
        scalers['ph_scaler'].fit(np.array(all_ph).reshape(-1, 1))
        
        scalers['soc_scaler'] = StandardScaler()
        scalers['soc_scaler'].fit(np.array(all_soc).reshape(-1, 1))
        
        scalers['tn_scaler'] = StandardScaler()
        scalers['tn_scaler'].fit(np.array(all_tn).reshape(-1, 1))
        
        return scalers
    
    def _apply_feature_engineering(self):
        """应用特征工程并保存处理后的数据"""
        self.processed_sequences = []
        
        for seq in self.sequences:
            seq_len = seq['seq_length']
            
            # 提取特征
            clay, cec, bd, ph, soc, tn = seq['numeric_static']
            sowdurs = np.array(seq['sowdurs'])
            
            # 计算 time_delta
            time_delta = np.zeros(seq_len)
            if seq_len > 1:
                time_delta[1:] = sowdurs[1:] - sowdurs[:-1]
            
            # 处理静态数值特征
            static_numeric = []
            static_numeric.append(self.scalers['clay_scaler'].transform([[clay]])[0, 0])
            static_numeric.append(self.scalers['cec_scaler'].transform([[cec]])[0, 0])
            static_numeric.append(self.scalers['bd_scaler'].transform([[bd]])[0, 0])
            static_numeric.append(self.scalers['ph_scaler'].transform([[ph]])[0, 0])
            static_numeric.append(self.scalers['soc_scaler'].transform([[soc]])[0, 0])
            static_numeric.append(self.scalers['tn_scaler'].transform([[tn]])[0, 0])
            
            # 处理动态数值特征
            dynamic_numeric = []
            for i in range(seq_len):
                temp, prec, st, wfps, split_n, ferdur = seq['numeric_dynamic'][i]
                
                feat = []
                feat.append(self.scalers['temp_scaler'].transform([[temp]])[0, 0])
                feat.append(self.scalers['prec_scaler'].transform([[np.log1p(prec)]])[0, 0])
                feat.append(self.scalers['st_scaler'].transform([[st]])[0, 0])
                feat.append(self.scalers['wfps_scaler'].transform([[wfps]])[0, 0])
                feat.append(self.scalers['split_n_scaler'].transform([[np.log1p(split_n)]])[0, 0])
                feat.append(self.scalers['ferdur_scaler'].transform([[np.log1p(ferdur)]])[0, 0])
                feat.append(time_delta[i])  # 添加 time_delta
                
                dynamic_numeric.append(feat)
            
            # 处理目标值
            targets = np.array(seq['targets']).reshape(-1, 1)
            targets_symlog = self.scalers['target_symlog'].transform(targets)
            targets_scaled = self.scalers['target_scaler'].transform(targets_symlog).flatten()
            
            processed_seq = {
                'seq_id': seq['seq_id'],
                'seq_length': seq_len,
                'static_numeric': np.array(static_numeric, dtype=np.float32),
                'dynamic_numeric': np.array(dynamic_numeric, dtype=np.float32),
                'static_categorical': np.array(seq['categorical_static'], dtype=np.int64),
                'dynamic_categorical': np.array(seq['categorical_dynamic'], dtype=np.int64),
                'targets': np.array(targets_scaled, dtype=np.float32),
                'targets_original': np.array(seq['targets'], dtype=np.float32),
            }
            
            self.processed_sequences.append(processed_seq)
    
    def __len__(self) -> int:
        return len(self.processed_sequences)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """返回一个序列的张量"""
        seq = self.processed_sequences[idx]
        return {
            'static_numeric': torch.from_numpy(seq['static_numeric']),
            'dynamic_numeric': torch.from_numpy(seq['dynamic_numeric']),
            'static_categorical': torch.from_numpy(seq['static_categorical']),
            'dynamic_categorical': torch.from_numpy(seq['dynamic_categorical']),
            'targets': torch.from_numpy(seq['targets']),
            'targets_original': torch.from_numpy(seq['targets_original']),
            'seq_length': seq['seq_length'],
        }
    
    def inverse_transform_targets(self, targets_scaled: np.ndarray) -> np.ndarray:
        """将缩放后的目标值转换回原始空间"""
        targets_symlog = self.scalers['target_scaler'].inverse_transform(targets_scaled.reshape(-1, 1))
        targets_original = self.scalers['target_symlog'].inverse_transform(targets_symlog)
        return targets_original.flatten()


class N2ODatasetForDailyStepRNN(Dataset):
    """每日步长RNN的数据集（每天作为一个时间步）"""
    
    def __init__(
        self,
        base_data: BaseN2ODataset,
        fit_scalers: bool = True,
        scalers: dict | None = None
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
            sowdurs = np.array(seq['sowdurs'])
            seq_len = seq['seq_length']
            
            # 确定序列的总天数范围
            start_day = int(sowdurs[0])
            end_day = int(sowdurs[-1])
            total_days = end_day - start_day + 1
            
            # 初始化每日数据
            daily_data = {
                'seq_id': seq['seq_id'],
                'start_day': start_day,
                'total_days': total_days,
                'numeric_static': seq['numeric_static'],
                'categorical_static': seq['categorical_static'],
                'daily_numeric': [],
                'daily_categorical': [],
                'daily_targets': [],
                'mask': [],  # True表示真实测量点，False表示插值点
            }
            
            # 创建观测索引映射
            obs_map = {int(sowdurs[i]): i for i in range(seq_len)}
            
            # 提取原始动态数据用于插值
            temp_vals = [seq['numeric_dynamic'][i][0] for i in range(seq_len)]
            st_vals = [seq['numeric_dynamic'][i][2] for i in range(seq_len)]
            wfps_vals = [seq['numeric_dynamic'][i][3] for i in range(seq_len)]
            
            # 用于记录施肥信息
            last_fert_day = -1
            last_fert_amount = 0
            
            for day in range(start_day, end_day + 1):
                if day in obs_map:
                    # 真实测量点
                    idx = obs_map[day]
                    daily_numeric = list(seq['numeric_dynamic'][idx])
                    daily_categorical = list(seq['categorical_dynamic'][idx])
                    target = seq['targets'][idx]
                    
                    # 更新施肥信息
                    if daily_numeric[4] > 0:  # Split N amount
                        last_fert_day = day
                        last_fert_amount = daily_numeric[4]
                    
                    daily_data['daily_numeric'].append(daily_numeric)
                    daily_data['daily_categorical'].append(daily_categorical)
                    daily_data['daily_targets'].append(target)
                    daily_data['mask'].append(True)
                else:
                    # 插值点
                    # 线性插值数值特征（除了Prec和Split N amount）
                    day_idx = day - start_day
                    
                    # 找到前后最近的观测点用于插值
                    before_days = [d for d in obs_map.keys() if d < day]
                    after_days = [d for d in obs_map.keys() if d > day]
                    
                    if before_days and after_days:
                        before_day = max(before_days)
                        after_day = min(after_days)
                        before_idx = obs_map[before_day]
                        after_idx = obs_map[after_day]
                        
                        # 插值比例
                        alpha = (day - before_day) / (after_day - before_day)
                        
                        # Temp, ST, WFPS 线性插值
                        temp = seq['numeric_dynamic'][before_idx][0] * (1 - alpha) + \
                               seq['numeric_dynamic'][after_idx][0] * alpha
                        st = seq['numeric_dynamic'][before_idx][2] * (1 - alpha) + \
                             seq['numeric_dynamic'][after_idx][2] * alpha
                        wfps = seq['numeric_dynamic'][before_idx][3] * (1 - alpha) + \
                               seq['numeric_dynamic'][after_idx][3] * alpha
                    elif before_days:
                        # 只有之前的点，前向填充
                        before_day = max(before_days)
                        before_idx = obs_map[before_day]
                        temp = seq['numeric_dynamic'][before_idx][0]
                        st = seq['numeric_dynamic'][before_idx][2]
                        wfps = seq['numeric_dynamic'][before_idx][3]
                    else:
                        # 只有之后的点，后向填充
                        after_day = min(after_days)
                        after_idx = obs_map[after_day]
                        temp = seq['numeric_dynamic'][after_idx][0]
                        st = seq['numeric_dynamic'][after_idx][2]
                        wfps = seq['numeric_dynamic'][after_idx][3]
                    
                    # Prec填充为0（非观测日无降水记录）
                    prec = 0.0
                    
                    # Split N amount: 只在施肥当天非0
                    split_n = 0.0
                    
                    # ferdur: 距离上次施肥的天数
                    if last_fert_day >= 0:
                        ferdur = day - last_fert_day
                    else:
                        ferdur = 0.0
                    
                    daily_numeric = [temp, prec, st, wfps, split_n, ferdur]
                    
                    # 分类特征前向填充
                    if before_days:
                        before_day = max(before_days)
                        before_idx = obs_map[before_day]
                        daily_categorical = list(seq['categorical_dynamic'][before_idx])
                    else:
                        # 后向填充
                        after_day = min(after_days)
                        after_idx = obs_map[after_day]
                        daily_categorical = list(seq['categorical_dynamic'][after_idx])
                    
                    # 目标值插值（但不会用于损失计算）
                    if before_days and after_days:
                        before_day = max(before_days)
                        after_day = min(after_days)
                        before_idx = obs_map[before_day]
                        after_idx = obs_map[after_day]
                        alpha = (day - before_day) / (after_day - before_day)
                        target = seq['targets'][before_idx] * (1 - alpha) + \
                                seq['targets'][after_idx] * alpha
                    elif before_days:
                        before_day = max(before_days)
                        before_idx = obs_map[before_day]
                        target = seq['targets'][before_idx]
                    else:
                        after_day = min(after_days)
                        after_idx = obs_map[after_day]
                        target = seq['targets'][after_idx]
                    
                    daily_data['daily_numeric'].append(daily_numeric)
                    daily_data['daily_categorical'].append(daily_categorical)
                    daily_data['daily_targets'].append(target)
                    daily_data['mask'].append(False)
            
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
            clay, cec, bd, ph, soc, tn = seq['numeric_static']
            
            for i in range(seq['total_days']):
                if seq['mask'][i]:  # 只收集真实测量点
                    all_targets.append(seq['daily_targets'][i])
                    
                    # 静态特征
                    all_clay.append(clay)
                    all_cec.append(cec)
                    all_bd.append(bd)
                    all_ph.append(ph)
                    all_soc.append(soc)
                    all_tn.append(tn)
                    
                    # 动态特征
                    temp, prec, st, wfps, split_n, ferdur = seq['daily_numeric'][i]
                    all_temp.append(temp)
                    all_prec.append(prec)
                    all_st.append(st)
                    all_wfps.append(wfps)
                    all_split_n.append(split_n)
                    all_ferdur.append(ferdur)
        
        # Daily fluxes: Symlog + StandardScaler
        scalers['target_symlog'] = SymlogTransformer(C=1.0)
        target_symlog = scalers['target_symlog'].fit_transform(np.array(all_targets).reshape(-1, 1))
        scalers['target_scaler'] = StandardScaler()
        scalers['target_scaler'].fit(target_symlog)
        
        # Prec, Split N amount, ferdur: log(x+1) + StandardScaler
        prec_log = np.log1p(np.array(all_prec).reshape(-1, 1))
        scalers['prec_scaler'] = StandardScaler()
        scalers['prec_scaler'].fit(prec_log)
        
        split_n_log = np.log1p(np.array(all_split_n).reshape(-1, 1))
        scalers['split_n_scaler'] = StandardScaler()
        scalers['split_n_scaler'].fit(split_n_log)
        
        ferdur_log = np.log1p(np.array(all_ferdur).reshape(-1, 1))
        scalers['ferdur_scaler'] = StandardScaler()
        scalers['ferdur_scaler'].fit(ferdur_log)
        
        # 其他数值特征: StandardScaler
        scalers['temp_scaler'] = StandardScaler()
        scalers['temp_scaler'].fit(np.array(all_temp).reshape(-1, 1))
        
        scalers['st_scaler'] = StandardScaler()
        scalers['st_scaler'].fit(np.array(all_st).reshape(-1, 1))
        
        scalers['wfps_scaler'] = StandardScaler()
        scalers['wfps_scaler'].fit(np.array(all_wfps).reshape(-1, 1))
        
        scalers['clay_scaler'] = StandardScaler()
        scalers['clay_scaler'].fit(np.array(all_clay).reshape(-1, 1))
        
        scalers['cec_scaler'] = StandardScaler()
        scalers['cec_scaler'].fit(np.array(all_cec).reshape(-1, 1))
        
        scalers['bd_scaler'] = StandardScaler()
        scalers['bd_scaler'].fit(np.array(all_bd).reshape(-1, 1))
        
        scalers['ph_scaler'] = StandardScaler()
        scalers['ph_scaler'].fit(np.array(all_ph).reshape(-1, 1))
        
        scalers['soc_scaler'] = StandardScaler()
        scalers['soc_scaler'].fit(np.array(all_soc).reshape(-1, 1))
        
        scalers['tn_scaler'] = StandardScaler()
        scalers['tn_scaler'].fit(np.array(all_tn).reshape(-1, 1))
        
        return scalers
    
    def _apply_feature_engineering(self):
        """应用特征工程"""
        self.processed_sequences = []
        
        for seq in self.daily_sequences:
            # 处理静态数值特征
            clay, cec, bd, ph, soc, tn = seq['numeric_static']
            static_numeric = []
            static_numeric.append(self.scalers['clay_scaler'].transform([[clay]])[0, 0])
            static_numeric.append(self.scalers['cec_scaler'].transform([[cec]])[0, 0])
            static_numeric.append(self.scalers['bd_scaler'].transform([[bd]])[0, 0])
            static_numeric.append(self.scalers['ph_scaler'].transform([[ph]])[0, 0])
            static_numeric.append(self.scalers['soc_scaler'].transform([[soc]])[0, 0])
            static_numeric.append(self.scalers['tn_scaler'].transform([[tn]])[0, 0])
            
            # 处理动态数值特征
            dynamic_numeric = []
            for i in range(seq['total_days']):
                temp, prec, st, wfps, split_n, ferdur = seq['daily_numeric'][i]
                
                feat = []
                feat.append(self.scalers['temp_scaler'].transform([[temp]])[0, 0])
                feat.append(self.scalers['prec_scaler'].transform([[np.log1p(prec)]])[0, 0])
                feat.append(self.scalers['st_scaler'].transform([[st]])[0, 0])
                feat.append(self.scalers['wfps_scaler'].transform([[wfps]])[0, 0])
                feat.append(self.scalers['split_n_scaler'].transform([[np.log1p(split_n)]])[0, 0])
                feat.append(self.scalers['ferdur_scaler'].transform([[np.log1p(ferdur)]])[0, 0])
                
                dynamic_numeric.append(feat)
            
            # 处理目标值
            targets = np.array(seq['daily_targets']).reshape(-1, 1)
            targets_symlog = self.scalers['target_symlog'].transform(targets)
            targets_scaled = self.scalers['target_scaler'].transform(targets_symlog).flatten()
            
            processed_seq = {
                'seq_id': seq['seq_id'],
                'seq_length': seq['total_days'],
                'static_numeric': np.array(static_numeric, dtype=np.float32),
                'dynamic_numeric': np.array(dynamic_numeric, dtype=np.float32),
                'static_categorical': np.array(seq['categorical_static'], dtype=np.int64),
                'dynamic_categorical': np.array(seq['daily_categorical'], dtype=np.int64),
                'targets': np.array(targets_scaled, dtype=np.float32),
                'targets_original': np.array(seq['daily_targets'], dtype=np.float32),
                'mask': np.array(seq['mask'], dtype=bool),
            }
            
            self.processed_sequences.append(processed_seq)
    
    def __len__(self) -> int:
        return len(self.processed_sequences)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """返回一个序列的张量"""
        seq = self.processed_sequences[idx]
        return {
            'static_numeric': torch.from_numpy(seq['static_numeric']),
            'dynamic_numeric': torch.from_numpy(seq['dynamic_numeric']),
            'static_categorical': torch.from_numpy(seq['static_categorical']),
            'dynamic_categorical': torch.from_numpy(seq['dynamic_categorical']),
            'targets': torch.from_numpy(seq['targets']),
            'targets_original': torch.from_numpy(seq['targets_original']),
            'mask': torch.from_numpy(seq['mask']),
            'seq_length': seq['seq_length'],
        }
    
    def inverse_transform_targets(self, targets_scaled: np.ndarray) -> np.ndarray:
        """将缩放后的目标值转换回原始空间"""
        targets_symlog = self.scalers['target_scaler'].inverse_transform(targets_scaled.reshape(-1, 1))
        targets_original = self.scalers['target_symlog'].inverse_transform(targets_symlog)
        return targets_original.flatten()


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """
    自定义collate函数，用于处理变长序列
    """
    # 获取批次中的最大序列长度
    max_len = max(item['seq_length'] for item in batch)
    batch_size = len(batch)
    
    # 获取特征维度
    num_static_numeric = batch[0]['static_numeric'].shape[0]
    num_dynamic_numeric = batch[0]['dynamic_numeric'].shape[1]
    num_static_categorical = batch[0]['static_categorical'].shape[0]
    num_dynamic_categorical = batch[0]['dynamic_categorical'].shape[1]
    
    # 初始化填充后的张量
    static_numeric = torch.zeros(batch_size, num_static_numeric)
    dynamic_numeric = torch.zeros(batch_size, max_len, num_dynamic_numeric)
    static_categorical = torch.zeros(batch_size, num_static_categorical, dtype=torch.long)
    dynamic_categorical = torch.zeros(batch_size, max_len, num_dynamic_categorical, dtype=torch.long)
    targets = torch.zeros(batch_size, max_len)
    targets_original = torch.zeros(batch_size, max_len)
    seq_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    # 检查是否有mask（DailyStepRNN才有）
    has_mask = 'mask' in batch[0]
    if has_mask:
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    # 填充数据
    for i, item in enumerate(batch):
        seq_len = item['seq_length']
        seq_lengths[i] = seq_len
        
        static_numeric[i] = item['static_numeric']
        dynamic_numeric[i, :seq_len] = item['dynamic_numeric']
        static_categorical[i] = item['static_categorical']
        dynamic_categorical[i, :seq_len] = item['dynamic_categorical']
        targets[i, :seq_len] = item['targets']
        targets_original[i, :seq_len] = item['targets_original']
        
        if has_mask:
            mask[i, :seq_len] = item['mask']
    
    result = {
        'static_numeric': static_numeric,
        'dynamic_numeric': dynamic_numeric,
        'static_categorical': static_categorical,
        'dynamic_categorical': dynamic_categorical,
        'targets': targets,
        'targets_original': targets_original,
        'seq_lengths': seq_lengths,
    }
    
    if has_mask:
        result['mask'] = mask
    
    return result


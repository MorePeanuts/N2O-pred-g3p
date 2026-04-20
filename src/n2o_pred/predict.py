"""
预测工具模块
"""

import pickle
import time
from pathlib import Path
from typing import Any
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import (
    BaseN2ODataset,
    N2ODatasetForDailyStepRNN,
    N2ODatasetForObsStepRNN,
    TifDataLoader,
    collate_fn,
)
from .evaluation import compute_metrics, save_predictions_to_csv
from .rf import N2OPredictorRF
from .rnn import N2OPredictorRNN
from .utils import create_logger, load_json

logger = create_logger(__name__)


class N2OPredictor:
    """N2O排放预测器（统一接口）"""

    def __init__(self, model_dir: Path | str):
        """
        Args:
            model_dir: 模型目录（包含模型文件和配置）
        """
        self.model_dir = Path(model_dir)

        if not self.model_dir.exists():
            raise FileNotFoundError(f'模型目录不存在: {self.model_dir}')

        # 加载配置
        config_path = self.model_dir / 'config.json'
        if not config_path.exists():
            raise FileNotFoundError(f'配置文件不存在: {config_path}')

        self.config = load_json(config_path)

        # 确定模型类型（从父目录的summary.json获取）
        summary_path = self.model_dir.parent / 'summary.json'
        if summary_path.exists():
            summary = load_json(summary_path)
            self.model_type = summary['model_type']
        else:
            # 尝试从配置推断
            if 'rnn_type' in self.config:
                # 需要额外信息确定是obs还是daily
                logger.warning('无法从summary.json确定模型类型，请手动指定')
                self.model_type = 'rnn-obs'  # 默认
            else:
                self.model_type = 'rf'

        logger.info(f'加载模型类型: {self.model_type}')

        # 加载模型
        self.model = self._load_model()

        # 加载预处理器（RNN需要）
        if self.model_type.startswith('rnn'):
            scalers_path = self.model_dir / 'scalers.pkl'
            if scalers_path.exists():
                with open(scalers_path, 'rb') as f:
                    self.scalers = pickle.load(f)
            else:
                logger.warning('未找到scalers.pkl，预测可能失败')
                self.scalers = None

    def _load_model(self) -> Any:
        """加载模型"""
        if self.model_type == 'rf':
            model_path = self.model_dir / 'model.pkl'
            if not model_path.exists():
                raise FileNotFoundError(f'模型文件不存在: {model_path}')
            return N2OPredictorRF.load(model_path)

        else:  # RNN模型
            model_path = self.model_dir / 'best_model.pt'
            if not model_path.exists():
                # 尝试另一个可能的路径
                model_path = self.model_dir / 'model.pt'
                if not model_path.exists():
                    raise FileNotFoundError(f'模型文件不存在')

            # 加载编码器以获取cardinality
            from .preprocessing import (
                CATEGORICAL_STATIC_FEATURES,
                CATEGORICAL_DYNAMIC_FEATURES,
            )

            encoders_path = Path(__file__).parents[2] / 'preprocessor' / 'encoders.pkl'
            with open(encoders_path, 'rb') as f:
                encoders = pickle.load(f)

            categorical_static_cardinalities = [
                len(encoders[feat].classes_) for feat in CATEGORICAL_STATIC_FEATURES
            ]
            categorical_dynamic_cardinalities = [
                len(encoders[feat].classes_) for feat in CATEGORICAL_DYNAMIC_FEATURES
            ]

            # 确定动态特征数量
            if self.model_type == 'rnn-obs':
                num_dynamic_numeric = 7  # 包含time_delta
            else:
                num_dynamic_numeric = 6

            # 创建模型
            model = N2OPredictorRNN(
                num_numeric_static=6,
                num_numeric_dynamic=num_dynamic_numeric,
                categorical_static_cardinalities=categorical_static_cardinalities,
                categorical_dynamic_cardinalities=categorical_dynamic_cardinalities,
                embedding_dim=self.config.get('embedding_dim', 8),
                hidden_size=self.config.get('hidden_size', 96),
                num_layers=self.config.get('num_layers', 2),
                rnn_type=self.config.get('rnn_type', 'GRU'),
                dropout=self.config.get('dropout', 0.2),
            )

            # 加载权重
            if str(model_path).endswith('best_model.pt'):
                checkpoint = torch.load(model_path, map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(torch.load(model_path, map_location='cpu'))

            model.eval()
            return model

    def predict(
        self,
        data: BaseN2ODataset | pd.DataFrame,
        device: str = 'cpu',
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """
        在新数据上进行预测

        Args:
            data: 输入数据（BaseN2ODataset或DataFrame）
            device: 设备
            batch_size: 批次大小（RNN使用）

        Returns:
            预测结果字典
        """
        if self.model_type == 'rf':
            return self._predict_rf(data)
        elif self.model_type == 'rnn-obs':
            return self._predict_rnn_obs(data, device, batch_size)
        elif self.model_type == 'rnn-daily':
            return self._predict_rnn_daily(data, device, batch_size)
        else:
            raise ValueError(f'不支持的模型类型: {self.model_type}')

    def _predict_rf(self, data: pd.DataFrame | BaseN2ODataset) -> dict[str, Any]:
        """随机森林预测"""
        is_base_dataset = isinstance(data, BaseN2ODataset)

        if is_base_dataset:
            # 如果是BaseN2ODataset，展开为DataFrame（使用RF专用方法）
            data_df = data.flatten_to_dataframe_for_rf()
        else:
            data_df = data

        predictions = self.model.predict(data_df)

        from .preprocessing import LABELS

        # 检查是否有标签
        has_labels = LABELS[0] in data_df.columns

        if has_labels:
            targets = data_df[LABELS[0]].values
            metrics = compute_metrics(targets, predictions)
        else:
            targets = None
            metrics = None

        # 添加预测值到DataFrame
        data_df_with_pred = data_df.copy()
        data_df_with_pred['predicted_daily_fluxes'] = predictions

        # 如果输入是BaseN2ODataset，转换回序列格式并添加预测字段
        if is_base_dataset:
            predicted_dataset = BaseN2ODataset.from_dataframe(data_df_with_pred)
            # 为每个序列添加预测值
            for i, seq in enumerate(predicted_dataset.sequences):
                seq['predicted_targets'] = seq['targets']  # 重命名原来的targets为predicted_targets
                # 从DataFrame中提取该序列的预测值
                seq_pred = data_df_with_pred[
                    (data_df_with_pred['Publication'] == seq['seq_id'][0])
                    & (data_df_with_pred['control_group'] == seq['seq_id'][1])
                ]['predicted_daily_fluxes'].values
                seq['predicted_targets'] = list(seq_pred)
        else:
            predicted_dataset = data_df_with_pred

        return {
            'predictions': predictions,
            'targets': targets,
            'metrics': metrics,
            'data_with_predictions': predicted_dataset,
        }

    def _predict_rnn_obs(
        self, data: BaseN2ODataset, device: str, batch_size: int
    ) -> dict[str, Any]:
        """观测步长RNN预测"""
        if not isinstance(data, BaseN2ODataset):
            raise TypeError('RNN模型需要BaseN2ODataset格式的数据')

        # 创建数据集
        dataset = N2ODatasetForObsStepRNN(data, fit_scalers=False, scalers=self.scalers)

        # 创建数据加载器
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # 预测
        device = torch.device(device)
        self.model = self.model.to(device)
        self.model.eval()

        all_predictions = []
        all_targets = []
        predictions_by_seq = []  # 按序列组织的预测结果

        with torch.no_grad():
            for batch in loader:
                static_numeric = batch['static_numeric'].to(device)
                dynamic_numeric = batch['dynamic_numeric'].to(device)
                static_categorical = batch['static_categorical'].to(device)
                dynamic_categorical = batch['dynamic_categorical'].to(device)
                seq_lengths = batch['seq_lengths'].to(device)
                targets_original = batch['targets_original']

                predictions = self.model(
                    static_numeric,
                    static_categorical,
                    dynamic_numeric,
                    dynamic_categorical,
                    seq_lengths,
                )

                predictions_np = predictions.cpu().numpy()

                for i in range(len(seq_lengths)):
                    seq_len = seq_lengths[i].item()
                    pred_scaled = predictions_np[i, :seq_len]
                    target_orig = targets_original[i, :seq_len].numpy()

                    # 逆转换
                    pred_orig = dataset.inverse_transform_targets(pred_scaled)

                    all_predictions.extend(pred_orig)
                    all_targets.extend(target_orig)
                    predictions_by_seq.append(list(pred_orig))

        predictions = np.array(all_predictions)
        targets = np.array(all_targets)

        # 检查是否有有效的标签
        has_labels = not np.all(targets == 0)
        if has_labels:
            metrics = compute_metrics(targets, predictions)
        else:
            metrics = None

        # 添加预测值到原始数据集
        predicted_dataset = BaseN2ODataset(sequences=[])
        for i, seq in enumerate(data.sequences):
            new_seq = seq.copy()
            new_seq['predicted_targets'] = predictions_by_seq[i]
            predicted_dataset.sequences.append(new_seq)

        return {
            'predictions': predictions,
            'targets': targets if has_labels else None,
            'metrics': metrics,
            'data_with_predictions': predicted_dataset,
        }

    def _predict_rnn_daily(
        self, data: BaseN2ODataset, device: str, batch_size: int
    ) -> dict[str, Any]:
        """每日步长RNN预测"""
        if not isinstance(data, BaseN2ODataset):
            raise TypeError('RNN模型需要BaseN2ODataset格式的数据')

        # 创建数据集
        dataset = N2ODatasetForDailyStepRNN(data, fit_scalers=False, scalers=self.scalers)

        # 创建数据加载器
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # 预测
        device = torch.device(device)
        self.model = self.model.to(device)
        self.model.eval()

        all_predictions = []  # 所有时间点的预测（用于与flatten_to_dataframe对齐）
        all_targets = []  # 所有时间点的targets（非测量点用NaN填充）
        all_predictions_masked = []  # 仅真实测量点（用于计算metrics）
        all_targets_masked = []
        predictions_by_seq = []  # 按序列组织的预测结果（只包含真实测量点）
        all_masks = []  # 记录每个点是否是真实测量点

        with torch.no_grad():
            for batch in loader:
                static_numeric = batch['static_numeric'].to(device)
                dynamic_numeric = batch['dynamic_numeric'].to(device)
                static_categorical = batch['static_categorical'].to(device)
                dynamic_categorical = batch['dynamic_categorical'].to(device)
                seq_lengths = batch['seq_lengths'].to(device)
                targets_original = batch['targets_original']
                mask = batch['mask']

                predictions = self.model(
                    static_numeric,
                    static_categorical,
                    dynamic_numeric,
                    dynamic_categorical,
                    seq_lengths,
                )

                predictions_np = predictions.cpu().numpy()

                for i in range(len(seq_lengths)):
                    seq_len = seq_lengths[i].item()
                    pred_scaled = predictions_np[i, :seq_len]
                    target_orig = targets_original[i, :seq_len].numpy()
                    mask_i = mask[i, :seq_len].numpy()

                    # 逆转换
                    pred_orig = dataset.inverse_transform_targets(pred_scaled)

                    # 保存所有时间点的预测和targets（与flatten_to_dataframe对齐）
                    all_predictions.extend(pred_orig)
                    # 非测量点的target用NaN填充
                    target_full = np.where(mask_i, target_orig, np.nan)
                    all_targets.extend(target_full)
                    all_masks.extend(mask_i)

                    # 只保留真实测量点用于计算metrics
                    all_predictions_masked.extend(pred_orig[mask_i])
                    all_targets_masked.extend(target_orig[mask_i])
                    predictions_by_seq.append(list(pred_orig[mask_i]))

        predictions = np.array(all_predictions)
        targets_full = np.array(all_targets)
        predictions_masked = np.array(all_predictions_masked)
        targets_masked = np.array(all_targets_masked)
        masks = np.array(all_masks)

        # 检查是否有有效的标签
        has_labels = not np.all(targets_masked == 0)
        if has_labels:
            metrics = compute_metrics(targets_masked, predictions_masked)
        else:
            metrics = None

        # 添加预测值到原始数据集
        predicted_dataset = BaseN2ODataset(sequences=[])
        for i, seq in enumerate(data.sequences):
            new_seq = seq.copy()
            new_seq['predicted_targets'] = predictions_by_seq[i]
            predicted_dataset.sequences.append(new_seq)

        return {
            'predictions': predictions,
            'targets': targets_full if has_labels else None,
            'targets_masked': targets_masked if has_labels else None,
            'metrics': metrics,
            'data_with_predictions': predicted_dataset,
            'masks': masks,  # 可选：返回掩码供后续使用
        }


def predict_with_model(
    model_dir: Path | str,
    data_path: Path | str,
    output_path: Path | str | None = None,
    device: str = 'cpu',
    plot_sequences: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """
    使用训练好的模型进行在对应的数据集划分上进行预测的便捷函数

    Args:
        model_dir: 模型目录（一定是`split_xxx`的格式，其中xxx是训练模型时进行data split的随机种子）
        data_path: 数据路径
        output_path: 输出路径（保存带预测结果的数据，包括feature_importance.csv, test_predictions.csv, train_predictions.csv, val_predictions.csv）
        device: 设备
        plot_sequences: 需要绘制预测图的序列ID列表，每个元素是(publication, control_group)元组

    Returns:
        预测结果
    """
    model_dir = Path(model_dir)
    data_path = Path(data_path)

    if output_path is None:
        output_path = model_dir
    else:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    # 从目录名中提取seed
    split_dir_name = model_dir.name
    if not split_dir_name.startswith('split_'):
        raise ValueError(f'模型目录名必须以split_开头，当前: {split_dir_name}')
    seed = int(split_dir_name.split('_')[1])
    logger.info(f'从模型目录提取seed: {seed}')

    # 从summary.json获取模型类型
    summary_path = model_dir.parent / 'summary.json'
    if summary_path.exists():
        summary = load_json(summary_path)
        model_type = summary['model_type']
    else:
        # 尝试从配置推断
        config_path = model_dir / 'config.json'
        config = load_json(config_path)
        if 'rnn_type' in config:
            logger.warning('无法从summary.json确定模型类型，默认使用rnn-obs')
            model_type = 'rnn-obs'
        else:
            model_type = 'rf'

    logger.info(f'模型类型: {model_type}')

    # 加载数据
    logger.info(f'从 {data_path} 加载数据...')
    with open(data_path, 'rb') as f:
        sequences = pickle.load(f)
    base_dataset = BaseN2ODataset(sequences=sequences)

    # 使用相同的seed进行数据划分
    from sklearn.model_selection import train_test_split as sklearn_split

    n_sequences = len(base_dataset)
    indices = list(range(n_sequences))

    train_split = 0.8  # 默认值，与训练时保持一致
    test_ratio = (1.0 - train_split) / 2
    train_val_indices, test_indices = sklearn_split(
        indices, train_size=1.0 - test_ratio, random_state=seed
    )

    val_ratio = test_ratio / (1.0 - test_ratio)
    train_indices, val_indices = sklearn_split(
        train_val_indices, train_size=1.0 - val_ratio, random_state=seed
    )

    logger.info(
        f'数据集划分: {len(train_indices)} 训练, {len(val_indices)} 验证, {len(test_indices)} 测试'
    )

    train_base = base_dataset[train_indices]
    val_base = base_dataset[val_indices]
    test_base = base_dataset[test_indices]

    # 加载预测器
    predictor = N2OPredictor(model_dir)

    # 创建输出目录
    tables_dir = output_path / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 总是创建figs目录，用于保存特征重要性图
    figs_dir = output_path / 'figs'
    figs_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'model_type': model_type,
        'seed': seed,
    }

    # 导入需要的函数
    from .evaluation import save_predictions_to_csv, compute_shap_values, plot_feature_importance

    if model_type == 'rf':
        # 随机森林预测 - 传入BaseN2ODataset而不是DataFrame，这样返回的data_with_predictions会有sequences属性
        train_result = predictor.predict(train_base, device=device)
        val_result = predictor.predict(val_base, device=device)
        test_result = predictor.predict(test_base, device=device)

        # 仍然需要DataFrame用于保存CSV和SHAP分析
        train_df = train_base.flatten_to_dataframe_for_rf()
        val_df = val_base.flatten_to_dataframe_for_rf()
        test_df = test_base.flatten_to_dataframe_for_rf()

        # 保存预测结果
        save_predictions_to_csv(
            train_result['predictions'],
            train_result['targets'],
            tables_dir / 'train_predictions.csv',
            additional_cols={
                'No. of obs': train_df['No. of obs'].values,
                'Publication': train_df['Publication'].values,
                'control_group': train_df['control_group'].values,
                'sowdur': train_df['sowdur'].values,
            },
        )
        save_predictions_to_csv(
            val_result['predictions'],
            val_result['targets'],
            tables_dir / 'val_predictions.csv',
            additional_cols={
                'No. of obs': val_df['No. of obs'].values,
                'Publication': val_df['Publication'].values,
                'control_group': val_df['control_group'].values,
                'sowdur': val_df['sowdur'].values,
            },
        )
        save_predictions_to_csv(
            test_result['predictions'],
            test_result['targets'],
            tables_dir / 'test_predictions.csv',
            additional_cols={
                'No. of obs': test_df['No. of obs'].values,
                'Publication': test_df['Publication'].values,
                'control_group': test_df['control_group'].values,
                'sowdur': test_df['sowdur'].values,
            },
        )

        # 使用compute_shap_values计算特征重要性（在所有数据集上）
        try:
            logger.info("计算RF模型的特征重要性（使用SHAP）...")
            # 合并所有数据集
            all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            shap_values, feature_names = compute_shap_values(
                predictor.model, all_df, model_type, device, max_samples=len(all_df),
                background_size=100, n_explain=200, nsamples=40
            )
            # 保存特征重要性CSV
            importance_df = pd.DataFrame(
                {
                    'feature': feature_names,
                    'importance': shap_values,
                }
            ).sort_values('importance', ascending=False)
            importance_df.to_csv(tables_dir / 'feature_importance.csv', index=False)
            # 保存特征重要性图
            plot_feature_importance(feature_names, shap_values, figs_dir / 'feature_importance.png')
        except Exception as e:
            logger.warning(f'SHAP分析失败: {e}')

        # 绘制序列预测图
        if plot_sequences and figs_dir is not None:
            from .evaluation import plot_sequence_predictions

            # 合并所有数据进行查找
            all_bases = [
                ('train', train_base, train_result),
                ('val', val_base, val_result),
                ('test', test_base, test_result),
            ]

            # 打印前几个序列ID帮助调试
            if len(train_base.sequences) > 0:
                sample_seq = train_base.sequences[0]
                sample_id = sample_seq['seq_id']
                logger.debug(
                    f'数据集中的序列ID示例: {sample_id}, 类型: ({type(sample_id[0]), type(sample_id[1])})'
                )

            for seq_id_tuple in plot_sequences:
                pub_arg, cg_arg = seq_id_tuple
                found = False

                for split_name, base, result in all_bases:
                    # 在该数据集中查找序列
                    for seq_idx, seq in enumerate(base.sequences):
                        seq_pub, seq_cg = seq['seq_id']

                        # 尝试类型转换来匹配
                        pub_match = False
                        cg_match = False

                        # 尝试直接比较
                        if seq_pub == pub_arg and seq_cg == cg_arg:
                            pub_match = True
                            cg_match = True
                        else:
                            # 尝试将参数转换为序列ID的类型
                            try:
                                pub_converted = type(seq_pub)(pub_arg)
                                cg_converted = type(seq_cg)(cg_arg)
                                if seq_pub == pub_converted and seq_cg == cg_converted:
                                    pub_match = True
                                    cg_match = True
                            except (ValueError, TypeError):
                                pass

                        if pub_match and cg_match:
                            # 找到序列，现在需要获取预测值
                            time_steps = np.array(seq['sowdurs'])
                            targets = np.array(seq['targets'])

                            # 从data_with_predictions中获取预测值
                            data_with_preds = result['data_with_predictions']
                            pred_seq = None
                            for s in data_with_preds.sequences:
                                s_pub, s_cg = s['seq_id']
                                # 同样尝试类型转换
                                if s_pub == seq_pub and s_cg == seq_cg:
                                    pred_seq = s
                                    break
                                else:
                                    try:
                                        if (
                                            type(s_pub)(seq_pub) == s_pub
                                            and type(s_cg)(seq_cg) == s_cg
                                        ):
                                            pred_seq = s
                                            break
                                    except (ValueError, TypeError):
                                        pass

                            if pred_seq is not None and 'predicted_targets' in pred_seq:
                                predictions = np.array(pred_seq['predicted_targets'])

                                # 使用原始序列ID作为文件名
                                filename_pub = str(seq_pub).replace('/', '_').replace('\\', '_')
                                filename_cg = str(seq_cg).replace('/', '_').replace('\\', '_')

                                plot_sequence_predictions(
                                    (seq_pub, seq_cg),
                                    time_steps,
                                    targets,
                                    predictions,
                                    figs_dir
                                    / f'sequence_predictions_{filename_pub}_{filename_cg}.png',
                                    mask=None,
                                )
                                logger.info(
                                    f'已绘制序列预测图: {seq_pub}_{seq_cg} ({split_name}集)'
                                )
                                found = True
                                break
                    if found:
                        break

                if not found:
                    # 列出所有可用的序列ID帮助调试
                    available_ids = []
                    for split_name, base, _ in all_bases:
                        for seq in base.sequences[:10]:  # 只显示前10个
                            available_ids.append(f'{seq["seq_id"]}')
                    logger.warning(
                        f'未找到序列: {pub_arg}_{cg_arg}。部分可用序列ID: {", ".join(available_ids)}...'
                    )

        results.update(
            {
                'train_metrics': train_result['metrics'],
                'val_metrics': val_result['metrics'],
                'test_metrics': test_result['metrics'],
            }
        )

    else:
        # RNN模型预测
        from .dataset import N2ODatasetForObsStepRNN, N2ODatasetForDailyStepRNN, collate_fn
        from torch.utils.data import DataLoader

        use_mask = model_type == 'rnn-daily'

        # 准备数据集（需要创建dataset用于SHAP分析
        if model_type == 'rnn-obs':
            train_dataset = N2ODatasetForObsStepRNN(
                train_base, fit_scalers=False, scalers=predictor.scalers
            )
            val_dataset = N2ODatasetForObsStepRNN(
                val_base, fit_scalers=False, scalers=predictor.scalers
            )
            test_dataset = N2ODatasetForObsStepRNN(
                test_base, fit_scalers=False, scalers=predictor.scalers
            )
        else:
            train_dataset = N2ODatasetForDailyStepRNN(
                train_base, fit_scalers=False, scalers=predictor.scalers
            )
            val_dataset = N2ODatasetForDailyStepRNN(
                val_base, fit_scalers=False, scalers=predictor.scalers
            )
            test_dataset = N2ODatasetForDailyStepRNN(
                test_base, fit_scalers=False, scalers=predictor.scalers
            )

        # 预测并获取完整结果
        train_result = predictor.predict(train_base, device=device)
        val_result = predictor.predict(val_base, device=device)
        test_result = predictor.predict(test_base, device=device)

        # 获取定位列信息
        def _get_location_cols_from_base(base_data, rnn_dataset=None, use_mask=False):
            """从基础数据集中获取定位列信息"""
            if not use_mask:
                df = base_data.flatten_to_dataframe()
                return {
                    'No. of obs': df['No. of obs'].values,
                    'Publication': df['Publication'].values,
                    'control_group': df['control_group'].values,
                    'sowdur': df['sowdur'].values,
                }
            else:
                no_of_obs_list = []
                publication_list = []
                control_group_list = []
                sowdur_list = []

                for seq_idx, seq in enumerate(base_data.sequences):
                    seq_id = seq['seq_id']
                    no_of_obs = seq['No. of obs']
                    sowdurs = seq['sowdurs']

                    if rnn_dataset is not None:
                        daily_seq = rnn_dataset.daily_sequences[seq_idx]
                        mask = daily_seq['mask']
                        start_day = daily_seq['start_day']

                        sowdur_to_orig_idx = {int(sowdurs[i]): i for i in range(len(sowdurs))}

                        for day_idx in range(len(mask)):
                            if mask[day_idx]:
                                day = start_day + day_idx
                                if day in sowdur_to_orig_idx:
                                    orig_idx = sowdur_to_orig_idx[day]
                                    no_of_obs_list.append(no_of_obs[orig_idx])
                                    publication_list.append(seq_id[0])
                                    control_group_list.append(seq_id[1])
                                    sowdur_list.append(day)
                    else:
                        no_of_obs_list.extend(no_of_obs)
                        publication_list.extend([seq_id[0]] * len(no_of_obs))
                        control_group_list.extend([seq_id[1]] * len(no_of_obs))
                        sowdur_list.extend(sowdurs)

                return {
                    'No. of obs': np.array(no_of_obs_list),
                    'Publication': np.array(publication_list),
                    'control_group': np.array(control_group_list),
                    'sowdur': np.array(sowdur_list),
                }

        train_loc_cols = _get_location_cols_from_base(
            train_base, train_dataset if use_mask else None, use_mask=use_mask
        )
        val_loc_cols = _get_location_cols_from_base(
            val_base, val_dataset if use_mask else None, use_mask=use_mask
        )
        test_loc_cols = _get_location_cols_from_base(
            test_base, test_dataset if use_mask else None, use_mask=use_mask
        )

        from .evaluation import save_predictions_to_csv

        # 确定使用的预测和目标值
        def get_predictions_and_targets(result):
            if use_mask and 'targets_masked' in result:
                return result['predictions'][result['masks']] if 'masks' in result else result[
                    'predictions'
                ], result['targets_masked']
            return result['predictions'], result['targets']

        train_preds, train_targets = get_predictions_and_targets(train_result)
        val_preds, val_targets = get_predictions_and_targets(val_result)
        test_preds, test_targets = get_predictions_and_targets(test_result)

        save_predictions_to_csv(
            train_preds,
            train_targets,
            tables_dir / 'train_predictions.csv',
            additional_cols=train_loc_cols,
        )
        save_predictions_to_csv(
            val_preds,
            val_targets,
            tables_dir / 'val_predictions.csv',
            additional_cols=val_loc_cols,
        )
        save_predictions_to_csv(
            test_preds,
            test_targets,
            tables_dir / 'test_predictions.csv',
            additional_cols=test_loc_cols,
        )

        # 使用compute_shap_values计算特征重要性（在所有数据集上）
        try:
            logger.info("计算RNN模型的特征重要性（使用SHAP）...")
            # 合并所有数据集
            all_sequences = train_base.sequences + val_base.sequences + test_base.sequences
            all_base = BaseN2ODataset(sequences=all_sequences)
            # 创建对应的 RNN 数据集
            if model_type == 'rnn-obs':
                all_dataset = N2ODatasetForObsStepRNN(
                    all_base, fit_scalers=False, scalers=predictor.scalers
                )
            else:
                all_dataset = N2ODatasetForDailyStepRNN(
                    all_base, fit_scalers=False, scalers=predictor.scalers
                )
            shap_values, feature_names = compute_shap_values(
                predictor.model, all_dataset, model_type, device, max_samples=len(all_sequences),
                background_size=100, n_explain=200, nsamples=40
            )
            # 保存特征重要性CSV
            importance_df = pd.DataFrame(
                {
                    'feature': feature_names,
                    'importance': shap_values,
                }
            ).sort_values('importance', ascending=False)
            importance_df.to_csv(tables_dir / 'feature_importance.csv', index=False)
            # 保存特征重要性图
            plot_feature_importance(feature_names, shap_values, figs_dir / 'feature_importance.png')
        except Exception as e:
            logger.warning(f'SHAP分析失败: {e}')

        # 绘制序列预测图
        if plot_sequences and figs_dir is not None:
            from .evaluation import plot_sequence_predictions

            # 合并所有数据进行查找
            all_bases = [
                ('train', train_base, train_dataset, train_result),
                ('val', val_base, val_dataset, val_result),
                ('test', test_base, test_dataset, test_result),
            ]

            # 打印前几个序列ID帮助调试
            if len(train_base.sequences) > 0:
                sample_seq = train_base.sequences[0]
                sample_id = sample_seq['seq_id']
                logger.debug(
                    f'数据集中的序列ID示例: {sample_id}, 类型: ({type(sample_id[0]), type(sample_id[1])})'
                )

            for seq_id_tuple in plot_sequences:
                pub_arg, cg_arg = seq_id_tuple
                found = False

                for split_name, base, dataset, result in all_bases:
                    # 在该数据集中查找序列
                    for seq_idx, seq in enumerate(base.sequences):
                        seq_pub, seq_cg = seq['seq_id']

                        # 尝试类型转换来匹配
                        pub_match = False
                        cg_match = False

                        # 尝试直接比较
                        if seq_pub == pub_arg and seq_cg == cg_arg:
                            pub_match = True
                            cg_match = True
                        else:
                            # 尝试将参数转换为序列ID的类型
                            try:
                                pub_converted = type(seq_pub)(pub_arg)
                                cg_converted = type(seq_cg)(cg_arg)
                                if seq_pub == pub_converted and seq_cg == cg_converted:
                                    pub_match = True
                                    cg_match = True
                            except (ValueError, TypeError):
                                pass

                        if pub_match and cg_match:
                            # 找到序列，从data_with_predictions中获取预测值
                            data_with_preds = result['data_with_predictions']
                            pred_seq = None
                            for s in data_with_preds.sequences:
                                s_pub, s_cg = s['seq_id']
                                # 同样尝试类型转换
                                if s_pub == seq_pub and s_cg == seq_cg:
                                    pred_seq = s
                                    break
                                else:
                                    try:
                                        if (
                                            type(s_pub)(seq_pub) == s_pub
                                            and type(s_cg)(seq_cg) == s_cg
                                        ):
                                            pred_seq = s
                                            break
                                    except (ValueError, TypeError):
                                        pass

                            if pred_seq is not None and 'predicted_targets' in pred_seq:
                                predictions = np.array(pred_seq['predicted_targets'])
                                time_steps = np.array(seq['sowdurs'])
                                targets = np.array(seq['targets'])

                                # 使用原始序列ID作为文件名
                                filename_pub = str(seq_pub).replace('/', '_').replace('\\', '_')
                                filename_cg = str(seq_cg).replace('/', '_').replace('\\', '_')

                                plot_sequence_predictions(
                                    (seq_pub, seq_cg),
                                    time_steps,
                                    targets,
                                    predictions,
                                    figs_dir
                                    / f'sequence_predictions_{filename_pub}_{filename_cg}.png',
                                    mask=None,
                                )
                                logger.info(
                                    f'已绘制序列预测图: {seq_pub}_{seq_cg} ({split_name}集)'
                                )
                                found = True
                                break
                    if found:
                        break

                if not found:
                    # 列出所有可用的序列ID帮助调试
                    available_ids = []
                    for split_name, base, _, _ in all_bases:
                        for seq in base.sequences[:10]:  # 只显示前10个
                            available_ids.append(f'{seq["seq_id"]}')
                    logger.warning(
                        f'未找到序列: {pub_arg}_{cg_arg}。部分可用序列ID: {", ".join(available_ids)}...'
                    )

        results.update(
            {
                'train_metrics': train_result['metrics'],
                'val_metrics': val_result['metrics'],
                'test_metrics': test_result['metrics'],
            }
        )

    logger.info(f'预测完成，结果已保存到 {tables_dir}')
    return results


def predict_tif_data(
    model_dir: Path | str,
    tif_dir: Path | str,
    output_dir: Path | str,
    device: str = 'cuda:0',
    batch_size: int = 256,
) -> dict[str, Any]:
    """
    使用训练好的RNN模型对TIF格式数据进行预测

    Args:
        model_dir: 模型目录
        tif_dir: TIF数据目录（如 input_2020）
        output_dir: 输出目录
        device: 设备
        batch_size: 批次大小

    Returns:
        预测结果摘要
    """
    model_dir = Path(model_dir)
    tif_dir = Path(tif_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载预测器
    predictor = N2OPredictor(model_dir)

    # 检查模型类型
    if not predictor.model_type.startswith('rnn'):
        raise ValueError(f'TIF预测只支持RNN模型，当前模型类型: {predictor.model_type}')

    # 加载TIF数据
    logger.info(f'从 {tif_dir} 加载TIF数据...')
    tif_loader = TifDataLoader(tif_dir)

    # 获取所有有效组合
    combinations = tif_loader.get_prediction_combinations()
    logger.info(f'共 {len(combinations)} 个有效组合')

    # 准备设备
    device_obj = torch.device(device)
    predictor.model = predictor.model.to(device_obj)
    predictor.model.eval()
    for m in predictor.model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.05
            m.train()
        if isinstance(m, (nn.GRU, nn.LSTM)):
            m.dropout = 0.05
            m.train()

    # 记录结果
    results = {
        'model_dir': str(model_dir),
        'tif_dir': str(tif_dir),
        'output_dir': str(output_dir),
        'total_combinations': len(combinations),
        'completed_files': [],
        'total_pixels_processed': 0,
    }

    # 记录总开始时间
    total_start_time = time.time()

    # 遍历所有组合进行预测
    progress_bar = tqdm(combinations, desc='预测进度')
    for idx, (crop, fert, appl, source) in enumerate(progress_bar, 1):
        combination_name = f'{crop}_{source}_{fert}_{appl}'
        combination_start_time = time.time()

        # 更新进度条描述
        progress_bar.set_description(f'预测 [{idx}/{len(combinations)}] {combination_name}')

        # 创建数据集
        logger.info(f'[{idx}/{len(combinations)}] 正在加载 {combination_name} 数据...')
        dataset_start_time = time.time()
        dataset = tif_loader.create_rnn_dataset(
            crop, fert, appl, source, predictor.scalers, model_type=predictor.model_type
        )
        dataset_load_time = time.time() - dataset_start_time

        if len(dataset) == 0:
            logger.warning(f'跳过空数据集: {combination_name}')
            continue

        n_pixels = len(dataset)
        logger.info(f'  数据加载完成: {n_pixels} 像素, 耗时 {dataset_load_time:.1f}s')

        # 创建数据加载器
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        # 计算批次数量
        n_batches = (n_pixels + batch_size - 1) // batch_size
        logger.info(f'  共 {n_batches} 个批次 (batch_size={batch_size})')

        # 预测
        all_predictions = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader, 1):
                if batch_idx % 10 == 0 or batch_idx == n_batches:
                    logger.info(f'  批次进度: {batch_idx}/{n_batches}')
                static_numeric = batch['static_numeric'].to(device_obj)
                dynamic_numeric = batch['dynamic_numeric'].to(device_obj)
                static_categorical = batch['static_categorical'].to(device_obj)
                dynamic_categorical = batch['dynamic_categorical'].to(device_obj)

                # TIF数据集所有样本的序列长度相同
                seq_len = dataset.n_days
                batch_size_actual = len(static_numeric)
                seq_lengths = torch.tensor([seq_len] * batch_size_actual, device=device_obj)

                predictions = predictor.model(
                    static_numeric,
                    static_categorical,
                    dynamic_numeric,
                    dynamic_categorical,
                    seq_lengths,
                )

                predictions_np = predictions.cpu().numpy()

                # 逆转换每个像素的预测值
                for i in range(len(predictions_np)):
                    pred_scaled = predictions_np[i, :seq_len]
                    pred_orig = dataset.inverse_transform_targets(pred_scaled)
                    all_predictions.append(pred_orig)

        # 转换为数组
        predictions_array = np.array(all_predictions)  # shape: (n_pixels, n_days)

        # 获取有效天掩码
        valid_masks = dataset.get_valid_masks()  # shape: (n_pixels, n_days)

        # 将休耕期（无效天）的预测值设为 NaN
        predictions_array[~valid_masks] = np.nan

        # 获取像素索引
        pixel_indices = dataset.get_pixel_indices()

        # 保存预测结果（包含 NaN 的休耕期）
        output_path = tif_loader.save_predictions(
            predictions_array,
            pixel_indices,
            crop,
            fert,
            appl,
            source,
            output_dir,
        )

        results['completed_files'].append(str(output_path))
        results['total_pixels_processed'] += len(pixel_indices)

        # 计算用时
        combination_time = time.time() - combination_start_time

        # 显示详细信息
        logger.info(
            f'[{idx}/{len(combinations)}] {combination_name}: '
            f'{n_pixels} 像素, 耗时 {combination_time:.1f}s, '
            f'累计 {results["total_pixels_processed"]} 像素'
        )

    # 计算总用时
    total_time = time.time() - total_start_time
    avg_time_per_combination = (
        total_time / len(results['completed_files']) if results['completed_files'] else 0
    )

    logger.info(f'\n预测完成！')
    logger.info(f'  生成文件数: {len(results["completed_files"])}')
    logger.info(f'  总处理像素数: {results["total_pixels_processed"]}')
    logger.info(f'  总耗时: {total_time:.1f}s ({total_time / 60:.1f}min)')
    logger.info(f'  平均每组合: {avg_time_per_combination:.1f}s')

    return results

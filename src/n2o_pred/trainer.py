"""
训练器模块
"""
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .evaluation import compute_metrics
from .rf import N2OPredictorRF
from .rnn import N2OPredictorRNN
from .utils import create_logger

logger = create_logger(__name__)


@dataclass
class RNNTrainConfig:
    """RNN训练配置"""
    max_epochs: int = 300
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 30
    gradient_clip: float = 5.0
    device: str = 'cuda:0'
    
    # 模型参数
    embedding_dim: int = 8
    hidden_size: int = 96
    num_layers: int = 2
    rnn_type: str = 'GRU'
    dropout: float = 0.2
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'max_epochs': self.max_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'patience': self.patience,
            'gradient_clip': self.gradient_clip,
            'device': self.device,
            'embedding_dim': self.embedding_dim,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'rnn_type': self.rnn_type,
            'dropout': self.dropout,
        }


class MaskedMSELoss(nn.Module):
    """带掩码的MSE损失（用于DailyStepRNN）"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size, seq_len]
            targets: [batch_size, seq_len]
            mask: [batch_size, seq_len], True表示真实点
        """
        # 只计算真实测量点的损失
        masked_predictions = predictions[mask]
        masked_targets = targets[mask]
        
        if len(masked_predictions) == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        return nn.functional.mse_loss(masked_predictions, masked_targets)


def train_rnn_predictor(
    model: N2OPredictorRNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: RNNTrainConfig,
    train_dataset: Any,
    val_dataset: Any,
    save_dir: Path,
    use_mask: bool = False
) -> dict[str, Any]:
    """
    训练RNN模型
    
    Args:
        model: RNN模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 训练配置
        train_dataset: 训练数据集（用于逆转换）
        val_dataset: 验证数据集（用于逆转换）
        save_dir: 保存目录
        use_mask: 是否使用掩码损失（DailyStepRNN为True）
    
    Returns:
        训练结果字典
    """
    device = torch.device(config.device)
    model = model.to(device)
    
    # 损失函数
    if use_mask:
        criterion = MaskedMSELoss()
    else:
        criterion = nn.MSELoss()
    
    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config.patience // 3
    )
    
    # 训练历史
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_path = save_dir / 'best_model.pt'
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"开始训练，共 {config.max_epochs} 轮")
    logger.info(f"模型参数数量: {model.count_parameters()}")
    
    for epoch in range(config.max_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.max_epochs}')
        for batch in pbar:
            # 准备数据
            static_numeric = batch['static_numeric'].to(device)
            dynamic_numeric = batch['dynamic_numeric'].to(device)
            static_categorical = batch['static_categorical'].to(device)
            dynamic_categorical = batch['dynamic_categorical'].to(device)
            targets = batch['targets'].to(device)
            seq_lengths = batch['seq_lengths'].to(device)
            
            # 前向传播
            predictions = model(static_numeric, static_categorical, 
                              dynamic_numeric, dynamic_categorical, seq_lengths)
            
            # 计算损失
            if use_mask:
                mask = batch['mask'].to(device)
                loss = criterion(predictions, targets, mask)
            else:
                # 创建掩码（只计算有效长度内的损失）
                batch_size = len(seq_lengths)
                max_len = predictions.shape[1]
                mask = torch.arange(max_len, device=device).unsqueeze(0) < seq_lengths.unsqueeze(1)
                # 只计算有效序列长度内的损失
                masked_predictions = predictions[mask]
                masked_targets = targets[mask]
                loss = criterion(masked_predictions, masked_targets)
            
            # 检查loss是否为NaN
            if torch.isnan(loss):
                logger.error(f"检测到NaN损失，跳过该批次")
                logger.error(f"预测值范围: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
                logger.error(f"目标值范围: [{targets.min().item():.4f}, {targets.max().item():.4f}]")
                continue
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                static_numeric = batch['static_numeric'].to(device)
                dynamic_numeric = batch['dynamic_numeric'].to(device)
                static_categorical = batch['static_categorical'].to(device)
                dynamic_categorical = batch['dynamic_categorical'].to(device)
                targets = batch['targets'].to(device)
                seq_lengths = batch['seq_lengths'].to(device)
                
                predictions = model(static_numeric, static_categorical,
                                  dynamic_numeric, dynamic_categorical, seq_lengths)
                
                if use_mask:
                    mask = batch['mask'].to(device)
                    loss = criterion(predictions, targets, mask)
                else:
                    batch_size = len(seq_lengths)
                    max_len = predictions.shape[1]
                    mask = torch.arange(max_len, device=device).unsqueeze(0) < seq_lengths.unsqueeze(1)
                    masked_predictions = predictions[mask]
                    masked_targets = targets[mask]
                    loss = criterion(masked_predictions, masked_targets)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / max(val_batches, 1)  # 避免除零
        val_losses.append(avg_val_loss)
        
        # 学习率调整
        scheduler.step(avg_val_loss)
        
        logger.info(f'Epoch {epoch+1}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}')
        
        # 早停检查
        if avg_val_loss < best_val_loss or epoch == 0:
            # 第一个epoch或者有改善时保存
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f'早停：{config.patience} 轮没有改善')
                break
    
    # 加载最佳模型
    checkpoint = torch.load(save_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f'训练完成，最佳epoch: {best_epoch+1}')
    
    # 在训练集和验证集上评估
    model.eval()
    
    train_preds_list = []
    train_targets_list = []
    val_preds_list = []
    val_targets_list = []
    
    with torch.no_grad():
        # 训练集
        for batch in train_loader:
            static_numeric = batch['static_numeric'].to(device)
            dynamic_numeric = batch['dynamic_numeric'].to(device)
            static_categorical = batch['static_categorical'].to(device)
            dynamic_categorical = batch['dynamic_categorical'].to(device)
            seq_lengths = batch['seq_lengths'].to(device)
            targets_original = batch['targets_original']
            
            predictions = model(static_numeric, static_categorical,
                              dynamic_numeric, dynamic_categorical, seq_lengths)
            
            # 转换回原始空间
            predictions_np = predictions.cpu().numpy()
            
            for i in range(len(seq_lengths)):
                seq_len = seq_lengths[i].item()
                pred_scaled = predictions_np[i, :seq_len]
                target_orig = targets_original[i, :seq_len].numpy()
                
                # 逆转换
                pred_orig = train_dataset.inverse_transform_targets(pred_scaled)
                
                if use_mask:
                    mask_i = batch['mask'][i, :seq_len].numpy()
                    train_preds_list.extend(pred_orig[mask_i])
                    train_targets_list.extend(target_orig[mask_i])
                else:
                    train_preds_list.extend(pred_orig)
                    train_targets_list.extend(target_orig)
        
        # 验证集
        for batch in val_loader:
            static_numeric = batch['static_numeric'].to(device)
            dynamic_numeric = batch['dynamic_numeric'].to(device)
            static_categorical = batch['static_categorical'].to(device)
            dynamic_categorical = batch['dynamic_categorical'].to(device)
            seq_lengths = batch['seq_lengths'].to(device)
            targets_original = batch['targets_original']
            
            predictions = model(static_numeric, static_categorical,
                              dynamic_numeric, dynamic_categorical, seq_lengths)
            
            predictions_np = predictions.cpu().numpy()
            
            for i in range(len(seq_lengths)):
                seq_len = seq_lengths[i].item()
                pred_scaled = predictions_np[i, :seq_len]
                target_orig = targets_original[i, :seq_len].numpy()
                
                pred_orig = val_dataset.inverse_transform_targets(pred_scaled)
                
                if use_mask:
                    mask_i = batch['mask'][i, :seq_len].numpy()
                    val_preds_list.extend(pred_orig[mask_i])
                    val_targets_list.extend(target_orig[mask_i])
                else:
                    val_preds_list.extend(pred_orig)
                    val_targets_list.extend(target_orig)
    
    train_preds = np.array(train_preds_list)
    train_targets = np.array(train_targets_list)
    val_preds = np.array(val_preds_list)
    val_targets = np.array(val_targets_list)
    
    # 计算指标
    train_metrics = compute_metrics(train_targets, train_preds)
    val_metrics = compute_metrics(val_targets, val_preds)
    
    logger.info(f"训练集 - R2: {train_metrics['R2']:.4f}, RMSE: {train_metrics['RMSE']:.4f}")
    logger.info(f"验证集 - R2: {val_metrics['R2']:.4f}, RMSE: {val_metrics['RMSE']:.4f}")
    
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'train_predictions': train_preds,
        'val_predictions': val_preds,
        'train_targets': train_targets,
        'val_targets': val_targets,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'n_parameters': model.count_parameters(),
    }
    
    return results


@dataclass
class RFTrainConfig:
    """随机森林训练配置"""
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: str = 'sqrt'
    random_state: int = 42
    n_jobs: int = -1
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
        }


def train_rf_predictor(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: RFTrainConfig,
    save_dir: Path
) -> tuple[N2OPredictorRF, dict[str, Any]]:
    """
    训练随机森林模型
    
    Args:
        train_df: 训练数据
        val_df: 验证数据
        config: 训练配置
        save_dir: 保存目录
    
    Returns:
        (模型, 结果字典)
    """
    logger.info("开始训练随机森林模型")
    
    # 创建并训练模型
    model = N2OPredictorRF(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        random_state=config.random_state,
        n_jobs=config.n_jobs
    )
    
    model.fit(train_df)
    logger.info(f"模型训练完成")
    logger.info(f"模型复杂度指标: {model.count_parameters()} 个节点")
    
    # 预测
    train_preds = model.predict(train_df)
    val_preds = model.predict(val_df)
    
    from .preprocessing import LABELS
    train_targets = train_df[LABELS[0]].values
    val_targets = val_df[LABELS[0]].values
    
    # 计算指标
    train_metrics = compute_metrics(train_targets, train_preds)
    val_metrics = compute_metrics(val_targets, val_preds)
    
    logger.info(f"训练集 - R2: {train_metrics['R2']:.4f}, RMSE: {train_metrics['RMSE']:.4f}")
    logger.info(f"验证集 - R2: {val_metrics['R2']:.4f}, RMSE: {val_metrics['RMSE']:.4f}")
    
    # 保存模型
    model_path = save_dir / 'model.pkl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    logger.info(f"模型已保存到 {model_path}")
    
    # 获取特征重要性
    feature_importances = model.get_feature_importances()
    
    results = {
        'train_predictions': train_preds,
        'val_predictions': val_preds,
        'train_targets': train_targets,
        'val_targets': val_targets,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'feature_importances': feature_importances,
        'n_parameters': model.count_parameters(),
    }
    
    return model, results


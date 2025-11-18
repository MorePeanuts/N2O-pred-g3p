"""
通用工具函数模块
"""
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str: str = 'cuda:0') -> torch.device:
    """获取可用的计算设备"""
    if device_str.startswith('cuda'):
        if torch.cuda.is_available():
            return torch.device(device_str)
        else:
            print(f"CUDA不可用，使用CPU")
            return torch.device('cpu')
    return torch.device(device_str)


def save_json(data: Any, path: Path) -> None:
    """保存JSON文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """加载JSON文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_logger(name: str, log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """创建日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def symlog(x: np.ndarray, C: float = 1.0) -> np.ndarray:
    """
    对称对数变换，用于处理包含负值的偏态分布数据
    symlog(x) = sign(x) * log(1 + |x|/C)
    """
    return np.sign(x) * np.log1p(np.abs(x) / C)


def inv_symlog(y: np.ndarray, C: float = 1.0) -> np.ndarray:
    """
    对称对数逆变换
    x = sign(y) * C * (exp(|y|) - 1)
    """
    return np.sign(y) * C * (np.exp(np.abs(y)) - 1.0)


class SymlogTransformer:
    """Symlog变换器，与sklearn的transformer接口兼容"""
    
    def __init__(self, C: float = 1.0):
        self.C = C
    
    def fit(self, X: np.ndarray) -> 'SymlogTransformer':
        """拟合（无需拟合参数）"""
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """变换"""
        return symlog(X, self.C)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """逆变换"""
        return inv_symlog(X, self.C)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """拟合并变换"""
        return self.fit(X).transform(X)


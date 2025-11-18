"""
RNN模型模块
"""
from typing import Literal

import torch
import torch.nn as nn


class N2OPredictorRNN(nn.Module):
    """N2O排放预测RNN模型"""
    
    def __init__(
        self,
        num_numeric_static: int,
        num_numeric_dynamic: int,
        categorical_static_cardinalities: list[int],
        categorical_dynamic_cardinalities: list[int],
        embedding_dim: int = 8,
        hidden_size: int = 96,
        num_layers: int = 2,
        rnn_type: Literal['GRU', 'LSTM'] = 'GRU',
        dropout: float = 0.2,
    ):
        """
        Args:
            num_numeric_static: 静态数值特征数量
            num_numeric_dynamic: 动态数值特征数量
            categorical_static_cardinalities: 静态分类特征的类别数列表
            categorical_dynamic_cardinalities: 动态分类特征的类别数列表
            embedding_dim: 嵌入维度
            hidden_size: RNN隐藏层大小
            num_layers: RNN层数
            rnn_type: RNN类型 ('GRU' 或 'LSTM')
            dropout: Dropout比例
        """
        super().__init__()
        
        self.num_numeric_static = num_numeric_static
        self.num_numeric_dynamic = num_numeric_dynamic
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # 静态分类特征的Embedding层
        self.static_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim)
            for cardinality in categorical_static_cardinalities
        ])
        
        # 动态分类特征的Embedding层
        self.dynamic_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim)
            for cardinality in categorical_dynamic_cardinalities
        ])
        
        # 计算静态特征的总维度
        static_dim = num_numeric_static + len(categorical_static_cardinalities) * embedding_dim
        
        # 静态特征MLP（生成RNN初始hidden state）
        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size * num_layers),
            nn.Tanh()
        )
        
        # 计算动态特征的总维度
        dynamic_dim = num_numeric_dynamic + len(categorical_dynamic_cardinalities) * embedding_dim
        
        # 动态特征投影层
        self.dynamic_projection = nn.Sequential(
            nn.Linear(dynamic_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # RNN层
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(
                hidden_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"不支持的RNN类型: {rnn_type}")
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'rnn' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        static_numeric: torch.Tensor,
        static_categorical: torch.Tensor,
        dynamic_numeric: torch.Tensor,
        dynamic_categorical: torch.Tensor,
        seq_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            static_numeric: [batch_size, num_static_numeric]
            static_categorical: [batch_size, num_static_categorical]
            dynamic_numeric: [batch_size, max_seq_len, num_dynamic_numeric]
            dynamic_categorical: [batch_size, max_seq_len, num_dynamic_categorical]
            seq_lengths: [batch_size]
        
        Returns:
            predictions: [batch_size, max_seq_len]
        """
        batch_size, max_seq_len = dynamic_numeric.shape[0], dynamic_numeric.shape[1]
        
        # 1. 处理静态特征
        # 嵌入静态分类特征
        static_cat_embedded = []
        for i, embedding in enumerate(self.static_embeddings):
            static_cat_embedded.append(embedding(static_categorical[:, i]))
        
        if static_cat_embedded:
            static_cat_embedded = torch.cat(static_cat_embedded, dim=1)
            static_features = torch.cat([static_numeric, static_cat_embedded], dim=1)
        else:
            static_features = static_numeric
        
        # 通过MLP生成初始hidden state
        h0 = self.static_mlp(static_features)  # [batch_size, hidden_size * num_layers]
        h0 = h0.view(batch_size, self.num_layers, self.hidden_size)  # [batch_size, num_layers, hidden_size]
        h0 = h0.transpose(0, 1).contiguous()  # [num_layers, batch_size, hidden_size]
        
        # 2. 处理动态特征
        # 嵌入动态分类特征
        dynamic_cat_embedded = []
        for i, embedding in enumerate(self.dynamic_embeddings):
            dynamic_cat_embedded.append(embedding(dynamic_categorical[:, :, i]))
        
        if dynamic_cat_embedded:
            dynamic_cat_embedded = torch.stack(dynamic_cat_embedded, dim=2)  # [batch_size, max_seq_len, num_cat, emb_dim]
            dynamic_cat_embedded = dynamic_cat_embedded.view(batch_size, max_seq_len, -1)  # [batch_size, max_seq_len, num_cat * emb_dim]
            dynamic_features = torch.cat([dynamic_numeric, dynamic_cat_embedded], dim=2)
        else:
            dynamic_features = dynamic_numeric
        
        # 投影动态特征
        dynamic_projected = self.dynamic_projection(dynamic_features)  # [batch_size, max_seq_len, hidden_size]
        
        # 3. 通过RNN
        # Pack padded sequence以处理变长序列
        packed_input = nn.utils.rnn.pack_padded_sequence(
            dynamic_projected,
            seq_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        if self.rnn_type == 'LSTM':
            # LSTM需要cell state
            c0 = torch.zeros_like(h0)
            packed_output, _ = self.rnn(packed_input, (h0, c0))
        else:
            packed_output, _ = self.rnn(packed_input, h0)
        
        # Unpack
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=max_seq_len
        )  # [batch_size, max_seq_len, hidden_size]
        
        # 4. 输出层
        predictions = self.output_layer(rnn_output).squeeze(-1)  # [batch_size, max_seq_len]
        
        return predictions
    
    def count_parameters(self) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


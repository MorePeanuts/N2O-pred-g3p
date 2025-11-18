# N2O排放预测系统

基于机器学习的土壤N2O排放预测系统，支持随机森林和RNN模型。

## 项目概述

本项目实现了一个完整的N2O排放预测系统，包括：

- **三种建模方案**：
  - 随机森林（Random Forest）- Baseline
  - 观测步长RNN（Observation-driven RNN）
  - 每日步长RNN（Daily-step RNN）

- **完整的实验管理**：
  - 多种子训练和验证
  - 自动模型评估和可视化
  - 模型对比分析
  - 新数据预测

## 安装

项目使用 `uv` 作为包管理器：

```bash
# 安装依赖
uv sync
```

## 使用方法

### 1. 数据预处理

```bash
uv run n2o-pred data --preprocessing
```

这会将原始数据 `datasets/data_EUR_raw.csv` 处理成时间序列格式，保存到 `datasets/data_EUR_processed.pkl`。

### 2. 训练模型

#### 训练随机森林模型

```bash
# 使用单个种子
uv run n2o-pred train --model rf --seed 42

# 使用多个随机种子
uv run n2o-pred train --model rf --max-split 20
```

#### 训练RNN模型（观测步长）

```bash
# 使用单个种子
uv run n2o-pred train --model rnn-obs --seed 42 --device cuda:0

# 使用多个随机种子
uv run n2o-pred train --model rnn-obs --max-split 20 --device cuda:0
```

#### 训练RNN模型（每日步长）

```bash
# 使用单个种子
uv run n2o-pred train --model rnn-daily --seed 42 --device cuda:0

# 从已有实验加载相同的种子
uv run n2o-pred train --model rnn-daily --seed-from outputs/exp_xxx/summary.json --device cuda:0
```

#### 训练参数

- `--model`: 模型类型（rf, rnn-obs, rnn-daily）
- `--seed`: 使用单个随机种子
- `--max-split`: 使用多个随机种子进行训练
- `--seed-from`: 从summary.json加载种子列表
- `--train-split`: 训练集比例（默认0.9）
- `--device`: 设备（默认cuda:0）
- `--output`: 指定输出目录
- `--max-epochs`: 最大训练轮次（默认300）
- `--batch-size`: 批次大小（默认32）
- `--lr`: 学习率（默认0.001）
- `--patience`: 早停耐心值（默认30）

### 3. 模型对比

```bash
# 比较2个模型
uv run n2o-pred compare --models outputs/exp_1 outputs/exp_2

# 比较多个模型
uv run n2o-pred compare --models outputs/exp_1 outputs/exp_2 outputs/exp_3 --output outputs/comparison
```

### 4. 预测

```bash
uv run n2o-pred predict --model outputs/exp_xxx/split_42 --dataset datasets/data_EUR_processed.pkl --output predictions.csv
```

## 项目结构

```
N2O-pred-g3p/
├── src/n2o_pred/
│   ├── cli.py              # 命令行接口
│   ├── preprocessing.py    # 数据预处理
│   ├── dataset.py          # 数据集类
│   ├── rnn.py              # RNN模型
│   ├── rf.py               # 随机森林模型
│   ├── trainer.py          # 训练器
│   ├── evaluation.py       # 评估和可视化
│   ├── experiment.py       # 实验管理器
│   ├── compare.py          # 模型对比工具
│   ├── predict.py          # 预测工具
│   └── utils.py            # 工具函数
├── datasets/
│   ├── data_EUR_raw.csv    # 原始数据
│   └── data_EUR_processed.pkl  # 处理后的数据
├── preprocessor/
│   └── encoders.pkl        # 标签编码器
├── outputs/                # 实验输出
└── docs/
    └── research-plan.md    # 研究计划
```

## 输出结构

每次实验会创建如下结构的输出：

```
outputs/exp_{datetime}/
├── split_{seed}/
│   ├── figs/
│   │   ├── train_loss_curve.png
│   │   ├── predictions_vs_actual.png
│   │   ├── feature_importance.png
│   │   └── sequence_predictions_{seq_id}.png
│   ├── tables/
│   │   ├── train_predictions.csv
│   │   ├── val_predictions.csv
│   │   └── feature_importance.csv
│   ├── metrics.json
│   ├── model.pt (或 model.pkl)
│   ├── config.json
│   └── scalers.pkl
├── summary.json
└── experiment.log
```

## 模型特点

### 随机森林（RF）
- 使用所有静态和动态特征
- 不需要特征缩放
- 训练速度快
- 作为Baseline

### 观测步长RNN
- 每个观测点作为一个时间步
- 添加时间间隔特征（time_delta）
- 适用于不规则时间序列

### 每日步长RNN
- 每天作为一个时间步
- 对非观测日进行线性插值
- 使用掩码损失函数，只计算真实测量点的损失
- 更适合捕捉每日变化

## 特征工程

- **Daily fluxes**: Symlog变换 + StandardScaler
- **Prec, Split N amount, ferdur**: log(x+1)变换 + StandardScaler
- **其他数值特征**: StandardScaler

## 评估指标

- R² (R-squared)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MRE (Mean Relative Error)
- 特征重要性（SHAP分析）

## 技术栈

- **深度学习**: PyTorch
- **机器学习**: scikit-learn
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn
- **特征重要性**: SHAP

## 注意事项

1. RNN模型的评估指标在原始空间计算，确保与RF模型可比
2. DailyStepRNN使用掩码损失，只在真实测量点计算损失
3. 所有scalers基于真实测量点拟合，但转换所有点
4. 建议使用GPU训练RNN模型以加快速度

## 许可证

MIT License


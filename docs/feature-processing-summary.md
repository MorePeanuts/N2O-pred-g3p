# 模型特征选择与数据处理总结

本文档总结了 N2O-pred-g3p 项目中三种模型（随机森林、RNN观测步长、RNN每日步长）的特征选择和数据处理方式。

## 目录

- [数据预处理](#数据预处理)
- [随机森林模型 (RF)](#随机森林模型-rf)
- [RNN观测步长模型 (rnn-obs)](#rnn观测步长模型-rnn-obs)
- [RNN每日步长模型 (rnn-daily)](#rnn每日步长模型-rnn-daily)
- [特征变换对比](#特征变换对比)

---

## 数据预处理

### 原始数据字段

| 类型 | 字段名 | 说明 |
|------|--------|------|
| **静态数值特征** | Clay | 粘土含量 |
| | CEC | 阳离子交换容量 |
| | BD | 土壤容重 |
| | pH | pH值 |
| | SOC | 有机碳含量 |
| | TN | 总氮含量 |
| **动态数值特征** | Temp | 温度 |
| | Prec | 降水量 |
| | ST | 土壤温度 |
| | WFPS | 土壤含水量 |
| | Split N amount | 上次施肥量 |
| | ferdur | 距上次施肥的天数 |
| | Total N amount | 总施肥量 |
| **静态分类特征** | crop_class | 作物类型 |
| **动态分类特征** | fertilization_class | 上次施肥类型 |
| | appl_class | 上次施肥方式 |
| **目标变量** | Daily fluxes | N2O排放通量 |

### 预处理步骤 (`preprocessing.py`)

1. **丢弃字段**：`NH4+-N`, `NO3_-N`, `MN`, `C/N`
2. **修正ferdur**：将 `ferdur=-1`（未施肥）修改为 `0`
3. **排序**：按 `(Publication, control_group, sowdur)` 升序排列
4. **分类编码**：使用 `LabelEncoder` 对分类变量编码
5. **TN缺失值填充**：同序列内前向填充 → 后向填充 → 均值填充
6. **构建序列**：按 `(Publication, control_group)` 分组构建时间序列

---

## 随机森林模型 (RF)

### 特征选择

**文件位置**: `src/n2o_pred/rf.py`

| 特征类型 | 字段 |
|----------|------|
| **静态数值特征** (6个) | Clay, CEC, BD, pH, SOC, TN |
| **动态数值特征** (5个) | Temp, Prec, ST, WFPS, **Total N amount** |
| **静态分类特征** (1个) | crop_class |
| **动态分类特征** (2个) | fertilization_class, appl_class |

**总计**: 14 个特征

### 关键差异

- ✅ 使用 **Total N amount**（总施肥量）
- ❌ 不使用 **Split N amount**（上次施肥量）
- ❌ 不使用 **ferdur**（距上次施肥天数）

### 数据处理

| 处理项 | 方式 |
|--------|------|
| 数据格式 | 展开为 DataFrame（每个观测点一行）|
| 特征缩放 | **无**（随机森林不需要缩放）|
| 目标变量变换 | **无**（直接使用原始值）|

### 模型参数 (`RFTrainConfig`)

```python
n_estimators: int = 1000          # 树的数量
max_depth: int | None = None        # 树的最大深度
min_samples_split: int = 10         # 分裂内部节点所需最小样本数
min_samples_leaf: int = 5           # 叶节点所需最小样本数
max_features: str = "sqrt"          # 寻找最佳分裂时考虑的特征数
```

---

## RNN观测步长模型 (rnn-obs)

### 特征选择

**文件位置**: `src/n2o_pred/dataset.py` - `N2ODatasetForObsStepRNN`

| 特征类型 | 字段 |
|----------|------|
| **静态数值特征** (6个) | Clay, CEC, BD, pH, SOC, TN |
| **动态数值特征** (7个) | Temp, Prec, ST, WFPS, **Split N amount**, **ferdur**, **time_delta** |
| **静态分类特征** (1个) | crop_class |
| **动态分类特征** (2个) | fertilization_class, appl_class |

**总计**: 16 个特征

### 关键差异

- ✅ 使用 **Split N amount**（上次施肥量）
- ✅ 使用 **ferdur**（距上次施肥天数）
- ✅ 新增 **time_delta**（与前一观测点的时间间隔）
- ❌ 不使用 **Total N amount**

### 数据处理

#### 特征变换

| 特征 | 变换方式 |
|------|----------|
| **Daily fluxes** (目标) | Symlog → StandardScaler |
| **Prec** | log(x+1) → StandardScaler |
| **Split N amount** | log(x+1) → StandardScaler |
| **ferdur** | log(x+1) → StandardScaler |
| **Temp, ST, WFPS** | StandardScaler |
| **Clay, CEC, BD, pH, SOC, TN** | StandardScaler |

#### time_delta 计算

```python
time_delta[0] = 0
time_delta[t] = sowdur[t] - sowdur[t-1]  # t >= 1
```

### 模型架构 (`N2OPredictorRNN`)

```python
embedding_dim: int = 32           # 分类特征嵌入维度
hidden_size: int = 256             # RNN隐藏层大小
num_layers: int = 3                 # RNN层数
rnn_type: str = "LSTM"              # RNN类型 (GRU/LSTM)
dropout: float = 0.3                # Dropout比例
```

**模型流程**:
1. 静态分类特征 → Embedding层
2. 静态数值特征 + 嵌入后的静态分类特征 → MLP → RNN初始hidden state
3. 动态分类特征 → Embedding层
4. 动态数值特征 + 嵌入后的动态分类特征 → 投影层 → RNN输入
5. RNN层 → 输出层 → 预测

---

## RNN每日步长模型 (rnn-daily)

### 特征选择

**文件位置**: `src/n2o_pred/dataset.py` - `N2ODatasetForDailyStepRNN`

| 特征类型 | 字段 |
|----------|------|
| **静态数值特征** (6个) | Clay, CEC, BD, pH, SOC, TN |
| **动态数值特征** (6个) | Temp, Prec, ST, WFPS, **Split N amount**, **ferdur** |
| **静态分类特征** (1个) | crop_class |
| **动态分类特征** (2个) | fertilization_class, appl_class |

**总计**: 15 个特征

### 关键差异

- ✅ 使用 **Split N amount**（上次施肥量）
- ✅ 使用 **ferdur**（距上次施肥天数）
- ❌ 不使用 **time_delta**（已按天对齐）
- ❌ 不使用 **Total N amount**
- ✅ 使用 **mask** 标识真实测量点

### 数据处理

#### 序列展开（每日步长）

原始观测序列 → 插值填充为每日序列：

| 特征 | 插值方式 |
|------|----------|
| **Temp, ST, WFPS** | 线性插值 |
| **Prec** | 填充为 0 |
| **Split N amount** | 前向填充 |
| **ferdur** | 重新计算（距上次施肥的天数）|
| **crop_class** | 保持静态 |
| **fertilization_class, appl_class** | 前向填充 |
| **Daily fluxes** (目标) | 线性插值（但不计入损失）|

#### Mask机制

- `mask=True`：真实测量点，计入损失
- `mask=False`：插值点，不计入损失（`MaskedMSELoss`）

#### 特征变换

与 rnn-obs 相同，但**只在真实测量点上拟合scaler**。

| 特征 | 变换方式 |
|------|----------|
| **Daily fluxes** (目标) | Symlog → StandardScaler |
| **Prec** | log(x+1) → StandardScaler |
| **Split N amount** | log(x+1) → StandardScaler |
| **ferdur** | log(x+1) → StandardScaler |
| **Temp, ST, WFPS** | StandardScaler |
| **Clay, CEC, BD, pH, SOC, TN** | StandardScaler |

### 模型架构

与 rnn-obs 使用相同的 `N2OPredictorRNN` 模型，唯一区别是：

- 使用 `MaskedMSELoss` 只在真实测量点上计算损失
- 动态数值特征维度为 6（不含 time_delta）

---

## 特征变换对比

### Symlog变换 (`utils.py`)

用于处理包含负值的偏态分布数据：

```python
symlog(x) = sign(x) * log(1 + |x| / C)
```

### 特征变换总结表

| 特征 | RF | rnn-obs | rnn-daily |
|------|----|---------|-----------|
| **Daily fluxes** | 原始值 | Symlog + StandardScaler | Symlog + StandardScaler |
| **Temp** | 原始值 | StandardScaler | StandardScaler |
| **Prec** | 原始值 | log(x+1) + StandardScaler | log(x+1) + StandardScaler |
| **ST** | 原始值 | StandardScaler | StandardScaler |
| **WFPS** | 原始值 | StandardScaler | StandardScaler |
| **Split N amount** | ❌ | log(x+1) + StandardScaler | log(x+1) + StandardScaler |
| **ferdur** | ❌ | log(x+1) + StandardScaler | log(x+1) + StandardScaler |
| **Total N amount** | ✅ 原始值 | ❌ | ❌ |
| **time_delta** | ❌ | ✅ StandardScaler | ❌ |
| **Clay, CEC, BD, pH, SOC, TN** | 原始值 | StandardScaler | StandardScaler |
| **crop_class, fertilization_class, appl_class** | Label编码 | Label编码 + Embedding | Label编码 + Embedding |

### 模型对比总表

| 特性 | 随机森林 (RF) | RNN观测步长 (rnn-obs) | RNN每日步长 (rnn-daily) |
|------|---------------|------------------------|--------------------------|
| 输入格式 | DataFrame | 序列（观测点驱动）| 序列（时间驱动，每日一步）|
| 时间步 | 无 | 每个观测点一步 | 每天一步 |
| 施肥特征 | Total N amount | Split N amount + ferdur | Split N amount + ferdur |
| 额外特征 | - | time_delta | mask（掩码）|
| 特征缩放 | 不需要 | 需要 | 需要 |
| 损失函数 | MSE（回归）| MSE | MaskedMSE（仅真实点）|
| 评估指标空间 | 原始空间 | 原始空间（逆变换后）| 原始空间（逆变换后）|

---

## 文件参考

| 模块 | 文件 | 说明 |
|------|------|------|
| 预处理 | `src/n2o_pred/preprocessing.py` | 数据预处理和特征定义 |
| 数据集 | `src/n2o_pred/dataset.py` | 三种模型的数据集类 |
| 随机森林 | `src/n2o_pred/rf.py` | 随机森林模型 |
| RNN | `src/n2o_pred/rnn.py` | RNN模型架构 |
| 训练器 | `src/n2o_pred/trainer.py` | 训练逻辑和配置 |
| 工具 | `src/n2o_pred/utils.py` | Symlog变换等工具函数 |

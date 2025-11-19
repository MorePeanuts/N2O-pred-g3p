# 研究计划

## 课题背景

我们有一批土壤N2O排放量的数据集，课题目标是通过机器学习建模，预测N2O排放的时序特征，以及各种变量对其的影响。建模方案是RNN、随机森林进行对比。

## 数据集介绍

### 数据集基本信息

原始数据集位于 `datasets/data_EUR_raw.csv` 

数据集包含以下字段：

```py
numeric_static_features = {
   'Clay': '粘土含量',
   'CEC': '阳离子交换容量',
   'BD': '土壤容重',
   'pH': 'pH',
   'SOC': '有机碳含量',
   'TN': '总氮含量',
}

numeric_dynamic_features = {
   'Temp': '温度',
   'Prec': '降水量',
   'ST': '土壤温度',
   'WFPS': '土壤含水量',
   'Split N amount': '上次施肥量',
   'ferdur': '该次测量距上次施肥的天数',
}

classification_static_features = {
   'crop_class': '作物类型',
}

classification_dynamic_features = {
   'fertilization_class': '上次施肥类型',
   'appl_class': '上次施肥方式',
}

group_variables = { # 这些字段不应该参与训练
   'No. of obs': 'ID',
   'Publication': '文献编号',
   'control_group': '组号',
   'sowdur': '该次测量到播种之间的天数',
}

drop_variables = { # 下面这些字段应该全部丢掉
   'NH4+-N',
   'NO3_-N',
   'MN',
   'C/N',
}

labels = {
    'Daily fluxes': 'N2O排放通量',
}
```

数据集基本统计信息如下：

- 数值变量统计信息：
                  count    mean     std    min    25%     50%      75%      max  missing_ratio (%)
TN              24144.0   1.684   2.535   0.20   1.00   1.300    1.600    33.60             33.138
pH              36110.0   6.993   0.844   4.00   6.40   6.900    7.700     8.60              0.000
Daily fluxes    36110.0  15.262  55.697 -84.64   0.99   3.320   10.230  2433.88              0.000
Temp            36110.0  12.663   6.750 -16.30   8.10  12.600   17.200    31.40              0.000
WFPS            36110.0  56.345  18.929   2.00  43.00  57.000   71.000   128.00              0.000
Split N amount  36110.0  71.710  82.474   0.00   0.00  60.000  110.000   984.00              0.000
ST              36110.0  13.311   7.135 -11.90   8.30  13.100   18.000    51.10              0.000
CEC             36110.0  15.996   8.240   3.00  13.00  16.000   16.000    87.00              0.000
ferdur          36110.0  35.180  50.417  -1.00  -1.00  15.000   52.000   367.00              0.000
BD              36110.0   1.358   0.162   0.30   1.30   1.400    1.400     1.80              0.000
SOC             36110.0  18.691  30.986   2.20   9.30  12.900   18.200   412.00              0.000
Clay            36110.0  22.392  11.462   2.00  13.00  21.000   28.000    64.00              0.000
Prec            36110.0   2.456   6.819   0.00   0.00   0.000    1.800   262.50              0.000

- 分类变量统计信息：

                     count unique      top   freq  missing_ratio (%)
crop_class           36110     11    wheat  11004                0.0
appl_class           36110      3  surface  22665                0.0
fertilization_class  36110      9       AN   8239                0.0

（这里需要根据分类变量的unique数量，在RNN中选择合适的Embedding大小。

### 数据预处理

- 去除冗余的字段，即上文中的 `drop_variables`
- 将 `ferdur` 为 -1 的值（表示还未施肥）全部修改为 0 
- 按照 `(Publication, control_group, sowdur)` 的顺序升序排列，这三个字段用于确定序列，`(Publication, control_group)` 相同则表示同一个序列，然后序列内按照 `sowdur` 升序确定顺序。
- `TN` 字段存在缺失值，需要在同一个序列中采用前向填充
- 所有的分类变量均需要先使用 `LabelEncoder` 编码成数值，并将编码器保存在 `preprocessor/encoders.pkl` 中
- 构建原始时间序列数据，每一个序列的数据结构为：

```json
{
   "seq_id": ["Publication", "control_group"], // 序列的 ID
   "seq_length": "The length of sequence.", // 序列的长度
   "No. of obs": ["All No. of obs of samples in the sequence"], // 一维数组，包含了序列中每个样本的 `No. of obs`，该值作为样本的 ID
   "sowdurs": ["All sowdur of samples in the sequence"], // 一维数组，包含了序列中每个 `sowdur`，辅助作用
   "numeric_static": ["Clay", "CEC", "BD", "pH", "SOC", "TN"], // 静态数值特征，同一个序列中的样本有相同的值，因此只保存一份，一维数组
   "numeric_dynamic": [["Temp", "Prec", "ST", "WFPS", "Split N amount", "ferdur"], "..."], // 动态数值特征，随时间变化的特征，因此是个二维数组
   "categorical_static": ["crop_class"], // 静态分类特征，同样只保存一份，一维数组
   "categorical_dynamic": [["fertilization_class", "appl_class"], "..."], //动态分类特征，随时间变化的特征，二维数组
   "targets": ["Daily fluxes"] // 标签（N2O排放量）
}
```

- 输出处理后的数据集到 `datasets/data_EUR_processed.pkl` 中，其中存放一个 `list[dict]` 对象。
- 代码创建到 `src/n2o_pred/preprocessing.py` 中

## 模型选择方案

进行三种建模方案的对比：

- 直接使用随机森林建模，作为 Baseline
- 将每个测量的点（序列中的每个点）作为RNN的一个step进行建模，特点是观测事件驱动，每个step对应一次N2O排放的观测
- 将每一天作为RNN的一个step进行建模，特点是时间驱动，每个step对应一天

RNN的默认参数为 2 层 GRU/LSTM（默认为GRU），隐藏层大小96

### RNN模型的实现细节

- 两种RNN方案的区别在于数据输入不同，模型是一样的，因此可以通用模型的代码：

```python
class N2OPredictorRNN(nn.Module):

   def __init__(self, ...): ...

   def forward(self, ...): ...

   def count_parameters(self) -> int: ...
```

基本结构为：
- 静态和动态的分类特征分别通过一个 Embedding 层进行嵌入
- 静态数值特征和静态分类特征拼接起来，输入到一个 MLP 中，输出为 hidden_size * num_layers，作为RNN模型的初始 hidden state 输入
- 动态数值特征和动态分类特征拼接起来，经过一个投影层变换后，输入到 RNN 模型中

将模型代码创建在 `src/n2o_pred/rnn.py` 中。

### 数据类实现

所有的数据类都创建在 `src/n2o_pred/dataset.py` 中。

首先是一个基本的数据类，用来加载 `datasets/data_EUR_processed.pkl` 数据：

```python
class BaseN2ODataset(Dataset):
   seq_data_path = Path(__file__).parents[2] / 'datasets/data_EUR_processed.pkl'

   def __init__(self, ...): ...

   def flatten_to_dataframe(self) -> pd.DataFrame: ... # 随机森林不需要torch格式的数据集，因此可以用这个方法将序列展开，重新变成表格数据
```

可以从这个数据集统一进行训练集和测试集的划分（针对序列进行划分，而不是原始的样本点）：

```python
base_data = BaseN2ODataset()
train_data, test_data = train_test_split(base_data, random_state=RANDOM_STATE)
train_rf = train_data.flatten_to_dataframe()
test_rf = test_data.flatten_to_dataframe()

train_rnn_obs = N2ODatasetForObsStepRNN(train_data, ...)
test_rnn_obs = N2ODatasetForObsStepRNN(test_data, ...)

train_rnn_daily = N2ODatasetForDailyStepRNN(train_data, ...)
test_rnn_daily = N2ODatasetForDailyStepRNN(test_data, ...)
```

两个 RNN 模型的数据集：

```python
class N2ODatasetForObsStepRNN(Dataset):
   def __init__(self, base_data, ...): ...

class N2ODatasetForDailyStepRNN(Dataset):
   def __init__(self, base_data, ...): ...
```

对于使用观测步长的RNN模型，需要计算时间间隔 `time_delta = sowdur[t] - sowdur[t-1]`（首个时间步为0），并将其添加到动态数值变量中。

对于使用时间步长的RNN模型，需要对原始序列进行填充，确保每天一个点，对于这些填充的点，需要格外注意的地方如下：

- `Prec`（日降水量）：填充为 0（非观测日无降水记录），其他数值特征（Temp, ST, WFPS等）：线性插值
- `ferdur` 在还没施肥时均为0，而在施肥后表示距离上次施肥的天数，这应该通过计算来得到
- `Split N amount` 保持为"上次施肥量"的原始意义，在填充的天数中使用前向填充，比如序列中是0,20,20,20,50,50,50，填充后仍然是0,20,20,20,50,50,50
- 静态特征均使用前向填充（保持序列中一致）
- 需要记录真实测量点和插值点的位置（使用掩码），在模型训练时，应该使用掩码损失函数，不对插值的点计算损失值（因此排放量插值是没有实际用处的）

### 数据特征工程

- `Daily fluxes`：先使用 Symlog 转换，再使用 StandardScaler
- `Prec`, `Split N amount`, `ferdur`：先使用 log(x+1) 转换，再使用 StandardScaler
- 其他数值特征均使用 StandardScaler 进行缩放

注意点：

- RNN在计算评估指标 R2，RMSE 时，必须把模型输出缩放回原来的空间，否则没法和随机森林比较，但是在计算梯度和反向传播时，务必使用缩放处理后的值。
- 特征工程缩放应该基于真实值，比如 StandardScaler 不能在插值上 fit，但是要 transform 到所有值上。
- 随机森林的数据不需要进行特征工程

将预处理器保存在每一次实验的目录下。

## 模型训练和实验

### 模型训练

在 `src/n2o_pred/trainer.py` 中创建训练函数：

```python
@dataclass
class RNNTrainConfig:
   # 训练RNN相关的超参数
   ...

def train_rnn_predictor(model, train_data, test_data, config, ...):...

def train_rf_predictor(mode, ...): ...
```

- 需要设置早停策略（`patience=max_epoch//10`），最大训练轮次设置为 `max_epoch=300`
- 使用正则化、Dropout等防止过拟合的措施，注意数值稳定性问题

### 模型评估

- R2
- RMSE
- MAE
- SHAP分析（注意效率，比如可以使用GPU推理，同时注意使用合适的kernel）
- 其他你能想到的可以对比的指标

注意：RNN模型在计算这些指标时，需要将模型输出值转换回原始空间，这些指标必须能在RNN模型和RF模型之间比较。

施肥理论上应该对N2O的排放产生巨大的影响，SHAP分析应该可以提炼出这种影响。

### 实验

在模型训练阶段，需要尝试使用多种seed进行train-test-split（默认划分比例为9:1），然后从中选择出对每一种建模方案最优的划分方式：

例如：
```python
class ExperimentRunner:
   def __init__(self, ...):
      ...

   def run(self, total_split=20):
```

运行结果保存在 `outputs/exp_{datetime}/split_{seed}` 下，其中应包含：

- `figs/`: 可视化图表，包括训练损失图（`train_loss_curve.png`），预测值和实际值的对比（`predictions_vs_actual.png`，区分训练集和测试集），SHAP分析特征相对重要性（`feature_importance.png`），几个预测结果良好（R2较高）的长序列（序列长度大于15）的时序预测图（`sequence_predictions_{seq_id}.png`），图中的文字请使用英文。
- `tables/`: 和上述图所对应的csv表格，记录具体的数据
- 模型评估结果（`metrics.json`）
- 模型文件
- 训练配置
- 数据预处理器（StandardScaler在训练集上fit，因此与数据集划分方式相关，应该保存到结果中）
- 其他你能想到的文件

在 `outputs/exp_{datetime}` 目录下应该有：

- 实验总结（`summary.json`）：包含模型表现最好的一个 split 对应的种子、所有使用的种子、模型评估结果等等（尽可能的详细）
- 日志文件

## TUI设计和其他功能

`pyproject.toml` 中记录了程序入口，修改 `src/n2o_pred/cli.py` 中的 `main` 函数，合理设计命令行参数（TUI）

TUI设计应该尽可能灵活，比如我至少需要如下功能：

```bash
uv run n2o-pred data --preprocessing # 预处理数据集
uv run n2o-pred train --model rnn-daily --seed 42 # 使用42为随机种子划分数据集，然后训练rnn-daily模型
uv run n2o-pred train --model rnn-daily --max-split 50 # 使用50个随机种子分别划分数据集并进行训练
uv run n2o-pred train --model rnn-obs --seed-from outputs/exp_20251112/summay.json # 使用与改实验完全相同的所有种子训练另一个模型
uv run n2o-pred compare --models outputs/exp_1 outputs/exp2 # 两个实验使用完全相同的种子，该命令比较两个模型的，生成比较结果（表格，可视化图）
uv run n2o-pred compare --models outputs/exp_1 outputs/exp2 outputs/exp3 # 比较三个模型
uv run n2o-pred predict --model outputs/exp_xxx --datasets path2data # 在新的数据集上做预测
```

还有一些辅助的参数：

- `--device cuda:0` 指定设备
- `--split-seed 42` 当使用 `--max-split` 时，可以用来指定生成seed的随机数种子（控制种子的随机性）
- `--output outputs/xxx` 指定输出位置


## 代码风格

- 尽量使用组合而不要使用继承
- 模块化设计，便于修改
- 以上所有规定的接口请根据需要，灵活修改
- 使用 `Path` 处理路径，避免依赖运行时所处位置
- 如果需要执行代码，请永远使用 `uv run` 命令，程序入口为 `uv run n2o-pred`

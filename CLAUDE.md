# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**N2O-pred-g3p** is a machine learning-based soil nitrous oxide (N₂O) emission prediction system. The project compares three different modeling approaches to predict N₂O emissions from soil based on various environmental and agricultural features.

## Package Management

Always use `uv` as the package manager and `uv run` to run Python programs. The entry point is defined in `pyproject.toml` as `n2o-pred = "n2o_pred.cli:main"`.

```bash
# Install dependencies
uv sync

# Run the CLI
uv run n2o-pred --help
```

## Common Commands

### Data Preprocessing
```bash
uv run n2o-pred data --preprocessing
```

### Training Models
```bash
# Random Forest (single seed)
uv run n2o-pred train --model rf --seed 42

# Random Forest (multiple seeds)
uv run n2o-pred train --model rf --max-split 20

# RNN Observation-step
uv run n2o-pred train --model rnn-obs --seed 42 --device cuda:0

# RNN Daily-step
uv run n2o-pred train --model rnn-daily --seed 42 --device cuda:0

# Load seeds from previous experiment
uv run n2o-pred train --model rnn-daily --seed-from outputs/exp_xxx/summary.json
```

### Model Comparison
```bash
uv run n2o-pred compare --models outputs/exp1 outputs/exp2 outputs/exp3 --output outputs/comparison
```

### Prediction
```bash
# On processed data
uv run n2o-pred predict --model outputs/exp_xxx/split_42 --dataset datasets/data_EUR_processed.pkl

# On TIF/GeoTIFF files (batch prediction)
uv run n2o-pred predict --model outputs/exp_xxx/split_42 --dataset input_dir --output output_dir
```

## Model Types

| Model | Description |
|-------|-------------|
| **rf** | Random Forest - baseline model using tabular data |
| **rnn-obs** | Observation-step RNN - each observation is a time step, includes `time_delta` |
| **rnn-daily** | Daily-step RNN - each day is a time step with interpolation, masked loss on measured points |

## Code Architecture

### Key Modules

- **`cli.py`** - Command-line interface with four main commands: `data`, `train`, `compare`, `predict`
- **`preprocessing.py`** - Data preprocessing: feature grouping, missing value handling, categorical encoding
- **`dataset.py`** - Dataset classes for all model types, including TIF data loading
- **`rnn.py`** - RNN model architecture (`N2OPredictorRNN`) with embedding layers, static MLP, and RNN layers
- **`rf.py`** - Random Forest model wrapper
- **`trainer.py`** - Training logic and configurations (early stopping, learning rate scheduling, gradient clipping)
- **`evaluation.py`** - Evaluation metrics (R², RMSE, MAE, MRE) and visualization
- **`experiment.py`** - Experiment manager for multi-seed training with parallel execution
- **`compare.py`** - Model comparison across experiments with common seeds
- **`predict.py`** - Prediction utilities for both processed data and TIF files
- **`utils.py`** - Utility functions (logging, seed setting, transforms)

### Key Design Points

1. **Masked Loss for Daily RNN**: Only computes loss on actually measured points, ignoring interpolated points
2. **Inverse Transformation**: RNN metrics are computed in original (unscaled) space for direct comparison with RF
3. **Multi-seed Training**: Supports training with multiple random splits for robust results
4. **Experiment Output Organization**: Automatic directory structure by timestamp and seed
5. **Geospatial Prediction**: Built-in support for GeoTIFF files with sliding window processing

### Feature Transformations

- `Daily fluxes` (target): Symlog transformation + StandardScaler
- `Prec`, `Split N amount`, `ferdur`: log(x+1) transformation + StandardScaler
- Other numeric features: StandardScaler
- Categorical features: LabelEncoding + Embedding layers (for RNN)

## Output Structure

```
outputs/exp_{timestamp}/
├── summary.json
├── experiment.log
└── split_{seed}/
    ├── figs/                  # Loss curves, predictions, feature importance
    ├── tables/                # CSV predictions and feature importance
    ├── metrics.json           # R², RMSE, MAE, MRE
    ├── config.json
    ├── scalers.pkl            # For RNN
    └── model.pkl / best_model.pt
```

## Python Version

Requires Python >= 3.12 (specified in `.python-version` and `pyproject.toml`).

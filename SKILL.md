---
name: autoresearch
description: Autonomous LLM training research framework with multi-platform support (CUDA/MLX/MPS/CPU)
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: machine-learning
---

# Autoresearch Skill

## 概述

这是一个自主 LLM 训练研究框架，让 AI 代理可以自主进行模型训练实验。

## 支持平台

| 平台 | 运行命令 | 加速方式 |
|------|----------|----------|
| NVIDIA GPU | `uv run train.py` | Flash Attention 3 |
| Apple Silicon (MLX) | `python mlx_train.py` | MLX 原生 |
| Apple Silicon (MPS) | `uv run train.py` | PyTorch SDPA |
| CPU | `uv run train.py` | 通用 |

## 快速开始

### 向导模式（推荐）

首次使用建议运行交互式向导：
```bash
python wizard.py
```

### NVIDIA GPU
```bash
uv sync
uv run prepare.py
uv run train.py
```

### Apple Silicon (MLX)
```bash
pip install mlx
uv sync
uv run prepare.py
python mlx_train.py
```

### Apple Silicon (MPS)
```bash
uv sync
uv run prepare.py
uv run train.py  # 自动检测 MPS
```

## 项目结构

```
├── prepare.py      # 数据准备、tokenizer（不可修改）
├── train.py        # PyTorch 训练脚本
├── mlx_train.py   # MLX 训练脚本
├── wizard.py       # 交互式实验向导
├── ml/            # 机器学习工具包
│   ├── datasets/  # 数据集加载
│   ├── models/    # 模型定义 (MLP/CNN/LSTM)
│   ├── metrics/   # 评估指标
│   └── tasks/     # 训练任务
├── mlx/           # MLX 模型实现
│   └── model.py   # GPT 模型（MLX 版本）
├── program.md     # 代理指令
├── pyproject.toml # 依赖
└── SKILL.md       # OpenCode 技能文件
```

## 核心文件说明

### prepare.py
- 固定常量、数据准备、tokenizer 训练
- 运行时工具（dataloader、评估）
- **不可修改**

### train.py
- PyTorch GPT 模型
- Muon + AdamW 优化器
- 训练循环
- **代理修改此文件**

### mlx_train.py
- MLX GPT 模型（Apple Silicon 优化）
- **独立脚本**

### ml/ 工具包
- `TabularDataset` / `TimeSeriesDataset` / `ImageDataset` - 数据集
- `MLP` / `CNN` / `LSTMModel` - 模型
- `accuracy` / `f1_score` / `rmse` / `mae` - 指标
- `train_classification` / `train_regression` / `train_forecast` - 训练函数

## 超参数调优

在 Apple Silicon 上运行建议调整：
```python
DEPTH = 4              # 原 8
TOTAL_BATCH_SIZE = 2**17  # 原 2^19
DEVICE_BATCH_SIZE = 32     # 原 128
```

## 自主实验模式

1. 创建实验分支：`git checkout -b autoresearch/<tag>`
2. 修改 `train.py`
3. 运行实验：`uv run train.py > run.log 2>&1`
4. 提取结果：`grep "^val_bpb:" run.log`
5. 记录到 `results.tsv`
6. 决定保留或回滚

训练固定 5 分钟时间预算，指标为 val_bpb（越低越好）。

---
name: autoresearch
description: Autonomous LLM training research framework - 支持 CUDA/MLX/MPS/CPU
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: machine-learning
---

# Autoresearch

自主 LLM 训练研究框架，让 AI 代理可以自主进行模型训练实验。

## 支持平台

| 平台 | 命令 | 加速 |
|------|------|------|
| NVIDIA GPU | `uv run train.py` | Flash Attention 3 |
| Apple Silicon (MLX) | `python mlx_train.py` | MLX |
| Apple Silicon (MPS) | `uv run train.py` | PyTorch SDPA |
| CPU | `uv run train.py` | 通用 |

## 快速开始

```bash
# 方式 1: 向导模式（推荐）
python wizard.py

# 方式 2: 手动
uv sync
uv run prepare.py
uv run train.py
```

## 项目结构

```
├── prepare.py      # 数据准备（不可修改）
├── train.py        # PyTorch 训练
├── mlx_train.py   # MLX 训练
├── wizard.py       # 交互式向导
├── ml/            # ML 工具包
├── program.md     # 代理指令
└── SKILL.md       # 本文件
```

## 核心文件

- **train.py** - 代理修改的主要训练脚本
- **mlx_train.py** - Apple Silicon 专用
- **ml/** - 分类/回归/时序工具包

## 实验模式

1. `git checkout -b autoresearch/<tag>`
2. 修改 train.py
3. `uv run train.py > run.log 2>&1`
4. `grep "^val_bpb:" run.log`
5. 记录到 results.tsv
6. 保留或回滚

训练固定 5 分钟，val_bpb 越低越好。

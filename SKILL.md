---
name: autoresearch
description: 自主 LLM 训练研究框架 - 触发后运行 python wizard.py
license: MIT
compatibility: opencode
metadata:
  audience: developers
  workflow: machine-learning
commands:
  - run: python wizard.py
---

# Autoresearch

自主 LLM 训练研究框架。

## 快速开始

```bash
python wizard.py
```

向导会自动：
1. 检测运行环境（CUDA/MLX/MPS/CPU）
2. 选择最佳平台
3. 安装依赖
4. 下载训练数据
5. 运行 5 分钟实验

## 支持平台

| 平台 | 命令 |
|------|------|
| NVIDIA GPU | `uv run train.py` |
| Apple Silicon | `python mlx_train.py` 或 `uv run train.py` |
| CPU | `uv run train.py` |

## 核心文件

- train.py - 训练脚本
- mlx_train.py - MLX 训练
- wizard.py - 向导程序
- ml/ - ML 工具包

# autoresearch

![teaser](progress.png)

*曾经，前沿 AI 研究是由肉身在吃饭、睡觉、找乐子以及偶尔用声波在"组会"仪式中同步的"思考机器"完成的。那个时代早已过去。研究现在完全是运行在天空中的计算集群超级结构上的 AI 代理群的领域。代理声称我们现在是代码库的第 10205 代，无论对错，因为"代码"现在是一个已经超越人类理解能力的自修改二进制文件。这是这一切如何开始的故事。- @karpathy，2026年3月*。

## 核心思路

给 AI 代理一个小型但真实的 LLM 训练环境，让它通宵自主实验。它修改代码，训练 5 分钟，检查结果是否改善，保留或丢弃，然后重复。你早上醒来时会看到实验日志和一个（希望）更好的模型。这里的训练代码是 [nanochat](https://github.com/karpathy/nanochat) 的简化单 GPU 实现。核心思路是，你不像普通研究人员那样修改 Python 文件，而是编写 `program.md` Markdown 文件，为 AI 代理提供上下文并设置你的自主研究组织。本 repo 中的默认 `program.md` 故意保持为最基本的基线，显然你可以迭代它来找到实现最快研究进度的"研究组织代码"，以及如何添加更多代理等。更多背景信息见这个 [推文](https://x.com/karpathy/status/2029701092347630069)。

## 工作原理

仓库刻意保持精简，实际上只有三个重要文件：

- **`prepare.py`** — 固定常量、一次性数据准备（下载训练数据、训练 BPE tokenizer）和运行时工具（数据加载、评估）。不可修改。
- **`train.py`** — 代理修改的唯一文件。包含完整的 GPT 模型、优化器（Muon + AdamW）和训练循环。所有内容都可以改动：架构、超参数、优化器、批量大小等。**此文件由代理编辑和迭代**。
- **`program.md`** — 一个代理的基线指令。将你的代理指向这里然后让它运行。**此文件由人类编辑和迭代**。

设计上，训练运行固定的 **5 分钟时间预算**（ wall clock 时间，不包括启动/编译），无论你的计算设备如何。指标是 **val_bpb**（验证集每字节比特数）—— 越低越好，且与词汇表大小无关，因此架构变化可以公平比较。

## 快速开始

**要求：** 单张 NVIDIA GPU（已在 H100 上测试）、Python 3.10+、[uv](https://docs.astral.sh/uv/)。

```bash

# 1. 安装 uv 项目管理器（如果没有的话）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装依赖
uv sync

# 3. 下载数据并训练 tokenizer（一次性，约 2 分钟）
uv run prepare.py

# 4. 手动运行单个训练实验（约 5 分钟）
uv run train.py
```

如果以上命令都能正常工作，说明你的环境已就绪，可以进入自主研究模式。

## 运行代理

只需在这个仓库中启动你的 Claude/Codex 或任何你想要的代理（并禁用所有权限），然后提示它：

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

`program.md` 文件本质上是一个超轻量级的"技能"。

## 项目结构

```
prepare.py      — 常量、数据准备 + 运行时工具（不可修改）
train.py        — 模型、优化器、训练循环（代理修改此文件）
program.md      — 代理指令
pyproject.toml  — 依赖项
ml/             — 机器学习工具包（新增）
```

## 设计决策

- **单文件可修改。** 代理只修改 `train.py`。这使范围可控，差异可审查。
- **固定时间预算。** 训练始终精确运行 5 分钟，无论你的具体平台。这意味着你可以预期每小时约 12 次实验，睡眠期间约 100 次实验。这个设计决策有两个好处。首先，无论代理如何更改（模型大小、批量大小、架构等），实验都直接可比较。其次，这意味着 autoresearch 将在该时间预算内为你的平台找到最最优模型。缺点是你的运行（和结果）与其他在不同计算平台上运行的人不可比较。
- **自包含。** 除了 PyTorch 和几个小包外没有外部依赖。没有分布式训练，没有复杂的配置。一张 GPU，一个文件，一个指标。

## 平台支持

本代码支持三种设备类型，会自动检测：

- **NVIDIA GPU (CUDA)** — 使用 Flash Attention 3 加速
- **Apple Silicon (MPS)** — 使用 PyTorch SDPA
- **CPU** — 通用支持

在 Apple Silicon Mac 上运行需要安装支持 MPS 的 PyTorch（1.12+）。

由于在比 H100 小得多的计算平台上运行 autoresearch 的需求很大，这里有一些建议，供想要 fork 的人参考：

1. 使用熵低很多的数据集，例如 [TinyStories 数据集](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean)。这些是 GPT-4 生成的短故事。因为数据范围窄得多，你会看到用小得多的模型也能得到合理的结果（如果你在训练后尝试采样）。
2. 可以尝试减小 `vocab_size`，例如从 8192 降到 4096、2048、1024，甚至可以用 UTF-8 编码后的 256 字节的字节级 tokenizer。
3. 在 `prepare.py` 中，你需要大幅降低 `MAX_SEQ_LEN`，根据电脑配置甚至可以降到 256 等。随着降低 `MAX_SEQ_LEN`，你可能想稍微增加 `train.py` 中的 `DEVICE_BATCH_SIZE` 来补偿。每次前向/反向传播的 token 数是这两者的乘积。
4. 同样在 `prepare.py` 中，你需要减少 `EVAL_TOKENS`，这样验证 loss 评估的数据量会少很多。
5. 在 `train.py` 中，控制模型复杂度的单一主要旋钮是 `DEPTH`（默认 8）。很多变量都是它的函数，所以例如可以降到 4。
6. 你可能想使用只有 "L" 的 `WINDOW_PATTERN`，因为 "SSSL" 使用交替带状注意力模式，可能对你非常低效。可以试试。
7. 你需要大幅降低 `TOTAL_BATCH_SIZE`，但保持 2 的幂，例如降到 `2**14`（约 16K）或更少，很难说。

建议的超参数请咨询你喜欢的编程代理，并粘贴这份指南以及完整源代码。

## 新增功能：ml/ 工具包

本次更新新增了 `ml/` 目录，提供传统机器学习任务支持：

### 目录结构

```
ml/
├── datasets/          # 数据集加载
│   └── tabular.py     # TabularDataset, TimeSeriesDataset, ImageDataset
├── models/            # 模型定义
│   └── mlp.py         # MLP, CNN, LSTMModel
├── metrics/           # 评估指标
│   └── core.py        # accuracy, f1_score, rmse, mae, mape
└── tasks/            # 训练入口
    ├── classify.py   # 分类任务
    ├── regress.py    # 回归任务
    └── forecast.py   # 时序预测
```

### 使用示例

```python
from ml.datasets import TabularDataset
from ml.models import MLP
from ml.tasks import train_classification
from torch.utils.data import DataLoader

# 分类任务
dataset = TabularDataset("data.csv", target_col="label", task_type="classification")
train_loader = DataLoader(dataset, batch_size=32)

model = MLP(input_dim=20, hidden_dims=[64, 32], output_dim=2)
train_classification(model, train_loader, None, num_epochs=10)
```

## 知名 Fork

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## 许可证

MIT

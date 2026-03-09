#!/usr/bin/env python3
"""
Autoresearch Wizard - 交互式实验向导
自动检测环境并指引用户完成实验
"""

import os
import sys
import subprocess
import platform


PLATFORMS = {
    "cuda": {
        "name": "NVIDIA GPU (CUDA)",
        "train_cmd": "uv run train.py",
        "prepare_cmd": "uv run prepare.py",
        "require": "NVIDIA GPU",
    },
    "mlx": {
        "name": "Apple Silicon (MLX)",
        "train_cmd": "python mlx_train.py",
        "prepare_cmd": "uv run prepare.py",
        "require": "Apple Silicon Mac + pip install mlx",
    },
    "mps": {
        "name": "Apple Silicon (MPS)",
        "train_cmd": "uv run train.py",
        "prepare_cmd": "uv run prepare.py",
        "require": "Apple Silicon Mac",
    },
    "cpu": {
        "name": "CPU",
        "train_cmd": "uv run train.py",
        "prepare_cmd": "uv run prepare.py",
        "require": "通用",
    },
}


def print_step(step, title):
    print(f"\n{'=' * 60}")
    print(f"  步骤 {step}: {title}")
    print('=' * 60)


def detect_environment():
    """自动检测运行环境"""
    print_step(1, "检测运行环境")
    
    env = {
        "os": platform.system(),
        "arch": platform.machine(),
        "has_cuda": False,
        "has_mps": False,
        "has_mlx": False,
        "has_torch": False,
        "torch_version": None,
    }
    
    print(f"\n操作系统: {env['os']} ({env['arch']})")
    
    try:
        import torch
        env["has_torch"] = True
        env["torch_version"] = torch.__version__
        print(f"PyTorch: {torch.version}")
        
        if torch.cuda.is_available():
            env["has_cuda"] = True
            print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            env["has_mps"] = True
            print(f"✓ MPS (Apple Silicon)")
    except ImportError:
        print("✗ PyTorch 未安装")
    
    try:
        import mlx
        env["has_mlx"] = True
        print(f"✓ MLX (Apple Silicon)")
    except ImportError:
        pass
    
    return env


def select_best_platform(env):
    """根据环境选择最佳平台"""
    print_step(2, "选择运行平台")
    
    if env["has_cuda"]:
        platform_type = "cuda"
    elif env["has_mlx"]:
        platform_type = "mlx"
    elif env["has_mps"]:
        platform_type = "mps"
    else:
        platform_type = "cpu"
    
    p = PLATFORMS[platform_type]
    print(f"\n推荐平台: {p['name']}")
    print(f"训练命令: {p['train_cmd']}")
    print(f"要求: {p['require']}")
    
    return platform_type


def check_dependencies():
    """检查并安装依赖"""
    print_step(3, "检查依赖")
    
    print("\n检查 Python 包...")
    
    result = subprocess.run(["uv", "sync"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ 依赖安装完成")
        return True
    else:
        print(f"✗ 依赖安装失败: {result.stderr}")
        return False


def prepare_data():
    """准备训练数据"""
    print_step(4, "准备训练数据")
    
    cache_dir = os.path.expanduser("~/.cache/autoresearch/")
    
    if os.path.exists(cache_dir):
        files = os.listdir(cache_dir)
        if "train.bin" in files and "val.bin" in files:
            print(f"✓ 数据已存在: {cache_dir}")
            return True
    
    print("\n数据目录不存在，正在下载...")
    print("(首次运行约需 2 分钟)")
    
    result = subprocess.run(["uv", "run", "prepare.py"], capture_output=False)
    
    if result.returncode == 0:
        print("✓ 数据准备完成")
        return True
    else:
        print("✗ 数据准备失败")
        return False


def run_experiment(platform_type):
    """运行训练实验"""
    print_step(5, "运行训练实验")
    
    p = PLATFORMS[platform_type]
    cmd = p["train_cmd"]
    
    print(f"\n执行命令: {cmd}")
    print("训练将运行 5 分钟...\n")
    
    result = subprocess.run(cmd.split(), capture_output=False)
    
    if result.returncode == 0:
        print("\n✓ 训练完成!")
        
        try:
            with open("run.log", "r") as f:
                for line in f:
                    if line.startswith("val_bpb:"):
                        print(f"\n{line.strip()}")
                    elif line.startswith("total_tokens_M:"):
                        print(f"{line.strip()}")
                    elif line.startswith("peak_vram_mb:"):
                        print(f"{line.strip()}")
        except:
            pass
        
        return True
    else:
        print("✗ 训练失败")
        return False


def show_next_steps():
    """显示后续步骤"""
    print_step(6, "后续步骤")
    
    print("""
恭喜完成第一次实验!

后续实验可以:
1. 修改 train.py 调整超参数（如 DEPTH, LR, BATCH_SIZE）
2. 再次运行: uv run train.py
3. 比较 val_bpb 结果

常用超参数调整:
- DEPTH: 模型层数（默认 8，建议 4-6）
- TOTAL_BATCH_SIZE: 批大小（建议 2^17）
- DEVICE_BATCH_SIZE: 设备批大小（根据显存调整）

指标 val_bpb 越低越好！
""")


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    Autoresearch 实验向导                      ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    print("正在检测环境，请稍候...")
    
    env = detect_environment()
    
    platform_type = select_best_platform(env)
    
    if not check_dependencies():
        print("\n依赖安装失败，请手动运行: uv sync")
        return
    
    if not prepare_data():
        print("\n数据准备失败，请检查网络连接")
        return
    
    print("\n开始训练...")
    run_experiment(platform_type)
    
    show_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已退出向导")
        sys.exit(0)
    except Exception as e:
        print(f"\n错误: {e}")
        sys.exit(1)

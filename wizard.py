#!/usr/bin/env python3
"""
Autoresearch Wizard - 交互式实验向导
"""

import os
import sys
import subprocess

PLATFORMS = {
    "1": {"name": "NVIDIA GPU (CUDA)", "cmd": "uv run train.py", "desc": "需要 NVIDIA GPU"},
    "2": {"name": "Apple Silicon (MLX)", "cmd": "python mlx_train.py", "desc": "需要 Apple Silicon Mac + pip install mlx"},
    "3": {"name": "Apple Silicon (MPS)", "cmd": "uv run train.py", "desc": "需要 Apple Silicon Mac，PyTorch 自动检测"},
    "4": {"name": "CPU", "cmd": "uv run train.py", "desc": "通用，但速度较慢"},
}


def print_header(title):
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)


def print_step(step, title):
    print(f"\n{'=' * 50}")
    print(f"  步骤 {step}: {title}")
    print("=" * 50)


def check_environment():
    print_step(1, "检查环境")
    
    print("\n检测到的环境:")
    
    has_cuda = False
    has_mps = False
    has_mlx = False
    
    try:
        import torch
        if torch.cuda.is_available():
            has_cuda = True
            print(f"  ✓ CUDA: {torch.cuda.get_device_name(0)}")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            has_mps = True
            print(f"  ✓ MPS (Apple Silicon)")
    except:
        pass
    
    try:
        import mlx
        has_mlx = True
        print(f"  ✓ MLX (Apple Silicon)")
    except:
        pass
    
    if not (has_cuda or has_mps or has_mlx):
        print(f"  ✓ CPU (仅支持推理)")
    
    return has_cuda, has_mps, has_mlx


def select_platform():
    print_step(2, "选择平台")
    
    print("\n请选择运行平台:")
    for key, platform in PLATFORMS.items():
        print(f"  {key}. {platform['name']}")
        print(f"     {platform['desc']}")
    
    while True:
        choice = input("\n请输入选项 (1-4): ").strip()
        if choice in PLATFORMS:
            return PLATFORMS[choice]
        print("无效选择，请重新输入")


def check_dependencies():
    print_step(3, "检查依赖")
    
    print("\n检查 Python 包...")
    
    missing = []
    
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except:
        missing.append("torch")
        print(f"  ✗ torch")
    
    try:
        import pandas
        print(f"  ✓ pandas")
    except:
        missing.append("pandas")
        print(f"  ✗ pandas")
    
    try:
        import numpy
        print(f"  ✓ numpy")
    except:
        missing.append("numpy")
        print(f"  ✗ numpy")
    
    if missing:
        print(f"\n缺少依赖，正在安装: {', '.join(missing)}")
        subprocess.run(["uv", "sync"], check=True)
        print("  ✓ 依赖安装完成")
    else:
        print("\n  ✓ 所有依赖已安装")
    
    return True


def prepare_data():
    print_step(4, "准备数据")
    
    cache_dir = os.path.expanduser("~/.cache/autoresearch/")
    
    if os.path.exists(cache_dir):
        print(f"\n数据目录已存在: {cache_dir}")
        response = input("是否重新下载数据? (y/N): ").strip().lower()
        if response != 'y':
            print("  ✓ 跳过数据下载")
            return True
    
    print("\n正在下载数据并训练 tokenizer...")
    print("(首次运行约需 2 分钟)")
    
    result = subprocess.run(["uv", "run", "prepare.py"], capture_output=False)
    
    if result.returncode == 0:
        print("  ✓ 数据准备完成")
        return True
    else:
        print("  ✗ 数据准备失败")
        return False


def run_experiment(platform):
    print_step(5, "运行实验")
    
    print(f"\n运行命令: {platform['cmd']}")
    print("\n按 Enter 开始训练 (5 分钟)...")
    input()
    
    print("\n开始训练...\n")
    
    result = subprocess.run(platform['cmd'].split(), capture_output=False)
    
    if result.returncode == 0:
        print("\n  ✓ 训练完成!")
        
        try:
            with open("run.log", "r") as f:
                for line in f:
                    if line.startswith("val_bpb:"):
                        print(f"\n{line.strip()}")
        except:
            pass
        
        return True
    else:
        print("  ✗ 训练失败")
        return False


def show_results():
    print_step(6, "查看结果")
    
    if os.path.exists("results.tsv"):
        print("\n实验结果:")
        with open("results.tsv", "r") as f:
            print(f.read())
    else:
        print("\nresults.tsv 尚不存在")
    
    print("\n恭喜完成第一次实验!")
    print("\n后续步骤:")
    print("  1. 修改 train.py 调整超参数")
    print("  2. 再次运行实验")
    print("  3. 查看 val_bpb 是否改善")


def main():
    print_header("Autoresearch 实验向导")
    
    print("""
欢迎使用 Autoresearch!

本向导将指引你完成:
  1. 检查运行环境
  2. 选择运行平台
  3. 安装依赖
  4. 准备训练数据
  5. 运行第一个实验
  6. 查看结果

按 Ctrl+C 可随时退出
""")
    
    input("按 Enter 继续...")
    
    has_cuda, has_mps, has_mlx = check_environment()
    
    platform = select_platform()
    
    check_dependencies()
    
    prepare_data()
    
    run_experiment(platform)
    
    show_results()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已退出向导")
        sys.exit(0)

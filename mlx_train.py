"""
MLX training script for Apple Silicon.
Usage: python mlx_train.py
"""

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import time
from dataclasses import dataclass, asdict
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx.model import GPT, GPTConfig

MAX_SEQ_LEN = 1024
TIME_BUDGET = 300

ASPECT_RATIO = 64
HEAD_DIM = 128
WINDOW_PATTERN = "SSSL"

TOTAL_BATCH_SIZE = 2**17
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
DEPTH = 4
DEVICE_BATCH_SIZE = 32

PEAK_FLOPS = 15e12


class Tokenizer:
    @staticmethod
    def from_directory():
        return SimpleTokenizer()


class SimpleTokenizer:
    def __init__(self):
        cache_dir = os.path.expanduser("~/.cache/autoresearch/")
        vocab_file = os.path.join(cache_dir, "tokenizer.json")
        if os.path.exists(vocab_file):
            import json
            with open(vocab_file) as f:
                data = json.load(f)
            self.encoder = data["model"]["vocab"]
            self.decoder = {v: k for k, v in self.encoder.items()}
        else:
            self.encoder = {}
            self.decoder = {}
    
    def get_vocab_size(self):
        return len(self.encoder) if self.encoder else 32768
    
    def encode(self, text):
        return [self.encoder.get(c, 0) for c in text]
    
    def decode(self, ids):
        return "".join([self.decoder.get(i, "") for i in ids])


def make_dataloader(tokenizer, batch_size, seq_len, split="train"):
    cache_dir = os.path.expanduser("~/.cache/autoresearch/")
    data_file = os.path.join(cache_dir, f"{split}.bin")
    
    if not os.path.exists(data_file):
        print(f"Warning: {data_file} not found. Using dummy data.")
        return DummyDataLoader(batch_size, seq_len)
    
    data = mx.fromfile(data_file, dtype=mx.uint16)
    
    class DataLoader:
        def __init__(self, data, batch_size, seq_len):
            self.data = data
            self.batch_size = batch_size
            self.seq_len = seq_len
            self.pos = 0
            self.epoch = 0
        
        def __iter__(self):
            return self
        
        def __next__(self):
            if self.pos + self.batch_size * (self.seq_len + 1) > len(self.data):
                self.pos = 0
                self.epoch += 1
            
            batch_idx = self.pos
            self.pos += self.batch_size
            
            x = mx.zeros((self.batch_size, self.seq_len), dtype=mx.int32)
            y = mx.zeros((self.batch_size, self.seq_len), dtype=mx.int32)
            
            for i in range(self.batch_size):
                start = batch_idx + i * self.seq_len
                x[i] = self.data[start:start + self.seq_len]
                y[i] = self.data[start + 1:start + self.seq_len + 1]
            
            return x, y, self.epoch
    
    return DataLoader(data, batch_size, seq_len)


def evaluate_bpb(model, tokenizer, batch_size):
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    total_loss = 0.0
    num_batches = 0
    
    for x, y, _ in val_loader:
        if num_batches >= 10:
            break
        loss = model(x, y)
        total_loss += loss.item()
        num_batches += 1
    
    if num_batches == 0:
        return 1.0
    
    avg_loss = total_loss / num_batches
    bpb = avg_loss / 0.693
    return bpb


def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )


t_start = time.time()
mx.random.seed(42)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")
print(f"Using device: mlx (Apple Silicon)")

config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

model = GPT(config)

num_params = sum(p.size for p in model.parameters())
print(f"Parameter count: {num_params:,}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = optim.AdamW(
    learning_rate=0.001,
    betas=(0.8, 0.95),
    weight_decay=0.0,
)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(iter(train_loader))

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")


def get_lr_multiplier(progress):
    if progress < 0.0:
        return 1.0
    elif progress < 0.5:
        return 1.0
    else:
        cooldown = (1.0 - progress) / 0.5
        return cooldown


t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0


def loss_fn(model, x, y):
    return model(x, y)


while True:
    t0 = time.time()
    
    for micro_step in range(grad_accum_steps):
        try:
            x, y, epoch = next(iter(train_loader))
        except StopIteration:
            train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
            x, y, epoch = next(iter(train_loader))
        
        loss, grads = mx.value_and_grad(loss_fn, argnums=None)(model, x, y)
        loss = loss / grad_accum_steps
        
        optimizer.update(model, grads)
    
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for group in optimizer.learning_rate_schedule:
        pass
    
    train_loss_f = float(loss)
    
    if train_loss_f > 100:
        print("FAIL")
        exit(1)
    
    t1 = time.time()
    dt = t1 - t0
    
    if step > 10:
        total_training_time += dt
    
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    remaining = max(0, TIME_BUDGET - total_training_time)
    
    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)
    
    step += 1
    
    if step > 10 and total_training_time >= TIME_BUDGET:
        break


print()

total_tokens = step * TOTAL_BATCH_SIZE

val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

t_end = time.time()

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     0.0")
print(f"mfu_percent:      0.00")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")

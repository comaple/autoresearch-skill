"""
MLX GPT Model for Apple Silicon training.
"""

import mlx.core as mx
import mlx.nn as nn
from dataclasses import dataclass


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


def rms_norm(x, eps=1e-6):
    return x * mx.rsqrt(x.square().mean(-1, keepdims=True) + eps)


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return mx.concatenate([y1, y2], axis=-1)


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        
        self.ve_gate_channels = 32
        if has_ve(layer_idx, config.n_layer):
            self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
        else:
            self.ve_gate = None

    def __call__(self, x, ve, cos_sin, window_size):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).reshape(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).reshape(B, T, self.n_kv_head, self.head_dim)

        if self.ve_gate is not None:
            ve = ve.reshape(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * mx.nn.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = rms_norm(q)
        k = rms_norm(k)

        if self.n_kv_head < self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = mx.repeat(k, n_rep, axis=2)
            v = mx.repeat(v, n_rep, axis=2)

        out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.reshape(B, T, -1)
        out = self.c_proj(out)
        return out


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = nn.functional.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def __call__(self, x, ve, cos_sin, window_size):
        x = x + self.attn(rms_norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(rms_norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.h = [Block(config, i) for i in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.resid_lambdas = mx.ones(config.n_layer)
        self.x0_lambdas = mx.ones(config.n_layer) * 0.1
        
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = {
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        }
        
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos = cos
        self.sin = sin

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        channel_range = mx.arange(0, head_dim, step=2, dtype=mx.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = t[:, None] * inv_freq[None, :]
        cos = mx.cos(freqs).astype(mx.bfloat16)
        sin = mx.sin(freqs).astype(mx.bfloat16)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def __call__(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cos.shape[1]
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.wte(idx)
        x = rms_norm(x)
        x0 = x
        for i, block in enumerate(self.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
        x = rms_norm(x)

        softcap = 15
        logits = self.lm_head(x)
        logits = logits.astype(mx.float32)
        logits = softcap * mx.tanh(logits / softcap)

        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), 
                targets.reshape(-1),
                reduction='mean'
            )
            return loss
        return logits

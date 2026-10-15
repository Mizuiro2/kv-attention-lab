from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def causal_mask(query_length: int, key_length: int, query_offset: int, device: torch.device) -> Tensor:
    """Right-offset causal visibility for a chunk following an existing KV prefix."""
    if query_offset < 0 or query_offset + query_length > key_length:
        raise ValueError("queries must correspond to positions inside the key sequence")
    query_positions = torch.arange(query_length, device=device) + query_offset
    key_positions = torch.arange(key_length, device=device)
    return key_positions[None, :] <= query_positions[:, None]


def attention(q: Tensor, k: Tensor, v: Tensor, backend: str = "sdpa", query_offset: int = 0) -> Tensor:
    """B,H,L,D attention. Boolean True means an allowed key in PyTorch SDPA."""
    if backend not in {"manual", "sdpa"}:
        raise ValueError(f"unknown backend: {backend}")
    mask = causal_mask(q.shape[-2], k.shape[-2], query_offset, q.device)
    if backend == "manual":
        scores = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        scores = scores.masked_fill(~mask, -torch.inf)
        return torch.softmax(scores, dim=-1) @ v
    # With L=1 and a longer cache, is_causal=True would use the upper-left
    # alignment and incorrectly hide past keys. Use absolute positions instead.
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    width: int = 128
    heads: int = 4
    layers: int = 2
    max_sequence: int = 512

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.width, self.heads, self.layers, self.max_sequence) <= 0:
            raise ValueError("model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")


class KVCache:
    """Preallocated dense K/V storage: writes do not concatenate or recopy the prefix."""

    def __init__(self, config: ModelConfig, batch: int, capacity: int, device: torch.device, dtype: torch.dtype):
        if capacity <= 0 or capacity > config.max_sequence:
            raise ValueError("cache capacity exceeds positional embedding range")
        shape = (batch, config.heads, capacity, config.width // config.heads)
        self.keys = [torch.empty(shape, device=device, dtype=dtype) for _ in range(config.layers)]
        self.values = [torch.empty(shape, device=device, dtype=dtype) for _ in range(config.layers)]
        self.capacity = capacity
        self.length = 0

    @property
    def allocated_tensor_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.keys + self.values)


class AttentionBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.width // config.heads
        self.norm1 = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.projection = nn.Linear(config.width, config.width, bias=False)
        self.norm2 = nn.LayerNorm(config.width)
        self.feed_forward = nn.Sequential(nn.Linear(config.width, 4 * config.width), nn.GELU(), nn.Linear(4 * config.width, config.width))

    def forward(self, hidden: Tensor, backend: str, cache: KVCache | None, layer: int, offset: int) -> Tensor:
        batch, length, width = hidden.shape
        q, k, v = self.qkv(self.norm1(hidden)).chunk(3, dim=-1)
        q, k, v = [t.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2) for t in (q, k, v)]
        if cache is not None:
            cache.keys[layer][:, :, offset:offset + length].copy_(k)
            cache.values[layer][:, :, offset:offset + length].copy_(v)
            k = cache.keys[layer][:, :, :offset + length]
            v = cache.values[layer][:, :, :offset + length]
        output = attention(q, k, v, backend, query_offset=offset if cache is not None else 0)
        hidden = hidden + self.projection(output.transpose(1, 2).contiguous().view(batch, length, width))
        return hidden + self.feed_forward(self.norm2(hidden))


class TinyCausalLM(nn.Module):
    """Randomly initialized two-layer decoder for systems correctness, not model quality."""

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.max_sequence, config.width)
        self.blocks = nn.ModuleList([AttentionBlock(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, config.vocab_size, bias=False)

    def new_cache(self, batch: int, capacity: int) -> KVCache:
        weight = self.token_embedding.weight
        return KVCache(self.config, batch, capacity, weight.device, weight.dtype)

    def forward(self, token_ids: Tensor, backend: str = "sdpa", cache: KVCache | None = None) -> Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("token_ids must have shape batch, positive length")
        offset = cache.length if cache is not None else 0
        end = offset + token_ids.shape[1]
        if end > self.config.max_sequence or (cache is not None and end > cache.capacity):
            raise ValueError("sequence exceeds cache/position capacity")
        if cache is not None and token_ids.shape[0] != cache.keys[0].shape[0]:
            raise ValueError("cache batch mismatch")
        positions = torch.arange(offset, end, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        for layer, block in enumerate(self.blocks):
            hidden = block(hidden, backend, cache, layer, offset)
        if cache is not None:
            cache.length = end
        return self.output(self.norm(hidden))


def full_recomputation(model: TinyCausalLM, tokens: Tensor, prompt: int, steps: int, backend: str) -> Tensor:
    return torch.stack([model(tokens[:, :prompt + i], backend=backend)[:, -1] for i in range(steps)], dim=1)


def cached_decode(model: TinyCausalLM, tokens: Tensor, prompt: int, steps: int, backend: str) -> Tensor:
    cache = model.new_cache(tokens.shape[0], prompt + steps)
    outputs = [model(tokens[:, :prompt], backend=backend, cache=cache)[:, -1]]
    for i in range(1, steps):
        outputs.append(model(tokens[:, prompt + i - 1:prompt + i], backend=backend, cache=cache)[:, -1])
    return torch.stack(outputs, dim=1)

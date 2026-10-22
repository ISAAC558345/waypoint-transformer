"""Dependency-light copy of the official causal Transformer primitives.

Source: ``src/wt/transformer.py`` in this repository.  The classes below are
kept behaviorally identical to the audited official ``GPTConfig``,
``CausalSelfAttention`` and ``Block`` implementations.  They are isolated here
only because importing the legacy module eagerly imports SB3, Lightning, D4RL
and environment integrations that the Round-5 offline policy core does not
use.  The official module itself remains unchanged.

The MIT License (MIT)

Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class GPTConfig:
    """Base GPT config copied from the official implementation."""

    embd_pdrop = 0.0
    resid_pdrop = float(os.environ.get("DROPOUT", 0.15))
    attn_pdrop = float(os.environ.get("DROPOUT", 0.15))

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for key, value in kwargs.items():
            setattr(self, key, value)


class CausalSelfAttention(nn.Module):
    """Official multi-head causal attention with key-padding masking."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size + 1, config.block_size + 1)).view(
                1, 1, config.block_size + 1, config.block_size + 1
            ),
        )
        self.n_head = config.n_head

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_past=None,
    ):
        batch_size, sequence_length, channels = x.size()
        keys = self.key(x).view(
            batch_size,
            sequence_length,
            self.n_head,
            channels // self.n_head,
        ).transpose(1, 2)
        queries = self.query(x).view(
            batch_size,
            sequence_length,
            self.n_head,
            channels // self.n_head,
        ).transpose(1, 2)
        values = self.value(x).view(
            batch_size,
            sequence_length,
            self.n_head,
            channels // self.n_head,
        ).transpose(1, 2)

        attention = (queries @ keys.transpose(-2, -1)) * (
            1.0 / math.sqrt(keys.size(-1))
        )
        allowed = self.mask[:, :, :sequence_length, :sequence_length].bool()
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, sequence_length):
                raise ValueError(
                    "attention_mask must have shape "
                    f"{(batch_size, sequence_length)}, got "
                    f"{tuple(attention_mask.shape)}"
                )
            valid_tokens = attention_mask.to(device=x.device, dtype=torch.bool)
            allowed = allowed & valid_tokens[:, None, None, :]
            diagonal = torch.eye(
                sequence_length, device=x.device, dtype=torch.bool
            )[None, None]
            padded_queries = ~valid_tokens[:, None, :, None]
            allowed = allowed | (padded_queries & diagonal)
        attention = attention.masked_fill(~allowed, float("-inf"))
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_drop(attention)
        output = attention @ values
        output = output.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, channels
        )
        output = self.resid_drop(self.proj(output))
        if attention_mask is not None:
            output = output * valid_tokens.unsqueeze(-1).to(output.dtype)
        return output


class Block(nn.Module):
    """Official pre-LayerNorm attention/4d-FFN Transformer block."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln2(x))
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        return x


__all__ = ["GELU", "GPTConfig", "CausalSelfAttention", "Block"]

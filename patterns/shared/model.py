"""
Based on [Neel Nanda's implementation](https://colab.research.google.com/drive/1F6_1_cWXE5M7WocUcpQWp3v8z4b1jL20).
"""

from typing import Callable, List, Literal, Tuple, Union

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, tensor


class ExtModule(nn.Module):
    def __init__(
        self,
        init_scale: float = 1.0,
        init_mode: Literal["uniform", "normal"] = "uniform",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.init_scale = init_scale
        self.init_mode = init_mode

    def init_weights(self):
        for p in self.parameters():
            # if self.init_mode == "uniform":
            #     nn.init.kaiming_uniform_(p.data, a=0, mode='fan_in', nonlinearity='relu')
            # else:
            #     nn.init.kaiming_normal_(p.data, a=0, mode='fan_in', nonlinearity='relu')

            p.data *= self.init_scale


class Embed(nn.Module):
    def __init__(self, d_vocab: int, d_model: int):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_model))

    def forward(self, x):
        return torch.einsum("dbp -> bpd", self.W_E[:, x.long()])


class Unembed(nn.Module):
    def __init__(self, d_vocab: int, d_model: int):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_vocab))

    def forward(self, x):
        return x @ self.W_U


class PosEmbed(nn.Module):
    def __init__(self, max_ctx: int, d_model: int):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model) / np.sqrt(d_model))

    def forward(self, x):
        return x + self.W_pos[: x.shape[-2]]


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, epsilon: float = 1e-4):
        super().__init__()
        self.w_ln = nn.Parameter(torch.ones(d_model))
        self.b_ln = nn.Parameter(torch.zeros(d_model))
        self.epsilon = epsilon

    def forward(self, x):
        x = x - x.mean(axis=-1)[..., None]
        x = x / (x.std(axis=-1)[..., None] + self.epsilon)
        x = x * self.w_ln
        x = x + self.b_ln
        return x


class Attention(nn.Module):
    mask: Tensor

    def __init__(self, d_model, num_heads, d_head, num_ctx):
        super().__init__()
        self.W_K = nn.Parameter(
            torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model)
        )
        self.W_Q = nn.Parameter(
            torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model)
        )
        self.W_V = nn.Parameter(
            torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model)
        )
        self.W_O = nn.Parameter(
            torch.randn(d_model, d_head * num_heads) / np.sqrt(d_model)
        )
        self.register_buffer("mask", torch.tril(torch.ones((num_ctx, num_ctx))))
        self.d_head = d_head

    def forward(self, x):
        k = torch.einsum("ihd,bpd->biph", self.W_K, x)
        q = torch.einsum("ihd,bpd->biph", self.W_Q, x)
        v = torch.einsum("ihd,bpd->biph", self.W_V, x)
        attn_scores_pre = torch.einsum("biph,biqh->biqp", k, q)
        attn_scores_masked = torch.tril(attn_scores_pre) - 1e10 * (
            1 - self.mask[: x.shape[-2], : x.shape[-2]]
        )
        attn_matrix = F.softmax(attn_scores_masked / np.sqrt(self.d_head), dim=-1)
        z = torch.einsum("biph,biqp->biqh", v, attn_matrix)
        z_flat = einops.rearrange(z, "b i q h -> b q (i h)")
        out = torch.einsum("df,bqf->bqd", self.W_O, z_flat)
        return out


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, act_fn: Callable = F.relu):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model) / np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp) / np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.act_fn = act_fn
        # self.ln = LayerNorm(d_mlp)

    def forward(self, x):
        x = torch.einsum("md,bpd->bpm", self.W_in, x) + self.b_in
        x = self.act_fn(x)
        x = torch.einsum("dm,bpm->bpd", self.W_out, x) + self.b_out

        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_mlp: int,
        d_head: int,
        num_heads: int,
        num_ctx: int,
        act_fn: Callable = F.relu,
    ):
        super().__init__()
        # self.ln1 = LayerNorm(d_model)
        self.attn = Attention(d_model, num_heads, d_head, num_ctx)
        # self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_mlp, act_fn)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp((x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_vocab: int,
        d_model: int,
        d_mlp: int,
        d_head: int,
        num_heads: int,
        num_ctx: int,
        act_fn: Callable = F.relu,
        use_ln: bool = True,
    ):
        super().__init__()

        self.embed = Embed(d_vocab, d_model)
        self.pos_embed = PosEmbed(num_ctx, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, d_mlp, d_head, num_heads, num_ctx, act_fn)
                for i in range(num_layers)
            ]
        )
        # self.ln = LayerNorm(d_model)
        self.unembed = Unembed(d_vocab, d_model)
        # self.use_ln = use_ln

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)

        for block in self.blocks:
            x = block(x)
        # x = self.ln(x)
        x = self.unembed(x)
        return x

"""Nano-Sora: Minimal Diffusion Transformer for Video Generation with Flash Attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


class PatchEmbed3D(nn.Module):
    """
    Video to Spacetime Patch Embedding (Tubelets).
    Implementation of the "Sora" tokenizer: Conv3d with stride=kernel_size.
    """
    def __init__(self, patch_size: Tuple[int, int, int] = (2, 8, 8), in_chans: int = 1, embed_dim: int = 384):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class FlashAttention(nn.Module):
    """Memory-efficient attention using PyTorch 2.0's scaled_dot_product_attention.

    This uses Flash Attention under the hood, which is O(N) memory instead of O(N²).
    Critical for long sequences (2048+ tokens with fine patches).
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Use PyTorch 2.0's memory-efficient attention (Flash Attention)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block with AdaLN-Zero conditioning and Flash Attention.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Flash Attention (memory efficient)
        self.attn = FlashAttention(hidden_size, num_heads, dropout)

        # MLP
        mlp_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_size),
            nn.Dropout(dropout),
        )

        # AdaLN-Zero modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # Zero-Init
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Self-Attention with AdaLN
        x_norm = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out = self.attn(x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP with AdaLN
        x_norm = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.sinusoidal_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class NanoSora(nn.Module):
    """
    Nano-Sora: A minimal Diffusion Transformer for video generation with Flash Attention.

    Now uses memory-efficient attention, enabling training with fine patches (2048 tokens)
    on A100 GPUs without OOM errors.
    """
    def __init__(
        self,
        input_size: Tuple[int, int, int] = (16, 64, 64),
        patch_size: Tuple[int, int, int] = (2, 8, 8),
        in_chans: int = 1,
        hidden_size: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        frequency_embedding_size: int = 256,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads

        t, h, w = input_size
        pt, ph, pw = patch_size
        assert t % pt == 0 and h % ph == 0 and w % pw == 0

        self.patch_embed = PatchEmbed3D(patch_size, in_chans, hidden_size)
        self.num_patches = (t // pt) * (h // ph) * (w // pw)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.t_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

        patch_dim = in_chans * pt * ph * pw
        self.final_linear = nn.Linear(hidden_size, patch_dim)

        self._init_weights()
        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x) + self.pos_embed
        c = self.t_embedder(t)

        for block in self.blocks:
            x = block(x, c)

        shift, scale = self.final_adaLN(c).chunk(2, dim=1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        return self.final_linear(x)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

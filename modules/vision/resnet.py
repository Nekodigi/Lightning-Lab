import math
from functools import partial

import torch
import torch.nn.functional as F
from denoising_diffusion_pytorch.attend import Attend  # type: ignore
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

from modules.utils.utils import default, exists


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),  # type: ignore
    )


def Downsample(dim, dim_out=None, use_max_pool=True):
    return nn.Sequential(
        # Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        # max pool 2d
        (
            nn.MaxPool2d(2)
            if use_max_pool
            else Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2)
        ),
        nn.Conv2d(dim if use_max_pool else dim * 4, default(dim_out, dim), 1),  # type: ignore
    )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift  # type: ignore
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, cond_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(int(default(time_emb_dim, 0)) + int(default(cond_emb_dim, 0)), dim_out * 2),  # type: ignore
            )
            if exists(time_emb_dim) or exists(cond_emb_dim)
            else None
        )  # type: ignore

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, cond_emb=None):
        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(cond_emb)):
            cond_emb = tuple(
                filter(exists, (time_emb, cond_emb))
            )  # * might be confusing
            cond_emb = torch.cat(cond_emb, dim=-1)  # type: ignore
            cond_emb = self.mlp(cond_emb)  # type: ignore
            cond_emb = rearrange(cond_emb, "b c -> b c 1 1")
            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

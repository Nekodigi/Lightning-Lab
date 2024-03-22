from dataclasses import dataclass

from confs.CIFAR_10.conf import CIFAR10Config


@dataclass
class CNNCfg:
    dim: int
    dim_mults: list[int]
    fc_dims: list[int]
    full_attn: bool


@dataclass
class CIFAR10_01Config(CIFAR10Config):
    cnn: CNNCfg

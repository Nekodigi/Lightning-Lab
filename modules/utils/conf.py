import os
import subprocess
from dataclasses import dataclass
from typing import cast

from hydra import compose, initialize

from env import CONFS_PATH, REL_CONFS_PATH


@dataclass
class BaseInfo:
    type: str
    hash: str


class BaseTrainer:
    dataset: str
    lr: float
    val_every: int
    batch_size: int


@dataclass
class BaseConfig:
    info: BaseInfo
    trainer: BaseTrainer


def get_cfg(argv):
    _, proj, act, ver = argv
    with initialize(
        version_base=None, config_path=f"../../{REL_CONFS_PATH}/{proj}/{act}"
    ):
        return cast(BaseConfig, compose(config_name=ver))


def syncHash(msg="") -> str:
    result = subprocess.run(["syncHash", f"{msg}:wrench:"], executable="/usr/bin/zsh")
    # get last line of output
    print(result)
    return result.stdout.decode().split("\n")[-1]

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from sys import argv
from typing import cast

from hydra import compose, initialize
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from env import LOGS_PATH
from modules.utils.path import listdir_by_date


@dataclass
class BaseInfoCfg:
    type: str
    hash: str | None
    note: str | None


@dataclass
class BaseTrainerCfg:
    dataset: str
    lr: float
    epochs: int
    val_every: int
    batch_size: int
    resume_from: None | str
    dry: bool
    grad_acm: int
    swa: float | None
    weight_decay: float
    metrics1: str | None
    metrics2: str | None


@dataclass
class BaseConfig:
    info: BaseInfoCfg
    trainer: BaseTrainerCfg


@dataclass
class ActConfig:
    name: str


@dataclass
class RunnerConfig:
    pipe_output: bool
    delete_queue: bool


@dataclass
class GlobalConfig:
    info: BaseInfoCfg
    runner: RunnerConfig


@dataclass
class Tools:
    cfg: BaseConfig
    run: str | None
    logger: TensorBoardLogger
    ckpt: str | None


def get_cfg():
    _, proj, act, ver = argv[-4:]
    print(f"{proj}/{act}/{ver}")
    with initialize(version_base=None, config_path=f"{proj}/{act}"):
        cfg = cast(BaseConfig, compose(config_name=ver))
        if cfg.info.hash is None:
            cfg.info.hash = get_hash()
        return cfg


def get_latest_run() -> str | None:
    _, proj, act, ver = argv[-4:]
    path = f"{LOGS_PATH}/{proj}/{act}/{ver}"
    if os.path.exists(path):
        dirs = listdir_by_date(path)
        # dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
        return dirs[-1]  # type: ignore
    else:
        return None


def get_latest_ckpt(resume_from) -> str | None:
    _, proj, act, ver = argv[-4:]
    path = f"{proj}/{act}/{ver}"
    assert os.path.exists(f"{LOGS_PATH}/{path}"), f"{path} has not executed yet"
    if os.path.isdir(f"{LOGS_PATH}/{path}/{resume_from}/checkpoints"):
        latest = listdir_by_date(f"{LOGS_PATH}/{path}/{resume_from}/checkpoints")[-1]
        return f"{LOGS_PATH}/{path}/{resume_from}/checkpoints/{latest}"
    else:
        print(f"=={path} does not have checkpoints. Starting new training==")
        return None


def get_logger():
    _, proj, act, ver = argv[-4:]
    run = get_latest_run()
    return TensorBoardLogger(LOGS_PATH, f"{proj}/{act}/{ver}", run)


def get_act_cfg(proj: str, act: str):
    with initialize(version_base=None, config_path=f"{proj}/{act}"):
        cfg = cast(ActConfig, compose(config_name="env"))
        return cfg


def get_tools():
    cfg = get_cfg()
    run = get_latest_run()
    logger = get_logger()
    _, proj, act, ver = argv[-4:]
    Path(f"{LOGS_PATH}/{proj}/{act}/{ver}").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, f"{LOGS_PATH}/{proj}/{act}/{ver}/cfg.yaml")
    resume_from = (
        run if cfg.trainer.resume_from == "latest" else cfg.trainer.resume_from
    )
    ckpt = None if resume_from is None else get_latest_ckpt(resume_from)
    return Tools(cfg, run, logger, ckpt)


def get_hash(msg="") -> str:
    # run this command and get result as string  ["syncHash", f"{msg}:wrench:"]
    try:
        result: bytes = subprocess.check_output(
            ["git", "rev-parse", "--verify", "HEAD"],
        )
        # get last line of output
        return result.decode().strip()
    except subprocess.CalledProcessError:
        return "None"

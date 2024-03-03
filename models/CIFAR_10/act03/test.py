from typing import cast

from confs.CIFAR_10.act00.conf import CIFAR10_00Config
from confs.conf import get_tools
from models.CIFAR_10.datamodule import DataModule

tools = get_tools()
cfg, run, logger, ckpt = tools.cfg, tools.run, tools.logger, tools.ckpt  # type: ignore
cfg: CIFAR10_00Config = cast(CIFAR10_00Config, cfg)
print(cfg)

datamodule = DataModule(cfg.trainer)  # , use_embed=True
datamodule.setup()

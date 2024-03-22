from typing import cast

from confs.conf import get_tools

tools = get_tools()
cfg, run, logger, ckpt = tools.cfg, tools.run, tools.logger, tools.ckpt  # type: ignore
# cfg = cast(Sandbox_00Config, cfg)
print(cfg)
print(cfg.a, cfg.b)
print(cfg.a.c, cfg.a.format, cfg.d)

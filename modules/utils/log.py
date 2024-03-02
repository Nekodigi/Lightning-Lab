from lightning import LightningModule

from confs.conf import BaseConfig


def print_metrics(model: LightningModule, cfg: BaseConfig):
    metrics1, metrics2 = cfg.trainer.metrics1, cfg.trainer.metrics2
    if metrics1 is not None and metrics2 is not None:
        if hasattr(model, metrics1) and hasattr(model, metrics2):
            metrics1, metrics2 = getattr(model, metrics1), getattr(model, metrics2)
            print(f"{metrics1},{metrics2},SUCCESS")
        else:
            print(",,NO_OUTPUT")
    else:
        print(",,METRICS_NOT_DEFINED")

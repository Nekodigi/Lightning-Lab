from typing import cast

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import LearningRateMonitor, StochasticWeightAveraging
from torch import Tensor, nn
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy

from confs.CIFAR_10.act00.conf import CIFAR10_00Config
from confs.conf import get_tools
from models.CIFAR_10.datamodule import DataModule
from modules.utils.log import print_metrics

tools = get_tools()
cfg, run, logger, ckpt = tools.cfg, tools.run, tools.logger, tools.ckpt  # type: ignore
cfg: CIFAR10_00Config = cast(CIFAR10_00Config, cfg)
print(cfg)


class Classifier(L.LightningModule):
    def __init__(self, cfg: CIFAR10_00Config):
        super().__init__()
        self.save_hyperparameters()
        # predict from embed

        # more layer high spec model
        self.model = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 10),
        )
        self.cfg = cfg
        self.lr = cfg.trainer.lr

    def forward(self, x: Tensor) -> Tensor:
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def evaluate(self, batch, stage=None):
        x, y, emb = batch
        # acuraccy
        logits = self(emb)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, "multiclass", num_classes=10)

        if stage:
            self.log(f"{stage}_loss", loss)
            self.log(f"{stage}_acc", acc, prog_bar=True)
            if stage == "val" and hasattr(self, "train_loss"):
                self.log("hp_metric", loss)
                self.log("val_train_loss", self.train_loss / loss)
            # set to this object attribute

            setattr(self, f"{stage}_loss", loss)
            setattr(self, f"{stage}_acc", acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self.evaluate(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=self.cfg.trainer.weight_decay,  # 5e-4
        )


# init the autoencoder
model = Classifier(cfg)


datamodule = DataModule(cfg.trainer, use_embed=True)
print(datamodule.cfg.batch_size)


# model = Classifier(lr=0.05)

trainer = L.Trainer(
    max_epochs=cfg.trainer.epochs,
    accelerator="auto",
    # devices=1 if torch.cuda.is_available() else None,  # type: ignore
    callbacks=[
        LearningRateMonitor(logging_interval="step"),  # type: ignore
        StochasticWeightAveraging(swa_lrs=1e-2),
        # accumulator,
    ],
    accumulate_grad_batches=cfg.trainer.grad_acm,
    check_val_every_n_epoch=cfg.trainer.val_every,
    logger=logger,
    profiler="simple",
)


# cifar10_dm = DataModule()


trainer.fit(model, datamodule, ckpt_path=ckpt)  # type: ignore
trainer.test(model, datamodule, ckpt_path=ckpt)  # type: ignore


print_metrics(model, cfg)

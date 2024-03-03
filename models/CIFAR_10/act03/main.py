from typing import cast

import lightning as L
import timm
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import (LearningRateMonitor,
                                         StochasticWeightAveraging)
from torch import Tensor, nn
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy
from transformers import ViTForImageClassification

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
        self.cfg = cfg
        self.lr = cfg.trainer.lr
        self.model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-384"
        )

    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        # out = self.model(x)
        out = self.model(x, labels=labels)
        return F.log_softmax(out, dim=1)

    def evaluate(self, batch, stage=None):
        x, y = batch
        # acuraccy
        logits = self(x, y)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, "multiclass", num_classes=10)
        if stage:
            self.log(f"{stage}_loss", loss)
            self.log(f"{stage}_acc", acc, prog_bar=True)
            setattr(self, f"{stage}_loss", loss)
            setattr(self, f"{stage}_acc", acc)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x, labels=y)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

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


datamodule = DataModule(cfg.trainer, use_vit = True)  # , use_embed=True
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

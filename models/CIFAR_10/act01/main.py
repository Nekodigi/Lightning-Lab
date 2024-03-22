from functools import partial
from typing import cast

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import (
    GradientAccumulationScheduler,
    LearningRateMonitor,
    StochasticWeightAveraging,
)
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torchmetrics.functional import accuracy

from confs.CIFAR_10.act01.conf import CIFAR10_01Config
from confs.conf import get_tools
from models.CIFAR_10.datamodule import DataModule, ImbCfDataModule
from modules.utils.log import print_metrics
from modules.utils.utils import default
from modules.vision.resnet import Downsample, ResnetBlock
from modules.vision.vit import Attention, LinearAttention

tools = get_tools()
cfg, run, logger, ckpt = tools.cfg, tools.run, tools.logger, tools.ckpt  # type: ignore
cfg: CIFAR10_01Config = cast(CIFAR10_01Config, cfg)

IMAGE_SIZE = 32
FLASH_ATTN = False


INPUT_CHANNELS = 3
RESNET_BLOCK_GROUPS = 8
ATTN_HEADS = 4
ATTN_DIM_HEAD = 32


class Classifier(L.LightningModule):
    def __init__(self, cfg: CIFAR10_01Config):
        super().__init__()
        self.save_hyperparameters()
        dim = cfg.cnn.dim
        dim_mults = cfg.cnn.dim_mults
        fc_dims = cfg.cnn.fc_dims

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        fc_in_out = list(zip(fc_dims[:-1], fc_dims[1:]))
        latent_size = IMAGE_SIZE // (2 ** (len(dim_mults) - 1))

        block_klass = partial(ResnetBlock, groups=RESNET_BLOCK_GROUPS)

        FullAttention = partial(Attention, flash=FLASH_ATTN)
        attn_klass = FullAttention if cfg.cnn.full_attn else LinearAttention
        attn_klass = partial(attn_klass, dim_head=dim, heads=ATTN_HEADS)

        self.init_conv = nn.Conv2d(INPUT_CHANNELS, dim, 7, padding=3)  # type: ignore
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in),
                        block_klass(dim_in, dim_in),
                        attn_klass(dim_in),  # type: ignore
                        Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1),  # type: ignore
                    ]
                )
            )
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim)
        self.flatten = nn.Flatten()
        self.init_fc = nn.Linear(mid_dim * latent_size * latent_size, fc_dims[0])  # type: ignore
        self.act = nn.GELU()
        self.fcs = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(fc_in_out):
            self.fcs.append(nn.ModuleList([nn.Linear(dim_in, dim_out), nn.GELU()]))
        self.final_fc = nn.Linear(fc_dims[-1], 10)

        self.cfg = cfg
        self.lr = cfg.trainer.lr

    def forward(self, x: Tensor) -> Tensor:
        x = self.init_conv(x)
        for block1, block2, attn, down in self.downs:  # type: ignore
            x = block1(x)
            x = block2(x)
            # x = attn(x) + x
            x = down(x)
        x = self.mid_block1(x)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x)
        x = self.flatten(x)
        x = self.init_fc(x)
        x = self.act(x)
        for fc, act in self.fcs:  # type: ignore
            x = fc(x)
            x = act(x)
        x = self.final_fc(x)
        return F.log_softmax(x, dim=1)

    def evaluate(self, batch, stage=None):
        x, y = batch
        # acuraccy
        logits = self(x)
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
        # x, y = batch
        # logits = self(x)
        # loss = F.nll_loss(logits, y)
        # self.log("train_loss", loss)

        return self.evaluate(batch, "train")

    def add_histogram(self):
        for name, param in self.named_parameters():
            self.logger.experiment.add_histogram(name, param, self.current_epoch)  # type: ignore

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    # def on_validation_epoch_end(self):
    #     self.add_histogram()

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,  # type: ignore
            momentum=0.9,
            weight_decay=self.cfg.trainer.weight_decay,
        )
        scheduler_dict = {
            "scheduler": CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,  # type: ignore
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}


# init the autoencoder
model = Classifier(cfg)


datamodule = ImbCfDataModule(cfg.trainer)
print(datamodule.cfg.batch_size)


# model = Classifier(lr=0.05)
accumulator = GradientAccumulationScheduler(scheduling={0: 8, 4: 4, 8: 1})
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
    # gradient_clip_val=0.5,
    check_val_every_n_epoch=cfg.trainer.val_every,
    logger=logger,
)


# cifar10_dm = DataModule()


trainer.fit(model, datamodule, ckpt_path=ckpt)  # type: ignore
trainer.test(model, datamodule, ckpt_path=ckpt)  # type: ignore

if hasattr(model, "val_loss") and hasattr(model, "test_loss"):
    train_loss, val_loss = model.val_loss, model.test_loss
    print_metrics(train_loss, val_loss)
else:
    print(",,NO_OUTPUT")

from typing import cast

import lightning as L
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import Dataset, load_from_disk
from lightning.pytorch.callbacks import LearningRateMonitor, StochasticWeightAveraging
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import Tensor, nn
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy

from confs.CIFAR_10.act00.conf import CIFAR10_00Config
from confs.conf import get_tools
from env import DATASETS_PATH
from models.CIFAR_10.datamodule import DataModule
from modules.utils.log import print_metrics

tools = get_tools()
cfg, run, logger, ckpt = tools.cfg, tools.run, tools.logger, tools.ckpt  # type: ignore
cfg: CIFAR10_00Config = cast(CIFAR10_00Config, cfg)
print(cfg)

embed_type = "vit"

datamodule = DataModule(cfg.trainer, use_embed=True, embed_type=embed_type)
print(datamodule.cfg.batch_size)

# Load the dataset
ref_dataset = cast(
    Dataset, load_from_disk(f"{DATASETS_PATH}/cifar10ib/vit_embed")["train"]
).with_format("torch")
ref_dataset.add_faiss_index("emb768")


class Classifier(L.LightningModule):
    def __init__(self, cfg: CIFAR10_00Config):
        super().__init__()
        self.save_hyperparameters()
        # predict from embed

        if embed_type == "vit":
            init_dim = 768
        elif embed_type == "clip":
            init_dim = 512
        elif embed_type == "api":
            init_dim = 512
        # more layer high spec model
        self.model = nn.Sequential(
            nn.Linear(init_dim, 10),
            # nn.GELU(),
            # nn.Linear(256, 128),
            # nn.GELU(),
            # nn.Linear(128, 64),
            # nn.GELU(),
            # nn.Linear(64, 10),
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
        scores, retrieved = datamodule.rag_dataset.get_nearest_examples_batch(
            "emb768", emb.cpu().numpy(), 1
        )
        # print(retrieved[:4])
        preds = torch.cat([torch.tensor(data["label"]) for data in retrieved])  # type: ignore
        # print(preds[:4], y[:4], scores[:4])
        # preds = torch.ones_like(y)
        acc = accuracy(preds.cpu(), y.cpu(), "multiclass", num_classes=10)

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
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.cfg.trainer.lr,  # type: ignore
            momentum=0.9,
            weight_decay=self.cfg.trainer.weight_decay,
        )
        scheduler_dict = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.trainer.epochs,  # type: ignore
        )
        return [optimizer], [scheduler_dict]


# init the autoencoder
model = Classifier(cfg)


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
    # profiler="simple",
)


# cifar10_dm = DataModule()


trainer.fit(model, datamodule, ckpt_path=ckpt)  # type: ignore
trainer.test(model, datamodule, ckpt_path=ckpt)  # type: ignore
# datamodule.setup()
# model = Classifier.load_from_checkpoint(ckpt)  # type: ignore


def class_wise_acc(model, loader, device):
    class_acc_list, y_preds, true_label = [], [], []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for img, labels, inputs in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            y_preds.extend(preds.cpu().numpy())
            true_label.extend(labels.cpu().numpy())
            # print(preds[:4], labels[:4])
        cf = confusion_matrix(true_label, y_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = np.divide(
            cls_hit, cls_cnt, out=np.zeros_like(cls_hit), where=cls_cnt != 0
        )
        # cls_acc = cls_hit / cls_cnt
        class_acc_list.append(cls_acc)
    model.train()
    return (
        class_acc_list[0],
        y_preds,
        true_label,
        np.round(confusion_matrix(true_label, y_preds), 2),
    )


acc, pred, label, cm = class_wise_acc(model, datamodule.test_dataloader(), "cuda:0")
# visualize cm
plt.figure(figsize=(10, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=range(10),  # type: ignore
    yticklabels=range(10),  # type: ignore
)
plt.xlabel("Predictions")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
# save as img
plt.savefig("ConfusionMatrix.png")
plt.show()

print_metrics(model, cfg)

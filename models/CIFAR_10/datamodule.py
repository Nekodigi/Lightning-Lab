from typing import cast

import lightning as L
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import random_split
from torchvision import transforms as T

from confs.conf import BaseTrainerCfg
from modules.datasets.dataloader import make_dataloader


class MyDataset(TorchDataset):
    def __init__(self, dataset: Dataset, test=False):
        self.dataset = dataset
        if test:
            self.trans = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        else:
            self.trans = T.Compose(
                [
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(10),
                    T.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    T.RandomResizedCrop(32, scale=(0.8, 1.0)),
                    T.RandomGrayscale(p=0.1),
                    T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            # random affine and erasing
            # self.trans = T.Compose(
            #     [
            #         T.RandomHorizontalFlip(),
            #         T.RandomRotation(10),
            #         T.ColorJitter(
            #             brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            #         ),
            #         T.RandomResizedCrop(32, scale=(0.8, 1.0)),
            #         T.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
            #         T.ToTensor(),
            #         T.RandomErasing(p=0.5, scale=(0.02, 0.2)),
            #         T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            #     ]
            # )

    def __len__(self):
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx):
        item = self.dataset[idx]
        img = self.trans(item["img"])
        label = item["label"]
        return img, label


# download cifar10 from huggingface
class DataModule(L.LightningDataModule):
    def __init__(self, cfg: BaseTrainerCfg):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self):
        self.dataset = cast(DatasetDict, load_dataset("cifar10"))

    def setup(self, stage=None):
        # self.train, self.val = random_split(
        #     MyDataset(self.dataset["train"]), [45000, 5000]
        # )
        self.train = MyDataset(self.dataset["train"])
        self.val = MyDataset(self.dataset["test"], test=True)
        self.test = MyDataset(self.dataset["test"], test=True)

    def train_dataloader(self):
        return make_dataloader(self.train, self.cfg)

    def val_dataloader(self):
        return make_dataloader(self.val, self.cfg, shuffle=False)

    def test_dataloader(self):
        return make_dataloader(self.test, self.cfg, shuffle=False)

import os
from typing import cast

import lightning as L
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Subset, random_split
from torchvision import transforms as T

from confs.conf import BaseTrainerCfg
from env import DATASETS_PATH
from modules.datasets.dataloader import make_dataloader


class MyDataset(TorchDataset):
    def __init__(
        self,
        dataset: Dataset,
        test=False,
        use_embed=False,
        use_vit=True,
        embed_type="clip",
    ):
        self.dataset = dataset
        self.use_vit = use_vit
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
        self.use_embed = use_embed
        self.embed_type = embed_type

    def __len__(self):
        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.use_vit:
            img = item["pixel_values"]
        else:
            img = self.trans(item["img"])
        label = item["label"]
        if self.use_embed:
            if self.embed_type == "vit":
                return img, label, torch.tensor(item["emb768"])
            elif self.embed_type == "clip":
                return img, label, torch.tensor(item["embed"])
            else:
                assert False, f"embed_type {self.embed_type} not implemented"
        else:
            return img, label


# download cifar10 from huggingface
class DataModule(L.LightningDataModule):
    def __init__(
        self,
        cfg: BaseTrainerCfg,
        use_embed=False,
        use_vit=False,
        embed_type="clip",
        strict_test=False,
    ):
        super().__init__()
        self.cfg = cfg
        self.use_embed = use_embed
        self.use_vit = use_vit
        self.embed_type = embed_type

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # self.train, self.val = random_split(
        #     MyDataset(self.dataset["train"]), [45000, 5000]
        # )
        if self.use_embed:
            if self.embed_type == "clip":
                embed_name = "embed"
                embed_key = "embed"
            elif self.embed_type == "vit":
                embed_name = "vit_embed"
                embed_key = "emb768"
            self.dataset = cast(
                DatasetDict, load_from_disk(f"{DATASETS_PATH}/cifar10ib/{embed_name}")
            )
        else:
            # self.dataset = cast(DatasetDict, load_dataset("cifar10ib"))
            self.dataset = cast(
                DatasetDict, load_from_disk(f"{DATASETS_PATH}/cifar10ib/base")
            )
        self.train_dataset = self.dataset["train"]
        self.rag_dataset = self.dataset["fsl5"]
        self.rag_dataset.add_faiss_index(embed_key)
        print("auth_test")
        print(self.dataset.keys())
        self.test_dataset = self.dataset["test"]

        if self.use_vit:
            self.train_dataset = self.train_dataset.map(
                image_processor,
                batched=True,
                num_proc=os.cpu_count(),
                batch_size=128,
            )
            self.test_dataset = self.test_dataset.map(
                image_processor,
                batched=True,
                num_proc=os.cpu_count(),
                batch_size=128,
            )
        print(self.train_dataset)
        self.train = MyDataset(
            self.train_dataset,
            use_embed=self.use_embed,
            use_vit=self.use_vit,
            embed_type=self.embed_type,
        )
        self.val = MyDataset(
            self.test_dataset,
            test=True,
            use_embed=self.use_embed,
            use_vit=self.use_vit,
            embed_type=self.embed_type,
        )
        self.test = MyDataset(
            self.test_dataset,
            test=True,
            use_embed=self.use_embed,
            use_vit=self.use_vit,
            embed_type=self.embed_type,
        )

    def train_dataloader(self):
        return make_dataloader(self.train, self.cfg)

    def val_dataloader(self):
        return make_dataloader(self.val, self.cfg, shuffle=False)

    def test_dataloader(self):
        return make_dataloader(self.test, self.cfg, shuffle=False)


import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(
        self,
        root,
        imb_type="exp",
        imb_factor=0.01,
        rand_number=0,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        fsl=-1,
    ):
        super(IMBALANCECIFAR10, self).__init__(
            root, train, transform, target_transform, download
        )
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor, fsl)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, fsl):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if fsl > 0:
            for cls_idx in range(cls_num):
                img_num_per_cls.append(fsl)
            return img_num_per_cls
        if imb_type == "exp":
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == "step":
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)

        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            # print(selec_idx)
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend(
                [
                    the_class,
                ]
                * the_img_num
            )
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }
    cls_num = 100


class ImbCfDataModule(L.LightningDataModule):
    def __init__(self, cfg: BaseTrainerCfg, use_embed=False, use_vit=False):
        super().__init__()
        self.cfg = cfg
        self.use_embed = use_embed
        self.use_vit = use_vit

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_val = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        train_dataset = IMBALANCECIFAR10(
            root="./data",
            imb_type="exp",
            imb_factor=0.1,
            rand_number=0,
            train=True,
            download=True,
            transform=transform_train,
        )
        train_indices, val_indices = train_test_split(
            list(range(len(train_dataset.targets))),
            test_size=0.3,
            stratify=train_dataset.targets,
        )
        self.val_dataset = Subset(train_dataset, val_indices)
        self.train_dataset = Subset(train_dataset, train_indices)
        self.test_dataset = IMBALANCECIFAR10(
            root="./data",
            imb_type="exp",
            imb_factor=0.1,
            rand_number=0,
            train=False,
            download=True,
            transform=transform_val,
        )

    def train_dataloader(self):
        return make_dataloader(self.train_dataset, self.cfg)

    def val_dataloader(self):
        return make_dataloader(self.val_dataset, self.cfg, shuffle=False)

    def test_dataloader(self):
        return make_dataloader(self.test_dataset, self.cfg, shuffle=False)


from pathlib import Path

from PIL.PngImagePlugin import PngImageFile
from transformers import AutoImageProcessor


def image_processor(
    batch: dict,
    # *,
    # model_name: str,
    # image_key: str,
    # cache_dir: str | Path,
):

    image_processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-384"
    )  # model_name, cache_dir=cache_dir
    images = (
        batch if isinstance(batch, PngImageFile) else batch["img"]
    )  # allow for inference input as raw text
    return image_processor(images, return_tensors="pt")

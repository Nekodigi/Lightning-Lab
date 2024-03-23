from functools import partial
from typing import Optional

import torch
import torchvision.transforms as transforms
import vertexai
from datasets import Dataset, DatasetDict
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor, CLIPModel
from vertexai.vision_models import MultiModalEmbeddingModel, MultiModalEmbeddingResponse

from env import DATASETS_PATH
from models.CIFAR_10.datamodule import IMBALANCECIFAR10

if __name__ == "__main__":
    train_dataset = IMBALANCECIFAR10(
        root="./data",
        imb_type="exp",
        imb_factor=0.1,
        rand_number=0,
        train=True,
        download=True,
    )
    test_dataset = IMBALANCECIFAR10(
        root="./data",
        imb_type="exp",
        imb_factor=0.1,
        rand_number=0,
        train=False,
        download=True,
    )
    auth_test_dataset = IMBALANCECIFAR10(
        "./data",
        imb_type="exp",
        imb_factor=0.95,
        download=True,
        train=False,
        rand_number=0,
    )
    train_images = [Image.fromarray(img) for img in train_dataset.data]
    test_images = [Image.fromarray(img) for img in test_dataset.data]
    auth_test_images = [Image.fromarray(img) for img in auth_test_dataset.data]
    train_dataset = Dataset.from_dict(
        {
            "img": train_images,
            "label": train_dataset.targets,
        }
    )
    test_dataset = Dataset.from_dict(
        {
            "img": test_images,
            "label": test_dataset.targets,
        }
    )
    auth_test_dataset = Dataset.from_dict(
        {
            "img": auth_test_images,
            "label": auth_test_dataset.targets,
        }
    )
    dataset_dict: DatasetDict = DatasetDict(
        {"train": train_dataset, "test": test_dataset, "auth_test": auth_test_dataset}
    )

    print("DATASET LOADED")
    dataset_dict.save_to_disk(f"{DATASETS_PATH}/cifar10ib/base")

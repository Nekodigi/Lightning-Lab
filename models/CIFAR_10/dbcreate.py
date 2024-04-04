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
    fsl5_dataset = IMBALANCECIFAR10(
        root="./data", rand_number=0, train=True, download=True, fsl=5
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
    datasets = [train_dataset, fsl5_dataset, test_dataset, auth_test_dataset]
    new_datasets = []
    for dataset in datasets:
        images = [Image.fromarray(img) for img in dataset.data]
        dataset = Dataset.from_dict(
            {
                "img": images,
                "label": dataset.targets,
            }
        )
        new_datasets.append(dataset)
    dataset_dict: DatasetDict = DatasetDict(
        {
            "train": new_datasets[0],
            "fsl5": new_datasets[1],
            "test": new_datasets[2],
            "auth_test": new_datasets[3],
        }
    )

    print("DATASET LOADED")
    dataset_dict.save_to_disk(f"{DATASETS_PATH}/cifar10ib/base")

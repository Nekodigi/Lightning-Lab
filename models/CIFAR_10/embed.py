import os
from functools import partial

import torch
from datasets import DatasetDict, load_dataset  # , load_from_disk

# from diffusers import AutoencoderKL
from rich.progress import Progress
from torchvision import transforms as T  # type: ignore
from transformers import AutoProcessor, CLIPModel

from env import DATASETS_PATH


def clip_embed(batch, rank, model, processor):
    device = f"cuda:{(rank or 0)}"
    model.to(device)

    def embed(x):
        inputs = processor(images=x, return_tensors="pt", do_rescale=False).to(device)
        emb = model.get_image_features(**inputs)[0]
        return emb

    batch["clip_embed"] = [embed(x) for x in batch["img"]]
    return batch


DATASET_NAME = "cifar10"


def create_embed():
    with Progress() as prog:
        if not os.path.exists(f"{DATASETS_PATH}/{DATASET_NAME}/embed"):
            print("!!Embedding has not created!!")
            task = prog.add_task("Applying Embedding", total=10)
            datasetDict: DatasetDict = load_dataset(DATASET_NAME)  # type: ignore  # , ignore_verifications=True, verification_mode="no_checks"
            assert isinstance(
                datasetDict, DatasetDict
            ), "Expected to be DatasetDict"  # * Possibly accept dataset as well
            prog.update(task, advance=2)  # 10% Dataset Loaded
            print(datasetDict)

            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            prog.update(task, advance=1)  # 30% Model Loaded

            gpu_comp = partial(clip_embed, model=model, processor=processor)
            newDatasetDict = DatasetDict()
            for key, dataset in datasetDict.items():
                newDatasetDict[key] = dataset.map(
                    gpu_comp,
                    batched=True,
                    batch_size=64,
                    with_rank=True,
                    num_proc=torch.cuda.device_count(),
                )
            prog.update(task, advance=6)  # 80% Embedding Completed
            newDatasetDict.save_to_disk(f"{DATASETS_PATH}/{DATASET_NAME}/embed")
            prog.update(task, advance=1)  # 90% Dataset Saved


if __name__ == "__main__":
    create_embed()

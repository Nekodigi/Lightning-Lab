import os
import tempfile
from functools import partial

import vertexai
from datasets import DatasetDict, load_from_disk
from multiprocess import set_start_method

# from diffusers import AutoencoderKL
from rich.progress import Progress
from torchvision import transforms as T  # type: ignore
from transformers import AutoProcessor, CLIPModel, ViTFeatureExtractor, ViTModel
from vertexai.vision_models import Image as VertexImage
from vertexai.vision_models import MultiModalEmbeddingModel, MultiModalEmbeddingResponse

from env import DATASETS_PATH


def clip_embed(batch, rank, model, processor):
    device = f"cuda:{(rank or 0)}"
    model.to(device)

    def c_embed(x):
        inputs = processor(images=x, return_tensors="pt", do_rescale=False).to(device)
        emb = model.get_image_features(**inputs)[0]
        return emb

    batch["clip_embed"] = [c_embed(x) for x in batch["img"]]
    return batch


def api_embed(batch, rank, model):

    def a_embed(x):
        temp_dir = tempfile.mkdtemp()
        temp_image_path = os.path.join(temp_dir, f"{rank}.jpg")
        x.save(temp_image_path)
        image = VertexImage.load_from_file(temp_image_path)
        emb = model.get_embeddings(
            image=image,
            dimension=512,
        )
        assert emb.image_embedding is not None, "Image embedding is None"
        return emb.image_embedding
        # return [0.0] * 512

    batch["emb512"] = [a_embed(x) for x in batch["img"]]
    return batch


def vit_embed(batch, rank, model, processor):
    device = f"cuda:{(rank or 0)}"
    model.to(device)

    def v_embed(x):
        inputs = processor(images=x, return_tensors="pt").to(device)
        emb = model(**inputs).last_hidden_state[:, 0][0]
        return emb

    batch["emb768"] = [v_embed(x) for x in batch["img"]]
    return batch


DATASET_NAME = "cifar10ib"
OPERATION = "vit_embed"


def create_embed():
    with Progress() as prog:
        # if not os.path.exists(f"{DATASETS_PATH}/{DATASET_NAME}/{OPERATION}"):
        print("!!Embedding has not been created!!")
        task = prog.add_task("Applying Embedding", total=10)
        # TODO LOAD FROM LOCAL AND USE!
        datasetDict: DatasetDict = load_from_disk(f"{DATASETS_PATH}/{DATASET_NAME}/base")  # type: ignore  # , ignore_verifications=True, verification_mode="no_checks"
        assert isinstance(
            datasetDict, DatasetDict
        ), "Expected to be DatasetDict"  # * Possibly accept dataset as well
        prog.update(task, advance=2)  # 10% Dataset Loaded
        print(datasetDict)

        if OPERATION == "api_embed":
            vertexai.init(project="sandbox-35d1d", location="us-central1")
            model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
            gpu_comp = partial(api_embed, model=model)
        elif OPERATION == "vit_embed":
            processor = ViTFeatureExtractor.from_pretrained(
                "google/vit-base-patch16-224-in21k"
            )
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            gpu_comp = partial(vit_embed, model=model, processor=processor)
        else:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            gpu_comp = partial(clip_embed, model=model, processor=processor)
        prog.update(task, advance=1)  # 30% Model Loaded

        newDatasetDict = DatasetDict()
        for key, dataset in datasetDict.items():
            newDatasetDict[key] = dataset.map(
                gpu_comp,
                batched=True,
                batch_size=64,
                with_rank=True,
                num_proc=1,  # torch.cuda.device_count()
            )
        prog.update(task, advance=6)  # 80% Embedding Completed
        newDatasetDict.save_to_disk(f"{DATASETS_PATH}/{DATASET_NAME}/{OPERATION}")
        prog.update(task, advance=1)  # 90% Dataset Saved
        # else:
        #     print("Embedding has been created")


if __name__ == "__main__":
    set_start_method("spawn")
    create_embed()

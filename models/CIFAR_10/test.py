import os
import tempfile
from typing import Optional

import vertexai
from PIL import Image as PILImage
from vertexai.vision_models import (
    Image,
    MultiModalEmbeddingModel,
    MultiModalEmbeddingResponse,
)


def get_image_embeddings(
    project_id: str,
    location: str,
    image_path: str,
    contextual_text: Optional[str] = None,
) -> MultiModalEmbeddingResponse:
    """Example of how to generate multimodal embeddings from image and text.

    Args:
        project_id: Google Cloud Project ID, used to initialize vertexai
        location: Google Cloud Region, used to initialize vertexai
        image_path: Path to image (local or Google Cloud Storage) to generate embeddings for.
        contextual_text: Text to generate embeddings for.
    """

    vertexai.init(project=project_id, location=location)

    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
    print("Loaded")

    # 1.1s / img
    images = ["flower.jpg", "flower2.jpg"]  #
    for image_path in images:
        print("TMP")
        pil_image = PILImage.open(image_path)
        temp_dir = tempfile.mkdtemp()
        temp_image_path = os.path.join(temp_dir, "temp_image.jpg")
        pil_image.save(temp_image_path)
        image = Image.load_from_file(temp_image_path)
        print("TMP")

        embeddings = model.get_embeddings(
            image=image,
            dimension=512,
        )
        assert embeddings.image_embedding is not None, "Image embedding is None"
        print(len(embeddings.image_embedding))
        # gcloud auth application-default login
        # gcloud auth application-default set-quota-project sandbox-35d1d


if __name__ == "__main__":
    get_image_embeddings(
        project_id="sandbox-35d1d",
        location="us-central1",
        image_path="flower.jpg",
    )

import gc
import logging

import torch
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, config: dict[str, str]) -> None:
        self.stopped: bool = True
        try:
            self.model_name: str = config["model"]
            self.batch_size: int = int(config["batch_size"])
            normalize_value = config["normalize"].lower()
            if normalize_value not in {"true", "false"}:
                raise ValueError("normalize must be either 'true' or 'false'.")
            self.normalize = normalize_value == "true"
            self.normalize: bool = bool(config["normalize"])
        except Exception as e:
            logger.error(f"Got Exception {e} reading configuration.")
            raise

        try:
            self.model: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                encode_kwargs={
                    "batch_size": self.batch_size,
                    "normalize_embeddings": self.normalize,
                },
            )
        except Exception as e:
            logger.error(f"Got Exception {e} loading model {self.model_name}")
            raise
        try:
            self.embedding_dim: int | None = (
                self.model._client.get_sentence_embedding_dimension()
            )
            if self.embedding_dim is None:
                raise TypeError("Embedding Dimension is None")
            if self.embedding_dim <= 0:
                raise ValueError(f"Invalid Embedding dimensions {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Got Exception {e} getting embedding dimensions")
            raise

        self.stopped = False

    def create_embedding(self, inputs) -> list[list[float]]:
        if self.stopped:
            raise RuntimeError("Embedder is stopped.")
        try:
            embeddings = self.model.embed_documents(texts=inputs)
        except Exception as e:
            logger.error(f"Got Exception {e} creating embedding.")
            raise
        return embeddings

    def cleanup(self) -> None:
        self.stopped = True
        if hasattr(self, "model") and self.model is not None:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

import gc
import logging

import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, config: dict[str, str]) -> None:
        self.stopped: bool = True
        try:
            self.model_name = config["model"]
            self.batch_size = int(config["batch_size"])
        except Exception as e:
            logger.error(f"Got Exception {e} reading configuration.")
            raise
        try:
            self.model = CrossEncoder(model_name_or_path=self.model_name)
        except Exception as e:
            logger.error(f"Got Exception {e} setting up retriever")
            raise
        self.stopped = False

    def rerank_documents(
        self, inputs: list[tuple[str, Document]], outcount: int
    ) -> list[Document]:
        if self.stopped:
            raise RuntimeError("Reranker is stopped.")
        if inputs == []:
            return []
        try:
            texts: list[str]
            docs: list[Document]
            texts, docs = map(list, zip(*inputs))
            doc_page = [doc.page_content for doc in docs]
            textpairs: list[tuple[str, str]] = list(zip(texts, doc_page))
            scores: torch.Tensor = self.model.predict(
                inputs=textpairs, batch_size=self.batch_size
            )
            # Some models e.g bert-multilingual-passage-reranking-msmarco
            # gives two score not_relevant and relevant as compared to the query.
            if len(scores.shape) > 1:  # we are going to get the relevant scores
                scores = scores[:, 1]
            scored_docs = list(zip(docs, scores))
            sorted_docs = sorted(scored_docs, key=lambda x: x[1].item(), reverse=True)
            reranked_docs = [doc for doc, _ in sorted_docs[:outcount]]
        except Exception as e:
            logger.error(f"Got Exception {e} invoking reranker")
            raise
        return reranked_docs

    def cleanup(self) -> None:
        self.stopped = True
        if hasattr(self, "model") and self.model is not None:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

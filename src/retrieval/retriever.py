import logging

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Retriever:
    def __init__(
        self, config: dict[str, str], retrive_chunk_count: int, vecstore: VectorStore
    ) -> None:
        self.stopped: bool = True
        try:
            self.chunk_count = retrive_chunk_count
            self.vdbname = config["vdbname"]
            if self.vdbname not in ["qdrant", "chroma", "pgvector"]:
                raise ValueError("Vector Database must be qdrant, chroma, or pgvector.")
            self.search_type = config["search_type"]
        except Exception as e:
            logger.error(f"Got Exception {e} reading configuration.")
            raise
        try:
            self.vecstore = vecstore.as_retriever(
                search_type=self.search_type, search_kwargs={"k": self.chunk_count}
            )
        except Exception as e:
            logger.error(f"Got Exception {e} setting up retriever")
            raise
        self.stopped = False

    def retrive_documents(self, query: str) -> list[Document]:
        if self.stopped:
            raise RuntimeError("Retriever is stopped.")
        try:
            retrieved_docs = self.vecstore.invoke(input=query)
        except Exception as e:
            logger.error(f"Got Exception {e} invoking retriever")
            raise
        return retrieved_docs

    def cleanup(self) -> None:
        self.stopped = True

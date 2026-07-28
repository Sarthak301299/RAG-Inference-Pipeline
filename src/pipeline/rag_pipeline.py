import asyncio
import logging
from uuid import uuid4

import yaml
from langchain_core.documents import Document

from src.generation.vllm import VLLMGenerator
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import Indexer
from src.ingestion.loader import Loader
from src.pipeline.schemas import RAGInputPromptFormat
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RAGPipeLine:
    def __init__(self) -> None:
        self.stopped: bool = True
        try:
            with open("config/config.yml", "r") as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Got Exception {e} parsing configuration file")
            raise
        try:
            self.loader = Loader(self.config["ingestion"]["loading"])
            self.embedder = Embedder(self.config["ingestion"]["embedding"])
            if self.embedder.embedding_dim is None:
                raise TypeError("Invalid Embedding embedding dimensions")
            self.indexer = Indexer(
                self.config["vectorDB"],
                self.embedder.embedding_dim,
                self.embedder.model,
            )
            self.chunker = Chunker(self.config["ingestion"]["chunking"])
        except Exception as e:
            logger.error(f"Got Exception {e} initializing ingestion objects")
            raise
        try:
            self.final_chunk_count = int(self.config["vectorDB"]["final_chunk_count"])
            self.retrieve_chunk_count = int(
                self.config["vectorDB"]["final_chunk_count"]
            ) * int(self.config["vectorDB"]["retrival_to_rerank_ratio"])
            self.retriver = Retriever(
                self.config["vectorDB"],
                self.retrieve_chunk_count,
                self.indexer.database,
            )
            self.reranker = Reranker(self.config["retrieval"]["reranker"])
        except Exception as e:
            logger.error(f"Got Exception {e} initializing retrieval objects")
            raise
        try:
            self.generator = VLLMGenerator(self.config["generation"])
        except Exception as e:
            logger.error(f"Got Exception {e} initializing generation objects")
            raise

        self.stopped = False

    def ingest_data(self) -> None:
        if self.stopped:
            raise RuntimeError("RAG Pipeline is stopped.")
        logger.info("Starting Context Ingestion.")
        try:
            grouped_docs: dict[str, list[Document]] = self.loader.parse_docs()
            chunked_docs: list[Document] = []
            for key, value in grouped_docs.items():
                chunks = self.chunker.generate_chunks(value, key)
                chunked_docs.extend(chunks)
            texts = [chunked_doc.page_content for chunked_doc in chunked_docs]
            metadatas = [chunked_doc.metadata for chunked_doc in chunked_docs]
            embeddings = self.embedder.create_embedding(texts)
            self.indexer.index_into_vectorDB(list(zip(texts, embeddings)), metadatas)
        except Exception as e:
            logger.error(f"Got Exception {e} while ingesting context.")
            raise
        logger.info("Context has been ingested.")

    def build_prompts(
        self, query_doc_pairs: list[tuple[str, list[Document]]]
    ) -> list[str]:
        prompts: list[str] = []
        for query, chunks in query_doc_pairs:
            raginput = RAGInputPromptFormat(user_query=query, chunks=chunks)
            prompt = raginput.get_formatted_prompt()
            prompts.append(prompt)
        return prompts

    async def generate_contextualized_output(self, queries: list[str]) -> list[str]:
        if self.stopped:
            raise RuntimeError("RAG Pipeline is stopped.")
        try:
            query_retrived_doc_pairs: list[list[tuple[str, Document]]] = [
                [(query, doc) for doc in self.retriver.retrive_documents(query)]
                for query in queries
            ]
            reranked_docs: list[list[Document]] = [
                self.reranker.rerank_documents(qrdpair, self.final_chunk_count)
                for qrdpair in query_retrived_doc_pairs
            ]
        except Exception as e:
            logger.error(f"Got Exception {e} while retrieving from VectorDB.")
            raise
        try:
            prompts: list[str] = self.build_prompts(list(zip(queries, reranked_docs)))
            tasks = [
                self.generator.generate(prompt, str(uuid4())) for prompt in prompts
            ]
            responses = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Got Exception {e} while generating outputs.")
            raise
        return responses

    def cleanup(self) -> None:
        self.stopped = True

import logging

import yaml
from langchain_core.documents import Document

from src.agent.agent import Agent, AgentRunResult, MaxIterationsExceeded
from src.agent.tools import CalculatorTool, FileLookupTool, RetrievalTool, Tool
from src.generation.vllm import VLLMGenerator
from src.ingestion.chunker import Chunker
from src.ingestion.embedder import Embedder
from src.ingestion.indexer import Indexer
from src.ingestion.loader import Loader
from src.pipeline.schemas import make_agent_step_schema
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AgentPipeline:
    def __init__(self) -> None:
        self.stopped = True
        self.context_ingested = False
        try:
            with open("config/config.yml", "r") as file:
                self.config = yaml.safe_load(file)
            self.max_iterations = int(self.config["agent"]["max_iterations"])
            self.max_thought_len = int(self.config["agent"]["max_thought_len"])
            self.max_action_input_len = int(
                self.config["agent"]["max_action_input_len"]
            )
        except Exception as e:
            logger.error(f"Got Exception {e} parsing configuration file")
            raise
        logger.info("Initializing ingestion modules.")
        try:
            self.loader = Loader(
                self.config["ingestion"]["loading"], save_documents=True
            )
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
        logger.info("Initializing Retrival Module.")
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
        logger.info("Initializing Agentic Tools.")
        try:
            tools: list[Tool] = [
                CalculatorTool(),
                RetrievalTool(self.retriver, self.reranker, self.final_chunk_count),
                FileLookupTool(documents_by_source=self.loader.documents_by_source),
            ]
        except Exception as e:
            logger.error(f"Got Exception {e} initializing agentic tools")
            raise
        logger.info("Initializing Generator Module.")
        try:
            self.generator = VLLMGenerator(
                self.config["generation"],
                make_agent_step_schema(
                    tool_names=[tool.name for tool in tools],
                    max_thought_len=self.max_thought_len,
                    max_action_input_len=self.max_action_input_len,
                ),
            )
        except Exception as e:
            logger.error(f"Got Exception {e} initializing generation objects")
            raise
        self.stopped = False
        logger.info("Initializing Agent.")
        try:
            self.agent = Agent(
                tools=tools,
                step_generator=self.generator,
                max_iterations=self.max_iterations,
            )
        except Exception as e:
            logger.error(f"Got Exception {e} initializing agent")
            raise
        self.stopped = False
        logger.info("Agent Pipeline Initialized.")

    async def run(self, query: str) -> AgentRunResult:
        try:
            return await self.agent.run(query)
        except MaxIterationsExceeded as e:
            logger.warning(f"Agent hit max iterations for query: {query!r}: {e}")
            return {
                "answer": "I was unable to find a satisfactory answer within the allowed number of iterations.",
                "scratchpad": e.scratchpad,
                "iterations_used": len(e.scratchpad),
            }
        except Exception as e:
            logger.error(f"Got Exception {e} while running the agent loop.")
            raise

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
        self.context_ingested = True
        logger.info("Context has been ingested.")

    def shutdown(self):
        self.stopped = True
        try:
            self.loader.cleanup()
            self.chunker.cleanup()
            self.embedder.cleanup()
            self.indexer.cleanup()
            self.retriver.cleanup()
            self.reranker.cleanup()
            self.generator.cleanup()
            self.agent.cleanup()
        except Exception as e:
            logger.error(f"Got Exception {e} while cleaning up Agentic modules")
            raise

    def is_stopped(self) -> bool:
        output: bool = (
            self.loader.stopped
            or self.chunker.stopped
            or self.embedder.stopped
            or self.indexer.stopped
            or self.retriver.stopped
            or self.reranker.stopped
            or self.generator.stopped
            or self.agent.stopped
            or self.stopped
        )
        return output

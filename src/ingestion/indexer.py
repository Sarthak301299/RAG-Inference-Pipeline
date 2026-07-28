import logging
import os
from typing import cast
from uuid import uuid4

import pgembed
from chromadb import PersistentClient as ChromaClient
from langchain_chroma import Chroma as ChromaVectorStore
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGEngine, PGVectorStore
from langchain_postgres.v2.indexes import DistanceStrategy
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Indexer:
    def __init__(
        self, config: dict[str, str], embedding_dim: int, embedding_model: Embeddings
    ) -> None:
        self.stopped = True
        self.database: QdrantVectorStore | ChromaVectorStore | PGVectorStore
        try:
            self.vdbname = config["vdbname"]
            if self.vdbname not in ["qdrant", "chroma", "pgvector"]:
                raise ValueError("Vector Database must be qdrant, chroma, or pgvector.")
            self.vdbpath = os.path.join(config["vdbpath"], self.vdbname)
            self.distmetname = config["distance_metric"]
            if self.distmetname not in ["cosine", "l2", "ip"]:
                raise ValueError("Distance metric must be cosine, l2, or ip.")
            if self.distmetname == "cosine":
                self.distance_metric = Distance.COSINE
            elif self.distmetname == "l2":
                self.distance_metric = Distance.EUCLID
            else:
                self.distance_metric = Distance.DOT
        except Exception as e:
            logger.error(f"Got Exception {e} while setting vectorDB.")
            raise
        try:
            if self.vdbname == "qdrant":
                self.vectorDB = QdrantClient(path=self.vdbpath)
                self.collection_name = "qdrant_collection"
                if not self.vectorDB.collection_exists(
                    collection_name=self.collection_name
                ):
                    self.vectorDB.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=embedding_dim, distance=self.distance_metric
                        ),
                    )
                self.database = QdrantVectorStore(
                    client=self.vectorDB,
                    collection_name=self.collection_name,
                    embedding=embedding_model,
                )
            elif self.vdbname == "chroma":
                self.vectorDB = ChromaClient(path=self.vdbpath)
                self.collection_name = "chroma_collection"
                self.vectorDB.get_or_create_collection(
                    name=self.collection_name, metadata={"hsnw:space": self.distmetname}
                )
                self.database = ChromaVectorStore(
                    client=self.vectorDB,
                    collection_name=self.collection_name,
                    embedding_function=embedding_model,
                )
            else:
                self.collection_name = "pgvector_collection"
                absolute_vdb_path = os.path.abspath(self.vdbpath)
                try:
                    self.pg_server = pgembed.postgres_server.get_server(
                        pgdata=absolute_vdb_path, cleanup_mode="delete"
                    )
                    self.pg_server.psql("CREATE EXTENSION IF NOT EXISTS vector;")
                    self.vectorDB = self.pg_server.get_uri()
                except Exception as e:
                    logger.error(
                        f"Failed to isolate or run local pgembed instance: {e}"
                    )
                    raise

                distance_strategy = DistanceStrategy.COSINE_DISTANCE
                if self.distmetname == "l2":
                    distance_strategy = DistanceStrategy.EUCLIDEAN
                elif self.distmetname == "ip":
                    distance_strategy = DistanceStrategy.INNER_PRODUCT

                try:
                    engine = PGEngine.from_connection_string(url=self.vectorDB)
                    engine.init_vectorstore_table(
                        table_name=self.collection_name, vector_size=embedding_dim
                    )
                    self.database = PGVectorStore.create_sync(
                        engine=engine,
                        table_name=self.collection_name,
                        embedding_service=embedding_model,
                        distance_strategy=distance_strategy,
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize PGEngine or VectorStore: {e}")
                    raise
        except Exception as e:
            logger.error(f"Got Exception {e} while initializing vector stores")
            raise
        self.stopped = False

    def _index_qdrant(
        self,
        database: QdrantVectorStore,
        texts: list[str],
        embeddings: list[list[float]],
        ids: list[str],
    ) -> None:
        points = [
            PointStruct(
                id=document_id,
                vector=embedding,
                payload={
                    "page_content": text,
                },
            )
            for text, embedding, document_id in zip(
                texts,
                embeddings,
                ids,
            )
        ]

        database.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def _index_chroma(
        self,
        database: ChromaVectorStore,
        texts: list[str],
        embeddings: list[list[float]],
        ids: list[str],
        metadatas: list[dict[str, str]],
    ) -> None:
        from chromadb.api.types import Embeddings, Metadatas

        collection = database._client.get_or_create_collection(
            name=self.collection_name
        )

        collection.add(
            ids=ids,
            embeddings=cast(Embeddings, embeddings),
            metadatas=cast(Metadatas, metadatas),
            documents=texts,
        )

    def _index_pgvector(
        self,
        database: PGVectorStore,
        texts: list[str],
        embeddings: list[list[float]],
        ids: list[str],
        metadatas: list[dict[str, str]],
    ) -> None:
        database.add_embeddings(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def index_into_vectorDB(self, inputs: list[tuple[str, list[float]]]) -> None:
        if self.stopped:
            raise RuntimeError("Indexer is stopped.")
        try:
            texts: list[str] = []
            embs: list[list[float]] = []
            ids: list[str] = []
            metadatas: list[dict[str, str]] = []
            for text, embedding in inputs:
                texts.append(text)
                embs.append(embedding)
                ids.append(str(uuid4()))
                metadatas.append({"page_content": text})
        except Exception as e:
            logger.error(f"Got Exception {e} while parsing texts, embeddings, and ids")
            raise

        if isinstance(self.database, QdrantVectorStore):
            self._index_qdrant(
                database=self.database,
                texts=texts,
                embeddings=embs,
                ids=ids,
            )
        elif isinstance(self.database, ChromaVectorStore):
            self._index_chroma(
                database=self.database,
                texts=texts,
                embeddings=embs,
                ids=ids,
                metadatas=metadatas,
            )
        else:
            self._index_pgvector(
                database=self.database,
                texts=texts,
                embeddings=embs,
                ids=ids,
                metadatas=metadatas,
            )

    def cleanup(self) -> None:
        self.stopped = True

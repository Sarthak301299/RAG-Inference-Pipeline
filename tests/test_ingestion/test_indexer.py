# tests/test_indexer.py

import logging
import os

import pytest

from src.ingestion import indexer


class TestIndexerInitialization:
    def test_rejects_invalid_vector_database(self):
        config = {
            "vdbname": "invalid",
            "vdbpath": "/tmp/vector-db",
            "distance_metric": "cosine",
        }

        with pytest.raises(
            ValueError,
            match="Vector Database must be qdrant, chroma, or pgvector",
        ):
            indexer.Indexer(
                config=config,
                embedding_dim=384,
                embedding_model=object(),  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore
            )

    def test_invalid_vector_database_is_logged(
        self,
        caplog,
    ):
        config = {
            "vdbname": "invalid",
            "vdbpath": "/tmp/vector-db",
            "distance_metric": "cosine",
        }

        with caplog.at_level(logging.ERROR), pytest.raises(ValueError):
            indexer.Indexer(
                config=config,
                embedding_dim=384,
                embedding_model=object(),  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore
            )

        assert "while setting vectorDB" in caplog.text

    def test_rejects_invalid_distance_metric(self):
        config = {
            "vdbname": "qdrant",
            "vdbpath": "/tmp/vector-db",
            "distance_metric": "invalid",
        }

        with pytest.raises(
            ValueError,
            match="Distance metric must be cosine, l2, or ip",
        ):
            indexer.Indexer(
                config=config,
                embedding_dim=384,
                embedding_model=object(),  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore
            )

    def test_missing_configuration_key_raises_key_error(self):
        config = {
            "vdbname": "qdrant",
            "vdbpath": "/tmp/vector-db",
        }

        with pytest.raises(KeyError):
            indexer.Indexer(
                config=config,
                embedding_dim=384,
                embedding_model=object(),  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore
            )


class TestQdrantInitialization:
    @pytest.mark.parametrize(
        ("metric", "expected_distance"),
        [
            ("cosine", "COSINE"),
            ("l2", "EUCLID"),
            ("ip", "DOT"),
        ],
    )
    def test_initializes_with_correct_distance(
        self,
        monkeypatch,
        tmp_path,
        metric,
        expected_distance,
    ):
        captured = {}

        class FakeQdrantClient:
            def __init__(self, **kwargs):
                captured["client_kwargs"] = kwargs

            def collection_exists(self, **kwargs):
                return False

            def create_collection(self, **kwargs):
                captured["create_kwargs"] = kwargs

        class FakeQdrantVectorStore:
            def __init__(self, **kwargs):
                captured["store_kwargs"] = kwargs

        monkeypatch.setattr(
            indexer,
            "QdrantClient",
            FakeQdrantClient,
        )
        monkeypatch.setattr(
            indexer,
            "QdrantVectorStore",
            FakeQdrantVectorStore,
        )

        instance = indexer.Indexer(
            config={
                "vdbname": "qdrant",
                "vdbpath": str(tmp_path),
                "distance_metric": metric,
            },
            embedding_dim=384,
            embedding_model="embedding-model",  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore
        )

        assert instance.stopped is False
        assert instance.vdbname == "qdrant"
        assert instance.vdbpath == os.path.join(tmp_path, "qdrant")
        assert instance.distmetname == metric
        assert instance.collection_name == "qdrant_collection"

        assert captured["client_kwargs"] == {"path": os.path.join(tmp_path, "qdrant")}

        vectors_config = captured["create_kwargs"]["vectors_config"]

        assert vectors_config.size == 384
        assert vectors_config.distance == getattr(
            indexer.Distance,
            expected_distance,
        )

        assert captured["store_kwargs"] == {
            "client": instance.vectorDB,
            "collection_name": "qdrant_collection",
            "embedding": "embedding-model",
        }

    def test_does_not_create_existing_collection(
        self,
        monkeypatch,
        tmp_path,
    ):
        create_called = False

        class FakeQdrantClient:
            def __init__(self, **kwargs):
                pass

            def collection_exists(self, **kwargs):
                return True

            def create_collection(self, **kwargs):
                nonlocal create_called
                create_called = True

        class FakeQdrantVectorStore:
            def __init__(self, **kwargs):
                pass

        monkeypatch.setattr(
            indexer,
            "QdrantClient",
            FakeQdrantClient,
        )
        monkeypatch.setattr(
            indexer,
            "QdrantVectorStore",
            FakeQdrantVectorStore,
        )

        indexer.Indexer(
            config={
                "vdbname": "qdrant",
                "vdbpath": str(tmp_path),
                "distance_metric": "cosine",
            },
            embedding_dim=384,
            embedding_model="embedding-model",  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore
        )

        assert create_called is False


class TestChromaInitialization:
    def test_initializes_correctly(
        self,
        monkeypatch,
        tmp_path,
    ):
        captured = {}

        class FakeChromaClient:
            def __init__(self, **kwargs):
                captured["client_kwargs"] = kwargs

            def get_or_create_collection(self, **kwargs):
                captured["collection_kwargs"] = kwargs

        class FakeChromaVectorStore:
            def __init__(self, **kwargs):
                captured["store_kwargs"] = kwargs

        monkeypatch.setattr(
            indexer,
            "ChromaClient",
            FakeChromaClient,
        )
        monkeypatch.setattr(
            indexer,
            "ChromaVectorStore",
            FakeChromaVectorStore,
        )

        instance = indexer.Indexer(
            config={
                "vdbname": "chroma",
                "vdbpath": str(tmp_path),
                "distance_metric": "cosine",
            },
            embedding_dim=384,
            embedding_model="embedding-model",  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore
        )

        assert instance.stopped is False
        assert instance.vdbname == "chroma"
        assert instance.vdbpath == os.path.join(tmp_path, "chroma")
        assert instance.collection_name == "chroma_collection"

        assert captured["client_kwargs"] == {"path": os.path.join(tmp_path, "chroma")}

        assert captured["collection_kwargs"] == {
            "name": "chroma_collection",
            "metadata": {
                "hnsw:space": "cosine",
            },
        }

        assert captured["store_kwargs"] == {
            "client": instance.vectorDB,
            "collection_name": "chroma_collection",
            "embedding_function": "embedding-model",
        }


class TestPGVectorInitialization:
    @pytest.mark.parametrize(
        ("metric", "expected_strategy"),
        [
            ("cosine", "COSINE_DISTANCE"),
            ("l2", "EUCLIDEAN"),
            ("ip", "INNER_PRODUCT"),
        ],
    )
    def test_initializes_with_correct_distance_strategy(
        self,
        monkeypatch,
        tmp_path,
        metric,
        expected_strategy,
    ):
        captured = {}

        class FakePGServer:
            def psql(self, query):
                captured["psql_query"] = query

            def get_uri(self):
                return "postgresql://fake"

        class FakePostgresServer:
            def get_server(self, **kwargs):
                captured["server_kwargs"] = kwargs
                return FakePGServer()

        class FakePGEmbed:
            postgres_server = FakePostgresServer()

        class FakeEngine:
            @classmethod
            def from_connection_string(cls, **kwargs):
                captured["engine_kwargs"] = kwargs
                return cls()

            def init_vectorstore_table(self, **kwargs):
                captured["table_kwargs"] = kwargs

        class FakePGVectorStore:
            @classmethod
            def create_sync(cls, **kwargs):
                captured["store_kwargs"] = kwargs
                return "pgvector-store"

        monkeypatch.setattr(
            indexer,
            "pgembed",
            FakePGEmbed(),
        )
        monkeypatch.setattr(
            indexer,
            "PGEngine",
            FakeEngine,
        )
        monkeypatch.setattr(
            indexer,
            "PGVectorStore",
            FakePGVectorStore,
        )

        instance = indexer.Indexer(
            config={
                "vdbname": "pgvector",
                "vdbpath": str(tmp_path),
                "distance_metric": metric,
            },
            embedding_dim=768,
            embedding_model="embedding-model",  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore
        )

        assert instance.stopped is False
        assert instance.collection_name == "pgvector_collection"
        assert instance.vectorDB == "postgresql://fake"
        assert instance.database == "pgvector-store"

        assert captured["server_kwargs"] == {
            "pgdata": str(tmp_path / "pgvector"),
            "cleanup_mode": "delete",
        }

        assert captured["psql_query"] == ("CREATE EXTENSION IF NOT EXISTS vector;")

        assert captured["engine_kwargs"] == {
            "url": "postgresql://fake",
        }

        assert captured["table_kwargs"] == {
            "table_name": "pgvector_collection",
            "vector_size": 768,
        }

        assert captured["store_kwargs"]["table_name"] == ("pgvector_collection")

        assert captured["store_kwargs"]["embedding_service"] == ("embedding-model")

        assert captured["store_kwargs"]["distance_strategy"] == getattr(
            indexer.DistanceStrategy,
            expected_strategy,
        )

    def test_pgembed_failure_is_logged(
        self,
        monkeypatch,
        tmp_path,
        caplog,
    ):
        class FakePostgresServer:
            def get_server(self, **kwargs):
                raise RuntimeError("pgembed failure")

        class FakePGEmbed:
            postgres_server = FakePostgresServer()

        monkeypatch.setattr(
            indexer,
            "pgembed",
            FakePGEmbed(),
        )

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(
                RuntimeError,
                match="pgembed failure",
            ),
        ):
            indexer.Indexer(
                config={
                    "vdbname": "pgvector",
                    "vdbpath": str(tmp_path),
                    "distance_metric": "cosine",
                },
                embedding_dim=384,
                embedding_model="embedding-model",  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore
            )

        assert "Failed to isolate or run local pgembed instance" in caplog.text

    def test_pgengine_failure_is_logged(
        self,
        monkeypatch,
        tmp_path,
        caplog,
    ):
        class FakePGServer:
            def psql(self, query):
                pass

            def get_uri(self):
                return "postgresql://fake"

        class FakePostgresServer:
            def get_server(self, **kwargs):
                return FakePGServer()

        class FakePGEmbed:
            postgres_server = FakePostgresServer()

        class FakePGEngine:
            @classmethod
            def from_connection_string(cls, **kwargs):
                raise RuntimeError("engine failure")

        monkeypatch.setattr(
            indexer,
            "pgembed",
            FakePGEmbed(),
        )
        monkeypatch.setattr(
            indexer,
            "PGEngine",
            FakePGEngine,
        )

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(
                RuntimeError,
                match="engine failure",
            ),
        ):
            indexer.Indexer(
                config={
                    "vdbname": "pgvector",
                    "vdbpath": str(tmp_path),
                    "distance_metric": "cosine",
                },
                embedding_dim=384,
                embedding_model="embedding-model",  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore
            )

        assert "Failed to initialize PGEngine or VectorStore" in caplog.text


class TestBackendSpecificIndexing:
    @pytest.fixture
    def indexer_instance(self):
        instance = indexer.Indexer.__new__(indexer.Indexer)

        instance.stopped = False
        instance.collection_name = "test_collection"

        return instance

    def test_index_qdrant(
        self,
        indexer_instance,
    ):
        captured = {}

        class FakeClient:
            def upsert(self, **kwargs):
                captured.update(kwargs)

        class FakeQdrantDatabase:
            client = FakeClient()

        indexer_instance._index_qdrant(
            database=FakeQdrantDatabase(),
            texts=[
                "first",
                "second",
            ],
            embeddings=[
                [0.1, 0.2],
                [0.3, 0.4],
            ],
            ids=[
                "id-1",
                "id-2",
            ],
            metadatas=[
                {"source": "a.txt"},
                {"source": "b.txt"},
            ],
        )

        assert captured["collection_name"] == ("test_collection")

        points = captured["points"]

        assert len(points) == 2

        assert points[0].id == "id-1"
        assert points[0].vector == [0.1, 0.2]
        assert points[0].payload == {
            "page_content": "first",
            "metadata": {"source": "a.txt"},
        }

        assert points[1].id == "id-2"
        assert points[1].vector == [0.3, 0.4]
        assert points[1].payload == {
            "page_content": "second",
            "metadata": {"source": "b.txt"},
        }

    def test_index_qdrant_with_empty_input(
        self,
        indexer_instance,
    ):
        captured = {}

        class FakeClient:
            def upsert(self, **kwargs):
                captured.update(kwargs)

        class FakeQdrantDatabase:
            client = FakeClient()

        indexer_instance._index_qdrant(
            database=FakeQdrantDatabase(), texts=[], embeddings=[], ids=[], metadatas=[]
        )

        assert captured["collection_name"] == ("test_collection")

        assert captured["points"] == []

    def test_index_chroma(
        self,
        indexer_instance,
    ):
        captured = {}

        class FakeCollection:
            def add(self, **kwargs):
                captured.update(kwargs)

        class FakeClient:
            def get_or_create_collection(self, **kwargs):
                captured["collection_kwargs"] = kwargs
                return FakeCollection()

        class FakeChromaDatabase:
            _client = FakeClient()

        indexer_instance._index_chroma(
            database=FakeChromaDatabase(),
            texts=[
                "first",
                "second",
            ],
            embeddings=[
                [0.1, 0.2],
                [0.3, 0.4],
            ],
            ids=[
                "id-1",
                "id-2",
            ],
            metadatas=[
                {"source": "file1"},
                {"author": "alice"},
            ],
        )

        assert captured["collection_kwargs"] == {
            "name": "test_collection",
        }

        assert captured["ids"] == [
            "id-1",
            "id-2",
        ]

        assert captured["embeddings"] == [
            [0.1, 0.2],
            [0.3, 0.4],
        ]

        assert captured["metadatas"] == [
            {"source": "file1"},
            {"author": "alice"},
        ]

        assert captured["documents"] == [
            "first",
            "second",
        ]

    def test_index_pgvector(
        self,
        indexer_instance,
    ):
        captured = {}

        class FakePGVectorDatabase:
            def add_embeddings(self, **kwargs):
                captured.update(kwargs)

        indexer_instance._index_pgvector(
            database=FakePGVectorDatabase(),
            texts=[
                "first",
                "second",
            ],
            embeddings=[
                [0.1, 0.2],
                [0.3, 0.4],
            ],
            ids=[
                "id-1",
                "id-2",
            ],
            metadatas=[
                {"source": "file1"},
                {"author": "alice"},
            ],
        )

        assert captured == {
            "texts": [
                "first",
                "second",
            ],
            "embeddings": [
                [0.1, 0.2],
                [0.3, 0.4],
            ],
            "ids": [
                "id-1",
                "id-2",
            ],
            "metadatas": [
                {"source": "file1"},
                {"author": "alice"},
            ],
        }


class TestIndexIntoVectorDB:
    @pytest.fixture
    def base_indexer(self):
        instance = indexer.Indexer.__new__(indexer.Indexer)
        instance.stopped = False
        instance.collection_name = "test_collection"
        return instance

    def test_raises_when_stopped(
        self,
        base_indexer,
    ):
        base_indexer.stopped = True

        with pytest.raises(
            RuntimeError,
            match="Indexer is stopped.",
        ):
            base_indexer.index_into_vectorDB(
                [
                    ("text", [0.1, 0.2]),
                ],
                [
                    {"source": "file1"},
                ],
            )

    def test_dispatches_to_qdrant(
        self,
        monkeypatch,
        base_indexer,
    ):
        captured = {}

        class FakeQdrantStore:
            pass

        database = FakeQdrantStore()
        base_indexer.database = database

        def fake_index_qdrant(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(
            indexer,
            "QdrantVectorStore",
            FakeQdrantStore,
        )

        base_indexer._index_qdrant = fake_index_qdrant

        monkeypatch.setattr(
            indexer,
            "uuid4",
            lambda: "generated-id",
        )

        base_indexer.index_into_vectorDB(
            [
                ("hello", [0.1, 0.2]),
            ],
            [
                {"source": "file.txt"},
            ],
        )

        assert captured["database"] is database
        assert captured["texts"] == ["hello"]
        assert captured["embeddings"] == [[0.1, 0.2]]
        assert captured["ids"] == ["generated-id"]

    def test_dispatches_to_chroma(
        self,
        monkeypatch,
        base_indexer,
    ):
        captured = {}

        class FakeChromaStore:
            pass

        database = FakeChromaStore()
        base_indexer.database = database

        def fake_index_chroma(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(
            indexer,
            "QdrantVectorStore",
            type("OtherStore", (), {}),
        )

        monkeypatch.setattr(
            indexer,
            "ChromaVectorStore",
            FakeChromaStore,
        )

        base_indexer._index_chroma = fake_index_chroma

        monkeypatch.setattr(
            indexer,
            "uuid4",
            lambda: "generated-id",
        )

        base_indexer.index_into_vectorDB(
            [
                ("hello", [0.1, 0.2]),
            ],
            [
                {"source": "file.txt"},
            ],
        )

        assert captured["database"] is database
        assert captured["texts"] == ["hello"]
        assert captured["embeddings"] == [[0.1, 0.2]]
        assert captured["ids"] == ["generated-id"]
        assert captured["metadatas"] == [
            {
                "source": "file.txt",
            }
        ]

    def test_dispatches_to_pgvector(
        self,
        monkeypatch,
        base_indexer,
    ):
        captured = {}

        class FakePGVectorStore:
            pass

        database = FakePGVectorStore()
        base_indexer.database = database

        def fake_index_pgvector(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(
            indexer,
            "PGVectorStore",
            FakePGVectorStore,
        )

        base_indexer._index_pgvector = fake_index_pgvector

        monkeypatch.setattr(
            indexer,
            "uuid4",
            lambda: "generated-id",
        )

        base_indexer.index_into_vectorDB(
            [
                ("hello", [0.1, 0.2]),
            ],
            [
                {"source": "file.txt"},
            ],
        )

        assert captured["database"] is database
        assert captured["texts"] == ["hello"]
        assert captured["embeddings"] == [[0.1, 0.2]]
        assert captured["ids"] == ["generated-id"]
        assert captured["metadatas"] == [
            {
                "source": "file.txt",
            }
        ]

    def test_empty_input_dispatches_empty_lists(
        self,
        monkeypatch,
        base_indexer,
    ):
        captured = {}

        class FakePGVectorStore:
            pass

        database = FakePGVectorStore()
        base_indexer.database = database

        monkeypatch.setattr(
            indexer,
            "PGVectorStore",
            FakePGVectorStore,
        )

        def fake_index_pgvector(**kwargs):
            captured.update(kwargs)

        base_indexer._index_pgvector = fake_index_pgvector

        base_indexer.index_into_vectorDB([], [])

        assert captured["texts"] == []
        assert captured["embeddings"] == []
        assert captured["ids"] == []
        assert captured["metadatas"] == []

    def test_malformed_input_is_logged_and_reraised(
        self,
        base_indexer,
        caplog,
    ):
        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(
                (ValueError, TypeError),
            ),
        ):
            base_indexer.index_into_vectorDB(
                [
                    ("text-only",),
                ],
                [{"source": "file.txt"}],
            )

        assert "while parsing texts, embeddings, and ids" in caplog.text

    def test_uuid_generation_failure_is_logged_and_reraised(
        self,
        monkeypatch,
        base_indexer,
        caplog,
    ):
        def raise_uuid_error():
            raise RuntimeError("UUID generation failed")

        monkeypatch.setattr(
            indexer,
            "uuid4",
            raise_uuid_error,
        )

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(
                RuntimeError,
                match="UUID generation failed",
            ),
        ):
            base_indexer.index_into_vectorDB(
                [
                    ("text", [0.1, 0.2]),
                ],
                [{"source": "file.txt"}],
            )

        assert "while parsing texts, embeddings, and ids" in caplog.text


class FakeClient:
    def __init__(self):
        self.collection = "abc"

    def delete_collection(self, input):
        del self.collection

    def drop_table(self, table_name):
        del self.collection


class FakeVectorStore:
    def __init__(self):
        self.client = FakeClient()
        self._client = FakeClient()
        self._engine = FakeClient()


class TestCleanup:
    def test_cleanup_stops_indexer_qdrant(self, monkeypatch):
        monkeypatch.setattr(
            indexer,
            "QdrantVectorStore",
            FakeVectorStore,
        )
        instance = indexer.Indexer.__new__(indexer.Indexer)
        instance.stopped = False
        instance.collection_name = "abc"
        instance.database = FakeVectorStore()  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore

        instance.cleanup()

        assert instance.stopped is True
        assert not hasattr(instance.database.client, "collection")  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore

    def test_cleanup_stops_indexer_chroma(self, monkeypatch):
        monkeypatch.setattr(
            indexer,
            "ChromaVectorStore",
            FakeVectorStore,
        )
        instance = indexer.Indexer.__new__(indexer.Indexer)
        instance.stopped = False
        instance.collection_name = "abc"
        instance.database = FakeVectorStore()  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore

        instance.cleanup()

        assert instance.stopped is True
        assert not hasattr(instance.database._client, "collection")  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore

    def test_cleanup_stops_indexer_pgvector(self, monkeypatch):
        monkeypatch.setattr(
            indexer,
            "PGVectorStore",
            FakeVectorStore,
        )
        instance = indexer.Indexer.__new__(indexer.Indexer)
        instance.stopped = False
        instance.collection_name = "abc"
        instance.database = FakeVectorStore()  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore

        instance.cleanup()

        assert instance.stopped is True
        assert not hasattr(instance.database._engine, "collection")  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore

    def test_cleanup_is_idempotent(self):
        instance = indexer.Indexer.__new__(indexer.Indexer)
        instance.stopped = False
        instance.collection_name = "abc"
        instance.database = FakeVectorStore()  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore

        instance.cleanup()
        instance.cleanup()

        assert instance.stopped is True

# tests/test_embedder.py

import logging

import pytest

from src.ingestion import embedder


@pytest.fixture
def valid_config():
    return {
        "model": "test-model",
        "batch_size": "32",
        "normalize": "true",
    }


@pytest.fixture
def uninitialized_embedder():
    """
    Creates an Embedder instance without running __init__.

    Useful for testing create_embedding() and cleanup() independently.
    """
    instance = embedder.Embedder.__new__(embedder.Embedder)
    instance.stopped = False
    return instance


class TestEmbedderConfiguration:
    def test_reads_configuration_correctly(
        self,
        monkeypatch,
        valid_config,
    ):
        class FakeClient:
            def get_sentence_embedding_dimension(self):
                return 384

        class FakeModel:
            def __init__(self, **kwargs):
                self._client = FakeClient()

        monkeypatch.setattr(
            embedder,
            "HuggingFaceEmbeddings",
            FakeModel,
        )

        instance = embedder.Embedder(valid_config)

        assert instance.model_name == "test-model"
        assert instance.batch_size == 32
        assert instance.normalize is True
        assert instance.embedding_dim == 384
        assert instance.stopped is False

    def test_batch_size_is_converted_to_int(
        self,
        monkeypatch,
    ):
        class FakeClient:
            def get_sentence_embedding_dimension(self):
                return 384

        class FakeModel:
            def __init__(self, **kwargs):
                self._client = FakeClient()

        monkeypatch.setattr(
            embedder,
            "HuggingFaceEmbeddings",
            FakeModel,
        )

        instance = embedder.Embedder(
            {
                "model": "test-model",
                "batch_size": "16",
                "normalize": "False",
            }
        )

        assert instance.batch_size == 16
        assert isinstance(instance.batch_size, int)

    @pytest.mark.parametrize(
        "missing_key",
        [
            "model",
            "batch_size",
            "normalize",
        ],
    )
    def test_missing_configuration_key_raises_key_error(
        self,
        monkeypatch,
        missing_key,
        caplog,
    ):
        config = {
            "model": "test-model",
            "batch_size": "32",
            "normalize": True,
        }

        del config[missing_key]

        with caplog.at_level(logging.ERROR), pytest.raises(KeyError):
            embedder.Embedder(config)

        assert "reading configuration" in caplog.text

    @pytest.mark.parametrize(
        ("batch_size", "normalize"), [("not-an-integer", "True"), ("32", "invalid")]
    )
    def test_invalid_batch_size_or_normalize_raises_value_error(
        self, caplog, batch_size, normalize
    ):
        config = {
            "model": "test-model",
            "batch_size": batch_size,
            "normalize": normalize,
        }

        with caplog.at_level(logging.ERROR), pytest.raises(ValueError):
            embedder.Embedder(config)

        assert "reading configuration" in caplog.text


class TestModelInitialization:
    def test_model_is_initialized_with_expected_arguments(
        self,
        monkeypatch,
        valid_config,
    ):
        captured = {}

        class FakeClient:
            def get_sentence_embedding_dimension(self):
                return 768

        class FakeModel:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self._client = FakeClient()

        monkeypatch.setattr(
            embedder,
            "HuggingFaceEmbeddings",
            FakeModel,
        )

        instance = embedder.Embedder(valid_config)

        assert captured == {
            "model_name": "test-model",
            "encode_kwargs": {
                "batch_size": 32,
                "normalize_embeddings": True,
            },
        }

        assert instance.model_name == "test-model"
        assert instance.batch_size == 32
        assert instance.normalize is True
        assert instance.embedding_dim == 768

    def test_model_initialization_failure_is_logged(
        self,
        monkeypatch,
        valid_config,
        caplog,
    ):
        class FakeModel:
            def __init__(self, **kwargs):
                raise RuntimeError("model loading failed")

        monkeypatch.setattr(
            embedder,
            "HuggingFaceEmbeddings",
            FakeModel,
        )

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(
                RuntimeError,
                match="model loading failed",
            ),
        ):
            embedder.Embedder(valid_config)

        assert "loading model test-model" in caplog.text

    def test_model_initialization_failure_does_not_start_embedder(
        self,
        monkeypatch,
        valid_config,
    ):
        class FakeModel:
            def __init__(self, **kwargs):
                raise RuntimeError("model loading failed")

        monkeypatch.setattr(
            embedder,
            "HuggingFaceEmbeddings",
            FakeModel,
        )

        with pytest.raises(RuntimeError):
            embedder.Embedder(valid_config)

        # __init__ failed before completion, so this test mainly ensures
        # the exception is not swallowed.


class TestEmbeddingDimension:
    def test_embedding_dimension_is_read_from_model(
        self,
        monkeypatch,
        valid_config,
    ):
        class FakeClient:
            def get_sentence_embedding_dimension(self):
                return 1536

        class FakeModel:
            def __init__(self, **kwargs):
                self._client = FakeClient()

        monkeypatch.setattr(
            embedder,
            "HuggingFaceEmbeddings",
            FakeModel,
        )

        instance = embedder.Embedder(valid_config)

        assert instance.embedding_dim == 1536

    def test_none_embedding_dimension_raises_type_error(
        self,
        monkeypatch,
        valid_config,
        caplog,
    ):
        class FakeClient:
            def get_sentence_embedding_dimension(self):
                return None

        class FakeModel:
            def __init__(self, **kwargs):
                self._client = FakeClient()

        monkeypatch.setattr(
            embedder,
            "HuggingFaceEmbeddings",
            FakeModel,
        )

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(
                TypeError,
                match="Embedding Dimension is None",
            ),
        ):
            embedder.Embedder(valid_config)

        assert "getting embedding dimensions" in caplog.text

    @pytest.mark.parametrize(
        "dimension",
        [
            0,
            -1,
            -384,
        ],
    )
    def test_non_positive_embedding_dimension_raises_value_error(
        self,
        monkeypatch,
        valid_config,
        dimension,
        caplog,
    ):
        class FakeClient:
            def get_sentence_embedding_dimension(self):
                return dimension

        class FakeModel:
            def __init__(self, **kwargs):
                self._client = FakeClient()

        monkeypatch.setattr(
            embedder,
            "HuggingFaceEmbeddings",
            FakeModel,
        )

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(
                ValueError,
                match=f"Invalid Embedding dimensions {dimension}",
            ),
        ):
            embedder.Embedder(valid_config)

        assert "getting embedding dimensions" in caplog.text

    def test_failure_to_get_embedding_dimension_is_logged(
        self,
        monkeypatch,
        valid_config,
        caplog,
    ):
        class FakeClient:
            def get_sentence_embedding_dimension(self):
                raise RuntimeError("dimension lookup failed")

        class FakeModel:
            def __init__(self, **kwargs):
                self._client = FakeClient()

        monkeypatch.setattr(
            embedder,
            "HuggingFaceEmbeddings",
            FakeModel,
        )

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(
                RuntimeError,
                match="dimension lookup failed",
            ),
        ):
            embedder.Embedder(valid_config)

        assert "getting embedding dimensions" in caplog.text


class TestCreateEmbedding:
    def test_raises_when_embedder_is_stopped(
        self,
        uninitialized_embedder,
    ):
        uninitialized_embedder.stopped = True

        with pytest.raises(
            RuntimeError,
            match="Embedder is stopped.",
        ):
            uninitialized_embedder.create_embedding(
                ["hello"],
            )

    def test_creates_embeddings(
        self,
        uninitialized_embedder,
    ):
        captured = {}

        class FakeModel:
            def embed_documents(self, **kwargs):
                captured.update(kwargs)
                return [
                    [0.1, 0.2],
                    [0.3, 0.4],
                ]

        uninitialized_embedder.model = FakeModel()

        result = uninitialized_embedder.create_embedding(
            [
                "first document",
                "second document",
            ]
        )

        assert captured == {
            "texts": [
                "first document",
                "second document",
            ],
        }

        assert result == [
            [0.1, 0.2],
            [0.3, 0.4],
        ]

    def test_empty_input_is_forwarded_to_model(
        self,
        uninitialized_embedder,
    ):
        captured = {}

        class FakeModel:
            def embed_documents(self, **kwargs):
                captured.update(kwargs)
                return []

        uninitialized_embedder.model = FakeModel()

        result = uninitialized_embedder.create_embedding([])

        assert captured == {
            "texts": [],
        }

        assert result == []

    def test_model_embedding_failure_is_logged_and_reraised(
        self,
        uninitialized_embedder,
        caplog,
    ):
        class FakeModel:
            def embed_documents(self, **kwargs):
                raise RuntimeError("embedding failed")

        uninitialized_embedder.model = FakeModel()

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(
                RuntimeError,
                match="embedding failed",
            ),
        ):
            uninitialized_embedder.create_embedding(
                ["hello"],
            )

        assert "creating embedding" in caplog.text

    def test_model_is_not_called_when_stopped(
        self,
        uninitialized_embedder,
    ):
        called = False

        class FakeModel:
            def embed_documents(self, **kwargs):
                nonlocal called
                called = True
                return []

        uninitialized_embedder.model = FakeModel()
        uninitialized_embedder.stopped = True

        with pytest.raises(RuntimeError):
            uninitialized_embedder.create_embedding(["hello"])

        assert called is False


class TestCleanup:
    def test_cleanup_marks_embedder_as_stopped(
        self,
        uninitialized_embedder,
    ):
        uninitialized_embedder.stopped = False

        uninitialized_embedder.cleanup()

        assert uninitialized_embedder.stopped is True

    def test_cleanup_deletes_model(
        self,
        uninitialized_embedder,
    ):
        uninitialized_embedder.model = object()

        uninitialized_embedder.cleanup()

        assert not hasattr(
            uninitialized_embedder,
            "model",
        )

    def test_cleanup_calls_cuda_cleanup_when_cuda_is_available(
        self,
        monkeypatch,
        uninitialized_embedder,
    ):
        calls = []

        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def empty_cache():
                calls.append("empty_cache")

            @staticmethod
            def synchronize():
                calls.append("synchronize")

        monkeypatch.setattr(
            embedder.torch,
            "cuda",
            FakeCuda,
        )

        uninitialized_embedder.model = object()

        monkeypatch.setattr(
            embedder.gc,
            "collect",
            lambda: calls.append("gc.collect"),
        )

        uninitialized_embedder.cleanup()

        assert calls == [
            "empty_cache",
            "synchronize",
            "gc.collect",
        ]

    def test_cleanup_does_not_call_cuda_cleanup_without_cuda(
        self,
        monkeypatch,
        uninitialized_embedder,
    ):
        calls = []

        class FakeCuda:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def empty_cache():
                calls.append("empty_cache")

            @staticmethod
            def synchronize():
                calls.append("synchronize")

        monkeypatch.setattr(
            embedder.torch,
            "cuda",
            FakeCuda,
        )

        monkeypatch.setattr(
            embedder.gc,
            "collect",
            lambda: calls.append("gc.collect"),
        )

        uninitialized_embedder.model = object()

        uninitialized_embedder.cleanup()

        assert calls == [
            "gc.collect",
        ]

    def test_cleanup_is_safe_without_model(
        self,
        monkeypatch,
        uninitialized_embedder,
    ):
        monkeypatch.setattr(
            embedder.torch.cuda,
            "is_available",
            lambda: False,
        )

        uninitialized_embedder.cleanup()

        assert uninitialized_embedder.stopped is True

    def test_cleanup_is_idempotent(
        self,
        monkeypatch,
        uninitialized_embedder,
    ):
        monkeypatch.setattr(
            embedder.torch.cuda,
            "is_available",
            lambda: False,
        )

        uninitialized_embedder.model = object()

        uninitialized_embedder.cleanup()
        uninitialized_embedder.cleanup()

        assert uninitialized_embedder.stopped is True
        assert not hasattr(
            uninitialized_embedder,
            "model",
        )

import pytest
from langchain_core.documents import Document

from src.pipeline.rag_pipeline import RAGPipeLine


def make_open_and_yaml(monkeypatch, config):
    class DummyFile:
        def __enter__(self):
            return object()

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: DummyFile())
    monkeypatch.setattr("src.pipeline.rag_pipeline.yaml.safe_load", lambda _: config)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def config():
    return {
        "ingestion": {
            "loading": {},
            "embedding": {},
            "chunking": {},
        },
        "vectorDB": {
            "vdbname": "qdrant",
            "final_chunk_count": "2",
            "retrival_to_rerank_ratio": "3",
        },
        "retrieval": {
            "reranker": {},
        },
        "generation": {},
    }


@pytest.fixture
def pipeline(monkeypatch, config):
    monkeypatch.setattr(
        "src.pipeline.rag_pipeline.yaml.safe_load",
        lambda f: config,
    )

    class DummyFile:
        def __enter__(self):
            return object()

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(
        "builtins.open",
        lambda *args, **kwargs: DummyFile(),
    )

    class DummyLoader:
        def __init__(self, cfg):
            pass

    class DummyEmbedder:
        def __init__(self, cfg):
            self.embedding_dim = 4
            self.model = object()

    class DummyIndexer:
        def __init__(self, *args):
            self.database = object()

        def index_into_vectorDB(self, *args):
            self.called = args

    class DummyChunker:
        def __init__(self, cfg):
            pass

    class DummyRetriever:
        def __init__(self, *args):
            pass

    class DummyReranker:
        def __init__(self, cfg):
            pass

    class DummyGenerator:
        def __init__(self, cfg):
            pass

    monkeypatch.setattr("src.pipeline.rag_pipeline.Loader", DummyLoader)
    monkeypatch.setattr("src.pipeline.rag_pipeline.Embedder", DummyEmbedder)
    monkeypatch.setattr("src.pipeline.rag_pipeline.Indexer", DummyIndexer)
    monkeypatch.setattr("src.pipeline.rag_pipeline.Chunker", DummyChunker)
    monkeypatch.setattr("src.pipeline.rag_pipeline.Retriever", DummyRetriever)
    monkeypatch.setattr("src.pipeline.rag_pipeline.Reranker", DummyReranker)
    monkeypatch.setattr("src.pipeline.rag_pipeline.VLLMGenerator", DummyGenerator)

    return RAGPipeLine()


# ---------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------


def test_init_success(pipeline):
    assert pipeline.stopped is False
    assert pipeline.retrieve_chunk_count == 6
    assert pipeline.final_chunk_count == 2


def test_init_bad_config(monkeypatch):
    monkeypatch.setattr(
        "builtins.open",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    with pytest.raises(FileNotFoundError):
        RAGPipeLine()


def test_init_loader_failure(monkeypatch, config):
    make_open_and_yaml(monkeypatch, config)

    monkeypatch.setattr(
        "src.pipeline.rag_pipeline.Loader",
        lambda cfg: (_ for _ in ()).throw(RuntimeError("loader")),
    )

    with pytest.raises(RuntimeError, match="loader"):
        RAGPipeLine()


def test_init_embedder_failure(monkeypatch, config):
    make_open_and_yaml(monkeypatch, config)

    class DummyLoader:
        def __init__(self, cfg):
            pass

    monkeypatch.setattr("src.pipeline.rag_pipeline.Loader", DummyLoader)

    monkeypatch.setattr(
        "src.pipeline.rag_pipeline.Embedder",
        lambda cfg: (_ for _ in ()).throw(RuntimeError("embedder")),
    )

    with pytest.raises(RuntimeError, match="embedder"):
        RAGPipeLine()


def test_init_none_embedding_dimension(monkeypatch, config):
    make_open_and_yaml(monkeypatch, config)

    class DummyLoader:
        def __init__(self, cfg):
            pass

    class DummyEmbedder:
        def __init__(self, cfg):
            self.embedding_dim = None
            self.model = object()

    monkeypatch.setattr("src.pipeline.rag_pipeline.Loader", DummyLoader)
    monkeypatch.setattr("src.pipeline.rag_pipeline.Embedder", DummyEmbedder)

    with pytest.raises(TypeError, match="Embedding"):
        RAGPipeLine()


def test_init_indexer_failure(monkeypatch, config):
    make_open_and_yaml(monkeypatch, config)

    class DummyLoader:
        def __init__(self, cfg):
            pass

    class DummyEmbedder:
        def __init__(self, cfg):
            self.embedding_dim = 384
            self.model = object()

    monkeypatch.setattr("src.pipeline.rag_pipeline.Loader", DummyLoader)
    monkeypatch.setattr("src.pipeline.rag_pipeline.Embedder", DummyEmbedder)

    monkeypatch.setattr(
        "src.pipeline.rag_pipeline.Indexer",
        lambda *args: (_ for _ in ()).throw(RuntimeError("indexer")),
    )

    with pytest.raises(RuntimeError, match="indexer"):
        RAGPipeLine()


def test_init_chunker_failure(monkeypatch, config):
    make_open_and_yaml(monkeypatch, config)

    class DummyLoader:
        def __init__(self, cfg):
            pass

    class DummyEmbedder:
        def __init__(self, cfg):
            self.embedding_dim = 384
            self.model = object()

    class DummyIndexer:
        def __init__(self, *args):
            self.database = object()

    monkeypatch.setattr("src.pipeline.rag_pipeline.Loader", DummyLoader)
    monkeypatch.setattr("src.pipeline.rag_pipeline.Embedder", DummyEmbedder)
    monkeypatch.setattr("src.pipeline.rag_pipeline.Indexer", DummyIndexer)

    monkeypatch.setattr(
        "src.pipeline.rag_pipeline.Chunker",
        lambda cfg: (_ for _ in ()).throw(RuntimeError("chunker")),
    )

    with pytest.raises(RuntimeError, match="chunker"):
        RAGPipeLine()


def test_init_retriever_failure(monkeypatch, config):
    make_open_and_yaml(monkeypatch, config)

    class DummyLoader:
        def __init__(self, cfg):
            pass

    class DummyEmbedder:
        def __init__(self, cfg):
            self.embedding_dim = 384
            self.model = object()

    class DummyIndexer:
        def __init__(self, *args):
            self.database = object()

    class DummyChunker:
        def __init__(self, cfg):
            pass

    monkeypatch.setattr("src.pipeline.rag_pipeline.Loader", DummyLoader)
    monkeypatch.setattr("src.pipeline.rag_pipeline.Embedder", DummyEmbedder)
    monkeypatch.setattr("src.pipeline.rag_pipeline.Indexer", DummyIndexer)
    monkeypatch.setattr("src.pipeline.rag_pipeline.Chunker", DummyChunker)

    monkeypatch.setattr(
        "src.pipeline.rag_pipeline.Retriever",
        lambda *args: (_ for _ in ()).throw(RuntimeError("retriever")),
    )

    with pytest.raises(RuntimeError, match="retriever"):
        RAGPipeLine()


def test_init_reranker_failure(monkeypatch, config):
    make_open_and_yaml(monkeypatch, config)

    class DummyLoader:
        def __init__(self, cfg):
            pass

    class DummyEmbedder:
        def __init__(self, cfg):
            self.embedding_dim = 384
            self.model = object()

    class DummyIndexer:
        def __init__(self, *args):
            self.database = object()

    class DummyChunker:
        def __init__(self, cfg):
            pass

    class DummyRetriever:
        def __init__(self, *args):
            pass

    monkeypatch.setattr("src.pipeline.rag_pipeline.Loader", DummyLoader)
    monkeypatch.setattr("src.pipeline.rag_pipeline.Embedder", DummyEmbedder)
    monkeypatch.setattr("src.pipeline.rag_pipeline.Indexer", DummyIndexer)
    monkeypatch.setattr("src.pipeline.rag_pipeline.Chunker", DummyChunker)
    monkeypatch.setattr("src.pipeline.rag_pipeline.Retriever", DummyRetriever)

    monkeypatch.setattr(
        "src.pipeline.rag_pipeline.Reranker",
        lambda cfg: (_ for _ in ()).throw(RuntimeError("reranker")),
    )

    with pytest.raises(RuntimeError, match="reranker"):
        RAGPipeLine()


def test_init_generator_failure(monkeypatch, config):
    make_open_and_yaml(monkeypatch, config)

    class DummyLoader:
        def __init__(self, cfg):
            pass

    class DummyEmbedder:
        def __init__(self, cfg):
            self.embedding_dim = 384
            self.model = object()

    class DummyIndexer:
        def __init__(self, *args):
            self.database = object()

    class DummyChunker:
        def __init__(self, cfg):
            pass

    class DummyRetriever:
        def __init__(self, *args):
            pass

    class DummyReranker:
        def __init__(self, cfg):
            pass

    monkeypatch.setattr("src.pipeline.rag_pipeline.Loader", DummyLoader)
    monkeypatch.setattr("src.pipeline.rag_pipeline.Embedder", DummyEmbedder)
    monkeypatch.setattr("src.pipeline.rag_pipeline.Indexer", DummyIndexer)
    monkeypatch.setattr("src.pipeline.rag_pipeline.Chunker", DummyChunker)
    monkeypatch.setattr("src.pipeline.rag_pipeline.Retriever", DummyRetriever)
    monkeypatch.setattr("src.pipeline.rag_pipeline.Reranker", DummyReranker)

    monkeypatch.setattr(
        "src.pipeline.rag_pipeline.VLLMGenerator",
        lambda cfg: (_ for _ in ()).throw(RuntimeError("generator")),
    )

    with pytest.raises(RuntimeError, match="generator"):
        RAGPipeLine()


# ---------------------------------------------------------------------
# ingest_data
# ---------------------------------------------------------------------


def test_ingest_data_success(pipeline):
    docs = {
        ".txt": [
            Document(
                page_content="abc",
                metadata={"source": "file"},
            )
        ]
    }

    pipeline.loader.parse_docs = lambda: docs

    pipeline.chunker.generate_chunks = lambda docs, ext: docs

    pipeline.embedder.create_embedding = lambda texts: [[1.0]]

    called = {}

    def fake_index(inputs, metadata):
        called["inputs"] = inputs
        called["metadata"] = metadata

    pipeline.indexer.index_into_vectorDB = fake_index

    pipeline.ingest_data()

    assert called["inputs"] == [("abc", [1.0])]
    assert called["metadata"] == [{"source": "file"}]


def test_ingest_data_stopped(pipeline):
    pipeline.stopped = True

    with pytest.raises(RuntimeError):
        pipeline.ingest_data()


def test_ingest_data_loader_error(pipeline):
    pipeline.loader.parse_docs = lambda: (_ for _ in ()).throw(ValueError())

    with pytest.raises(ValueError):
        pipeline.ingest_data()


# ---------------------------------------------------------------------
# build_prompts
# ---------------------------------------------------------------------


def test_build_prompts(pipeline):
    docs = [Document(page_content="chunk")]

    prompts = pipeline.build_prompts([("question", docs)])

    assert len(prompts) == 1
    assert "question" in prompts[0]
    assert "chunk" in prompts[0]


# ---------------------------------------------------------------------
# generate_contextualized_output
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_contextualized_output(pipeline):
    doc = Document(page_content="chunk")

    pipeline.retriver.retrive_documents = lambda q: [doc]

    pipeline.reranker.rerank_documents = lambda pairs, n: [doc]

    pipeline.build_prompts = lambda pairs: ["prompt"]

    async def fake_generate(prompt, rid):
        return "answer"

    pipeline.generator.generate = fake_generate

    output = await pipeline.generate_contextualized_output(["query"])

    assert output == ["answer"]


@pytest.mark.asyncio
async def test_generate_contextualized_output_stopped(pipeline):
    pipeline.stopped = True

    with pytest.raises(RuntimeError):
        await pipeline.generate_contextualized_output(["q"])


@pytest.mark.asyncio
async def test_generate_contextualized_output_retriever_failure(
    pipeline,
):
    pipeline.retriver.retrive_documents = lambda q: (_ for _ in ()).throw(
        RuntimeError()
    )

    with pytest.raises(RuntimeError):
        await pipeline.generate_contextualized_output(["q"])


@pytest.mark.asyncio
async def test_generate_contextualized_output_generator_failure(
    pipeline,
):
    doc = Document(page_content="chunk")

    pipeline.retriver.retrive_documents = lambda q: [doc]

    pipeline.reranker.rerank_documents = lambda p, c: [doc]

    pipeline.build_prompts = lambda p: ["prompt"]

    async def fail(*args):
        raise RuntimeError()

    pipeline.generator.generate = fail

    with pytest.raises(RuntimeError):
        await pipeline.generate_contextualized_output(["q"])


# ---------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------


def test_cleanup(pipeline):
    pipeline.cleanup()

    assert pipeline.stopped

import json

import pytest
from langchain_core.documents import Document

from src.agent.agent import MaxIterationsExceeded
from src.pipeline.agent_pipeline import AgentPipeline


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
        "rag": {
            "max_thought_process_words": 100,
            "max_answer_words": 100,
        },
        "agent": {
            "max_iterations": 10,
            "max_thought_len": 100,
            "max_action_input_len": 100,
        },
    }


def dummy_make_agent_step_schema(
    tool_names=None, max_thought_len=100, max_action_input_len=100
):
    return {}


def make_open_and_yaml(monkeypatch, config):
    class DummyFile:
        def __enter__(self):
            return object()

        def __exit__(self, *args):
            pass

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: DummyFile())
    monkeypatch.setattr("src.pipeline.rag_pipeline.yaml.safe_load", lambda _: config)


class _FakeDoc:
    def __init__(self, content: str, source: str):
        self.page_content = content
        self.metadata = {"source": source}


class _FakeRetriever:
    def retrive_documents(self, query):
        return [_FakeDoc("relevant chunk", "doc.py")]


class _FakeReranker:
    def rerank_documents(self, pairs, n):
        return [doc for _, doc in pairs]


class _ScriptedGenerator:
    def __init__(self, responses):
        self.responses = list(responses)
        self._index = 0

    async def generate(self, prompt, request_id, max_tokens=None):
        response = self.responses[self._index]
        self._index += 1
        return response


def _step(thought: str, action: str, action_input: str) -> str:
    return json.dumps(
        {"thought": thought, "action": action, "action_input": action_input}
    )


@pytest.fixture
def pipeline(monkeypatch, config):
    monkeypatch.setattr(
        "src.pipeline.rag_pipeline.yaml.safe_load",
        lambda f: config,
    )

    max_iterations = 10

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
        def __init__(self, cfg, save_documents=False):
            self.stopped = False
            self.documents_by_source = {}

    class DummyEmbedder:
        def __init__(self, cfg):
            self.embedding_dim = 4
            self.model = object()
            self.stopped = False

    class DummyIndexer:
        def __init__(self, *args):
            self.database = object()
            self.stopped = False

        def index_into_vectorDB(self, *args):
            self.called = args

    class DummyChunker:
        def __init__(self, cfg):
            self.stopped = False

    class DummyRetriever:
        def __init__(self, *args):
            self.stopped = False

    class DummyReranker:
        def __init__(self, cfg):
            self.stopped = False

    class DummyCalculatorTool:
        name = "calculator"

    class DummyRetrievalTool:
        name = "retriever"

        def __init__(self, retriever, reranker, final_chunk_count):
            pass

    class DummyFileLookupTool:
        name = "lookup"

        def __init__(self, documents_by_source):
            pass

    class DummyGenerator:
        def __init__(self, cfg, schema):
            self.stopped = False
            self.schema = schema
            self.scriptgen = _ScriptedGenerator(
                responses=[
                    {
                        "thought": "I need to retrieve documents",
                        "action": "retrieve_documents",
                        "action_input": "how does X work?",
                    },
                    {
                        "thought": "I have enough information to generate an answer",
                        "action": "final_answer",
                        "action_input": "X works via Y.",
                    },
                ]
            )
            self.scratchpad = []

        async def generate(self, prompt, request_id, max_tokens=None):
            for i in range(max_iterations):
                resp = await self.scriptgen.generate(prompt, request_id, max_tokens)
                if resp["action"] == "final_answer":
                    return {
                        "answer": resp["action_input"],
                        "iterations_used": self.scriptgen._index,
                        "scratchpad": self.scratchpad,
                    }
                self.scratchpad.append(
                    {
                        "thought": resp["thought"],
                        "action": resp["action"],
                        "action_input": resp["action_input"],
                        "observation": "Y makes X work.",
                    }
                )

    class DummyAgent:
        def __init__(self, *args, **kwargs):
            self.max_iterations = kwargs.get("max_iterations", 6)
            self.step_generator = kwargs.get("step_generator", _step)
            self.stopped = False

        async def run(self, query):
            return await self.step_generator.generate(
                query, request_id="", max_tokens=None
            )

    monkeypatch.setattr("src.pipeline.agent_pipeline.Loader", DummyLoader)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Embedder", DummyEmbedder)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Indexer", DummyIndexer)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Chunker", DummyChunker)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Retriever", DummyRetriever)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Reranker", DummyReranker)
    monkeypatch.setattr(
        "src.pipeline.agent_pipeline.CalculatorTool", DummyCalculatorTool
    )
    monkeypatch.setattr("src.pipeline.agent_pipeline.RetrievalTool", DummyRetrievalTool)
    monkeypatch.setattr(
        "src.pipeline.agent_pipeline.FileLookupTool", DummyFileLookupTool
    )
    monkeypatch.setattr(
        "src.pipeline.agent_pipeline.make_agent_step_schema",
        dummy_make_agent_step_schema,
    )
    monkeypatch.setattr("src.pipeline.agent_pipeline.VLLMGenerator", DummyGenerator)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Agent", DummyAgent)

    return AgentPipeline()


@pytest.mark.asyncio
async def test_agent_pipeline_runs_successfully(pipeline):
    result = await pipeline.run("how does X work?")

    assert result["answer"] == "X works via Y."
    assert result["iterations_used"] == 2
    assert result["scratchpad"][0]["thought"] == "I need to retrieve documents"
    assert result["scratchpad"][0]["action"] == "retrieve_documents"
    assert result["scratchpad"][0]["action_input"] == "how does X work?"
    assert result["scratchpad"][0]["observation"] == "Y makes X work."


def test_init_bad_config(monkeypatch):
    monkeypatch.setattr(
        "builtins.open",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )

    with pytest.raises(FileNotFoundError):
        AgentPipeline()


def test_init_none_embedding_dimension(monkeypatch, config):
    make_open_and_yaml(monkeypatch, config)

    class DummyLoader:
        def __init__(self, cfg, save_documents=False):
            self.documents_by_source = {}

    class DummyEmbedder:
        def __init__(self, cfg):
            self.embedding_dim = None
            self.model = object()

    monkeypatch.setattr("src.pipeline.agent_pipeline.Loader", DummyLoader)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Embedder", DummyEmbedder)

    with pytest.raises(TypeError, match="Embedding"):
        AgentPipeline()


def test_init_loader_failure(monkeypatch, config):
    make_open_and_yaml(monkeypatch, config)

    monkeypatch.setattr(
        "src.pipeline.agent_pipeline.Loader",
        lambda cfg, save_documents: (_ for _ in ()).throw(RuntimeError("loader")),
    )

    with pytest.raises(RuntimeError, match="loader"):
        AgentPipeline()


def test_init_embedder_failure(monkeypatch, config):
    make_open_and_yaml(monkeypatch, config)

    class DummyLoader:
        def __init__(self, cfg, save_documents=False):
            self.documents_by_source = {}

    monkeypatch.setattr("src.pipeline.agent_pipeline.Loader", DummyLoader)

    monkeypatch.setattr(
        "src.pipeline.agent_pipeline.Embedder",
        lambda cfg: (_ for _ in ()).throw(RuntimeError("embedder")),
    )

    with pytest.raises(RuntimeError, match="embedder"):
        AgentPipeline()


def test_init_indexer_failure(monkeypatch, config):
    make_open_and_yaml(monkeypatch, config)

    class DummyLoader:
        def __init__(self, cfg, save_documents=False):
            self.documents_by_source = {}

    class DummyEmbedder:
        def __init__(self, cfg):
            self.embedding_dim = 384
            self.model = object()

    monkeypatch.setattr("src.pipeline.agent_pipeline.Loader", DummyLoader)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Embedder", DummyEmbedder)

    monkeypatch.setattr(
        "src.pipeline.agent_pipeline.Indexer",
        lambda *args: (_ for _ in ()).throw(RuntimeError("indexer")),
    )

    with pytest.raises(RuntimeError, match="indexer"):
        AgentPipeline()


def test_init_chunker_failure(monkeypatch, config):
    make_open_and_yaml(monkeypatch, config)

    class DummyLoader:
        def __init__(self, cfg, save_documents=False):
            self.documents_by_source = {}

    class DummyEmbedder:
        def __init__(self, cfg):
            self.embedding_dim = 384
            self.model = object()

    class DummyIndexer:
        def __init__(self, *args):
            self.database = object()

    monkeypatch.setattr("src.pipeline.agent_pipeline.Loader", DummyLoader)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Embedder", DummyEmbedder)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Indexer", DummyIndexer)

    monkeypatch.setattr(
        "src.pipeline.agent_pipeline.Chunker",
        lambda cfg: (_ for _ in ()).throw(RuntimeError("chunker")),
    )

    with pytest.raises(RuntimeError, match="chunker"):
        AgentPipeline()


def test_init_retriever_failure(monkeypatch, config):
    make_open_and_yaml(monkeypatch, config)

    class DummyLoader:
        def __init__(self, cfg, save_documents=False):
            self.documents_by_source = {}

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

    monkeypatch.setattr("src.pipeline.agent_pipeline.Loader", DummyLoader)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Embedder", DummyEmbedder)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Indexer", DummyIndexer)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Chunker", DummyChunker)

    monkeypatch.setattr(
        "src.pipeline.agent_pipeline.Retriever",
        lambda *args: (_ for _ in ()).throw(RuntimeError("retriever")),
    )

    with pytest.raises(RuntimeError, match="retriever"):
        AgentPipeline()


def test_init_reranker_failure(monkeypatch, config):
    make_open_and_yaml(monkeypatch, config)

    class DummyLoader:
        def __init__(self, cfg, save_documents=False):
            self.documents_by_source = {}

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

    monkeypatch.setattr("src.pipeline.agent_pipeline.Loader", DummyLoader)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Embedder", DummyEmbedder)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Indexer", DummyIndexer)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Chunker", DummyChunker)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Retriever", DummyRetriever)

    monkeypatch.setattr(
        "src.pipeline.agent_pipeline.Reranker",
        lambda cfg: (_ for _ in ()).throw(RuntimeError("reranker")),
    )

    with pytest.raises(RuntimeError, match="reranker"):
        AgentPipeline()


def test_init_tool_failure(monkeypatch, config):
    make_open_and_yaml(monkeypatch, config)

    class DummyLoader:
        def __init__(self, cfg, save_documents=False):
            self.documents_by_source = {}

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

    monkeypatch.setattr("src.pipeline.agent_pipeline.Loader", DummyLoader)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Embedder", DummyEmbedder)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Indexer", DummyIndexer)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Chunker", DummyChunker)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Retriever", DummyRetriever)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Reranker", DummyReranker)
    monkeypatch.setattr(
        "src.pipeline.agent_pipeline.CalculatorTool",
        lambda: (_ for _ in ()).throw(RuntimeError("calctool")),
    )
    with pytest.raises(RuntimeError, match="calctool"):
        AgentPipeline()


def test_init_generator_failure(monkeypatch, config):
    make_open_and_yaml(monkeypatch, config)

    class DummyLoader:
        def __init__(self, cfg, save_documents=False):
            self.documents_by_source = {}

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

    class DummyCalculatorTool:
        name = "calculator"

    class DummyRetrievalTool:
        name = "retriever"

        def __init__(self, retriever, reranker, final_chunk_count):
            pass

    class DummyFileLookupTool:
        name = "lookup"

        def __init__(self, documents_by_source):
            pass

    monkeypatch.setattr("src.pipeline.agent_pipeline.Loader", DummyLoader)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Embedder", DummyEmbedder)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Indexer", DummyIndexer)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Chunker", DummyChunker)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Retriever", DummyRetriever)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Reranker", DummyReranker)
    monkeypatch.setattr(
        "src.pipeline.agent_pipeline.CalculatorTool", DummyCalculatorTool
    )
    monkeypatch.setattr("src.pipeline.agent_pipeline.RetrievalTool", DummyRetrievalTool)
    monkeypatch.setattr(
        "src.pipeline.agent_pipeline.FileLookupTool", DummyFileLookupTool
    )
    monkeypatch.setattr(
        "src.pipeline.agent_pipeline.make_agent_step_schema",
        dummy_make_agent_step_schema,
    )

    monkeypatch.setattr(
        "src.pipeline.agent_pipeline.VLLMGenerator",
        lambda cfg, sch: (_ for _ in ()).throw(RuntimeError("generator")),
    )

    with pytest.raises(RuntimeError, match="generator"):
        AgentPipeline()


def test_init_agent_failure(monkeypatch, config):
    make_open_and_yaml(monkeypatch, config)

    class DummyLoader:
        def __init__(self, cfg, save_documents=False):
            self.documents_by_source = {}

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

    class DummyCalculatorTool:
        name = "calculator"

    class DummyRetrievalTool:
        name = "retriever"

        def __init__(self, retriever, reranker, final_chunk_count):
            pass

    class DummyFileLookupTool:
        name = "lookup"

        def __init__(self, documents_by_source):
            pass

    class DummyGenerator:
        def __init__(self, cfg, schema):
            pass

        def generate(self, query, max_tokens=None):
            return query

    monkeypatch.setattr("src.pipeline.agent_pipeline.Loader", DummyLoader)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Embedder", DummyEmbedder)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Indexer", DummyIndexer)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Chunker", DummyChunker)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Retriever", DummyRetriever)
    monkeypatch.setattr("src.pipeline.agent_pipeline.Reranker", DummyReranker)
    monkeypatch.setattr(
        "src.pipeline.agent_pipeline.CalculatorTool", DummyCalculatorTool
    )
    monkeypatch.setattr("src.pipeline.agent_pipeline.RetrievalTool", DummyRetrievalTool)
    monkeypatch.setattr(
        "src.pipeline.agent_pipeline.FileLookupTool", DummyFileLookupTool
    )
    monkeypatch.setattr(
        "src.pipeline.agent_pipeline.make_agent_step_schema",
        dummy_make_agent_step_schema,
    )
    monkeypatch.setattr("src.pipeline.agent_pipeline.VLLMGenerator", DummyGenerator)
    monkeypatch.setattr(
        "src.pipeline.agent_pipeline.Agent",
        lambda tools, step_generator, max_iterations: (_ for _ in ()).throw(
            RuntimeError("agent")
        ),
    )

    with pytest.raises(RuntimeError, match="agent"):
        AgentPipeline()


@pytest.mark.asyncio
async def test_run_returns_with_maxiterexception(monkeypatch):
    agentPipeline = object.__new__(AgentPipeline)

    class DummyAgent:
        def run(self, query):
            raise MaxIterationsExceeded("Max Iterations Exceeded", scratchpad=[{}])  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore

    agentPipeline.agent = DummyAgent()  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore

    output = await agentPipeline.run("Test")
    assert (
        output["answer"]
        == "I was unable to find a satisfactory answer within the allowed number of iterations."
    )
    assert output["scratchpad"] == [{}]
    assert output["iterations_used"] == 1


@pytest.mark.asyncio
async def test_run_raises_exception(monkeypatch):
    agentPipeline = object.__new__(AgentPipeline)

    class DummyAgent:
        def run(self, query):
            raise RuntimeError("runfail")

    agentPipeline.agent = DummyAgent()  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore

    with pytest.raises(RuntimeError, match="runfail"):
        await agentPipeline.run("Test")


# ---------------------------------------------------------------------
# ingest_data
# ---------------------------------------------------------------------


def test_context_ingested_initially_false(pipeline):
    assert pipeline.context_ingested is False


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
    assert pipeline.context_ingested is True


def test_ingest_data_stopped(pipeline):
    pipeline.stopped = True

    with pytest.raises(RuntimeError):
        pipeline.ingest_data()


def test_ingest_data_loader_error(pipeline):
    pipeline.loader.parse_docs = lambda: (_ for _ in ()).throw(ValueError())

    with pytest.raises(ValueError):
        pipeline.ingest_data()


# ---------------------------------------------------------------------
# shutdown
# ---------------------------------------------------------------------


def test_shutdown_success(pipeline):
    called = []

    def make_cleanup(name):
        def cleanup():
            called.append(name)

        return cleanup

    pipeline.loader.cleanup = make_cleanup("loader")
    pipeline.chunker.cleanup = make_cleanup("chunker")
    pipeline.embedder.cleanup = make_cleanup("embedder")
    pipeline.indexer.cleanup = make_cleanup("indexer")
    pipeline.retriver.cleanup = make_cleanup("retriever")
    pipeline.reranker.cleanup = make_cleanup("reranker")
    pipeline.generator.cleanup = make_cleanup("generator")
    pipeline.agent.cleanup = make_cleanup("agent")

    pipeline.shutdown()

    assert pipeline.stopped is True

    assert called == [
        "loader",
        "chunker",
        "embedder",
        "indexer",
        "retriever",
        "reranker",
        "generator",
        "agent",
    ]


def test_shutdown_cleanup_failure(pipeline):
    pipeline.loader.cleanup = lambda: (_ for _ in ()).throw(RuntimeError("boom"))

    with pytest.raises(RuntimeError, match="boom"):
        pipeline.shutdown()

    assert pipeline.stopped is True


# ---------------------------------------------------------------------
# is_stopped
# ---------------------------------------------------------------------


def test_is_stopped_false(pipeline):
    pipeline.stopped = False

    pipeline.loader.stopped = False
    pipeline.chunker.stopped = False
    pipeline.embedder.stopped = False
    pipeline.indexer.stopped = False
    pipeline.retriver.stopped = False
    pipeline.reranker.stopped = False
    pipeline.generator.stopped = False

    assert pipeline.is_stopped() is False


@pytest.mark.parametrize(
    "attr",
    [
        "loader",
        "chunker",
        "embedder",
        "indexer",
        "retriver",
        "reranker",
        "generator",
        "agent",
    ],
)
def test_is_stopped_when_component_stopped(pipeline, attr):
    getattr(pipeline, attr).stopped = True
    assert pipeline.is_stopped() is True

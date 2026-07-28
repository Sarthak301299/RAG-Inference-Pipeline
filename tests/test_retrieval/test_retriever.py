import logging

import pytest
from langchain_core.documents import Document

from src.retrieval.retriever import Retriever


class FakeRetriever:
    def __init__(self):
        self.docs = []
        self.exception = None
        self.last_input = None

    def invoke(self, input):
        self.last_input = input
        if self.exception is not None:
            raise self.exception
        return self.docs


class FakeVectorStore:
    def __init__(self):
        self.retriever = FakeRetriever()
        self.exception = None
        self.search_type = None
        self.search_kwargs = None

    def as_retriever(self, search_type, search_kwargs):
        self.search_type = search_type
        self.search_kwargs = search_kwargs

        if self.exception is not None:
            raise self.exception

        return self.retriever


@pytest.fixture
def valid_config():
    return {
        "vdbname": "qdrant",
        "search_type": "similarity",
    }


@pytest.fixture
def fake_vectorstore():
    return FakeVectorStore()


def test_init(valid_config, fake_vectorstore):
    retriever = Retriever(
        config=valid_config,
        retrive_chunk_count=5,
        vecstore=fake_vectorstore,
    )

    assert retriever.stopped is False
    assert retriever.chunk_count == 5
    assert retriever.vdbname == "qdrant"
    assert retriever.search_type == "similarity"

    assert fake_vectorstore.search_type == "similarity"
    assert fake_vectorstore.search_kwargs == {"k": 5}
    assert retriever.vecstore is fake_vectorstore.retriever


@pytest.mark.parametrize(
    "missing_key",
    [
        "vdbname",
        "search_type",
    ],
)
def test_init_missing_config_key(valid_config, fake_vectorstore, missing_key):
    del valid_config[missing_key]

    with pytest.raises(KeyError):
        Retriever(valid_config, 5, fake_vectorstore)


def test_init_invalid_vdbname(valid_config, fake_vectorstore):
    valid_config["vdbname"] = "pinecone"

    with pytest.raises(
        ValueError,
        match="Vector Database must be qdrant, chroma, or pgvector.",
    ):
        Retriever(valid_config, 5, fake_vectorstore)


def test_init_as_retriever_failure(valid_config, fake_vectorstore):
    fake_vectorstore.exception = RuntimeError("failed")

    with pytest.raises(RuntimeError, match="failed"):
        Retriever(valid_config, 5, fake_vectorstore)


def test_retrieve_documents(valid_config, fake_vectorstore):
    docs = [
        Document(page_content="Doc 1"),
        Document(page_content="Doc 2"),
    ]

    fake_vectorstore.retriever.docs = docs

    retriever = Retriever(valid_config, 2, fake_vectorstore)

    result = retriever.retrive_documents("hello")

    assert result == docs
    assert fake_vectorstore.retriever.last_input == "hello"


def test_retrieve_documents_when_stopped(valid_config, fake_vectorstore):
    retriever = Retriever(valid_config, 2, fake_vectorstore)

    retriever.cleanup()

    with pytest.raises(RuntimeError, match="Retriever is stopped."):
        retriever.retrive_documents("query")


def test_retrieve_documents_propagates_exception(
    valid_config,
    fake_vectorstore,
    caplog,
):
    fake_vectorstore.retriever.exception = RuntimeError("invoke failed")

    retriever = Retriever(valid_config, 2, fake_vectorstore)

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(RuntimeError, match="invoke failed"),
    ):
        retriever.retrive_documents("query")

    assert "invoking retriever" in caplog.text


def test_cleanup(valid_config, fake_vectorstore):
    retriever = Retriever(valid_config, 2, fake_vectorstore)

    assert retriever.stopped is False

    retriever.cleanup()

    assert retriever.stopped is True

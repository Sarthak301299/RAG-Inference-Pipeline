import gc
import logging

import pytest
import torch
from langchain_core.documents import Document

from src.retrieval import reranker
from src.retrieval.reranker import Reranker


class FakeCrossEncoder:
    def __init__(self, model_name_or_path):
        self.model_name = model_name_or_path
        self.predictions = None
        self.exception = None
        self.last_inputs = None
        self.last_batch_size = None

    def predict(self, inputs, batch_size):
        self.last_inputs = inputs
        self.last_batch_size = batch_size

        if self.exception is not None:
            raise self.exception

        return self.predictions


@pytest.fixture
def valid_config():
    return {
        "model": "cross-encoder-model",
        "batch_size": "4",
    }


@pytest.fixture
def fake_cross_encoder(monkeypatch):
    encoder = FakeCrossEncoder("unused")

    monkeypatch.setattr(
        reranker,
        "CrossEncoder",
        lambda model_name_or_path: encoder,
    )

    return encoder


def test_init(valid_config, fake_cross_encoder):
    rr = Reranker(valid_config)

    assert rr.stopped is False
    assert rr.model_name == "cross-encoder-model"
    assert rr.batch_size == 4
    assert rr.model is fake_cross_encoder


@pytest.mark.parametrize(
    "missing_key",
    [
        "model",
        "batch_size",
    ],
)
def test_init_missing_config(valid_config, missing_key):
    del valid_config[missing_key]

    with pytest.raises(KeyError):
        Reranker(valid_config)


def test_init_invalid_batch_size(valid_config):
    valid_config["batch_size"] = "abc"

    with pytest.raises(ValueError):
        Reranker(valid_config)


def test_init_cross_encoder_failure(monkeypatch, valid_config):
    def raise_error(*args, **kwargs):
        raise RuntimeError("failed")

    monkeypatch.setattr(
        reranker,
        "CrossEncoder",
        raise_error,
    )

    with pytest.raises(RuntimeError, match="failed"):
        Reranker(valid_config)


def test_rerank_documents_1d_scores(valid_config, fake_cross_encoder):
    rr = Reranker(valid_config)

    docs = [
        Document(page_content="doc1"),
        Document(page_content="doc2"),
        Document(page_content="doc3"),
    ]

    inputs = [
        ("query", docs[0]),
        ("query", docs[1]),
        ("query", docs[2]),
    ]

    fake_cross_encoder.predictions = torch.tensor([0.2, 0.9, 0.5])

    result = rr.rerank_documents(inputs, outcount=2)

    assert result == [
        docs[1],
        docs[2],
    ]

    assert fake_cross_encoder.last_inputs == inputs
    assert fake_cross_encoder.last_batch_size == 4


def test_rerank_documents_2d_scores(valid_config, fake_cross_encoder):
    rr = Reranker(valid_config)

    docs = [
        Document(page_content="a"),
        Document(page_content="b"),
        Document(page_content="c"),
    ]

    inputs = [
        ("q", docs[0]),
        ("q", docs[1]),
        ("q", docs[2]),
    ]

    fake_cross_encoder.predictions = torch.tensor(
        [
            [0.9, 0.1],
            [0.1, 0.8],
            [0.4, 0.5],
        ]
    )

    result = rr.rerank_documents(inputs, outcount=2)

    assert result == [
        docs[1],
        docs[2],
    ]


def test_rerank_outcount_exceeds_documents(valid_config, fake_cross_encoder):
    rr = Reranker(valid_config)

    docs = [
        Document(page_content="one"),
        Document(page_content="two"),
    ]

    inputs = [
        ("q", docs[0]),
        ("q", docs[1]),
    ]

    fake_cross_encoder.predictions = torch.tensor([0.3, 0.1])

    result = rr.rerank_documents(inputs, outcount=10)

    assert result == docs


def test_rerank_empty_inputs(valid_config, fake_cross_encoder):
    rr = Reranker(valid_config)

    fake_cross_encoder.predictions = torch.tensor([])

    result = rr.rerank_documents([], outcount=5)

    assert result == []


def test_rerank_when_stopped(valid_config, fake_cross_encoder):
    rr = Reranker(valid_config)

    rr.cleanup()

    with pytest.raises(RuntimeError, match="Reranker is stopped."):
        rr.rerank_documents([], 1)


def test_rerank_predict_failure(
    valid_config,
    fake_cross_encoder,
    caplog,
):
    rr = Reranker(valid_config)

    fake_cross_encoder.exception = RuntimeError("prediction failed")

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(RuntimeError, match="prediction failed"),
    ):
        rr.rerank_documents([], 1)

    assert "invoking reranker" in caplog.text


def test_cleanup_without_cuda(
    monkeypatch,
    valid_config,
    fake_cross_encoder,
):
    rr = Reranker(valid_config)

    collected = []

    monkeypatch.setattr(
        reranker.torch.cuda,
        "is_available",
        lambda: False,
    )

    monkeypatch.setattr(
        gc,
        "collect",
        lambda: collected.append(True),
    )

    rr.cleanup()

    assert rr.stopped is True
    assert collected == [True]


def test_cleanup_with_cuda(
    monkeypatch,
    valid_config,
    fake_cross_encoder,
):
    rr = Reranker(valid_config)

    calls = []

    monkeypatch.setattr(
        reranker.torch.cuda,
        "is_available",
        lambda: True,
    )

    monkeypatch.setattr(
        reranker.torch.cuda,
        "empty_cache",
        lambda: calls.append("empty_cache"),
    )

    monkeypatch.setattr(
        reranker.torch.cuda,
        "synchronize",
        lambda: calls.append("synchronize"),
    )

    monkeypatch.setattr(
        gc,
        "collect",
        lambda: calls.append("collect"),
    )

    rr.cleanup()

    assert rr.stopped is True
    assert calls == [
        "empty_cache",
        "synchronize",
        "collect",
    ]


def test_cleanup_is_safe_without_model(
    monkeypatch,
    valid_config,
    fake_cross_encoder,
):
    rr = Reranker(valid_config)

    collected = []

    monkeypatch.setattr(
        reranker.torch.cuda,
        "is_available",
        lambda: False,
    )

    monkeypatch.setattr(
        gc,
        "collect",
        lambda: collected.append(True),
    )

    del rr.model
    rr.cleanup()

    assert rr.stopped is True

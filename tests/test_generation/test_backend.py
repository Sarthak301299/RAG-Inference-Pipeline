import gc

import pytest

from src.generation import backend


class DummyGenerator(backend.Generator):
    def __init__(self, config):
        self.cleanup_called = False
        super().__init__(config)

    async def generate(self, prompt, request_id, max_tokens=None):
        return "response"

    def _backend_cleanup(self):
        self.cleanup_called = True


@pytest.fixture
def valid_config():
    return {
        "model": "test-model",
        "backend": "vllm",
    }


def test_init_reads_configuration(valid_config):
    gen = DummyGenerator(valid_config)

    assert gen.model_name == "test-model"
    assert gen.backend == "vllm"
    assert gen.stopped is True


@pytest.mark.parametrize(
    "missing_key",
    [
        "model",
        "backend",
    ],
)
def test_missing_config_key_raises(valid_config, missing_key):
    del valid_config[missing_key]

    with pytest.raises(KeyError):
        DummyGenerator(valid_config)


def test_cleanup_marks_generator_stopped(monkeypatch, valid_config):
    gen = DummyGenerator(valid_config)
    gen.stopped = False

    collected = []

    monkeypatch.setattr(
        gc,
        "collect",
        lambda: collected.append(True),
    )

    gen.cleanup()

    assert gen.stopped is True
    assert gen.cleanup_called is True
    assert collected == [True]


def test_cleanup_is_idempotent(monkeypatch, valid_config):
    gen = DummyGenerator(valid_config)

    monkeypatch.setattr(gc, "collect", lambda: None)

    gen.cleanup()
    gen.cleanup()

    assert gen.stopped is True
    assert gen.cleanup_called is True

import logging
from types import SimpleNamespace

import pytest

import src.generation.vllm as vllm_generator
from src.generation import backend


@pytest.fixture
def valid_config():
    return {
        "model": "llama",
        "backend": "vllm",
        "vllm": {
            "temperature": "0.7",
            "max_num_seqs": "8",
            "max_num_batched_tokens": "1024",
            "max_tokens": "256",
            "top_p": "0.95",
        },
    }


class FakeEngine:
    def shutdown(self):
        pass


class FakeSchema:
    @staticmethod
    def model_json_schema():
        return {"type": "object"}


def test_init(monkeypatch, valid_config):
    monkeypatch.setattr(
        vllm_generator,
        "AsyncEngineArgs",
        lambda: SimpleNamespace(),
    )

    monkeypatch.setattr(
        vllm_generator.AsyncLLMEngine,
        "from_engine_args",
        lambda args: FakeEngine(),
    )

    monkeypatch.setattr(
        vllm_generator,
        "StructuredOutputsParams",
        lambda **kwargs: kwargs,
    )

    monkeypatch.setattr(
        vllm_generator,
        "SamplingParams",
        lambda **kwargs: kwargs,
    )

    monkeypatch.setattr(
        vllm_generator,
        "RAGResponseSchema",
        FakeSchema,
    )

    gen = vllm_generator.VLLMGenerator(valid_config)

    assert gen.model_name == "llama"
    assert gen.backend == "vllm"

    assert gen.temperature == 0.7
    assert gen.max_tokens == 256
    assert gen.max_num_seqs == 8
    assert gen.max_num_batched_tokens == 1024
    assert gen.top_p == 0.95

    assert gen.stopped is False


@pytest.mark.parametrize(
    "missing",
    [
        "model",
        "backend",
        "vllm",
    ],
)
def test_missing_top_level_key(valid_config, missing):
    del valid_config[missing]

    with pytest.raises(KeyError):
        vllm_generator.VLLMGenerator(valid_config)


def test_non_dict_vllm_config(valid_config):
    valid_config["vllm"] = "bad"

    with pytest.raises(TypeError):
        vllm_generator.VLLMGenerator(valid_config)


def test_non_str_model_name(valid_config):
    valid_config["model"] = 1235

    with pytest.raises(TypeError):
        vllm_generator.VLLMGenerator(valid_config)


def test_non_str_backend_name(valid_config):
    valid_config["backend"] = 1235

    with pytest.raises(TypeError):
        vllm_generator.VLLMGenerator(valid_config)


@pytest.mark.parametrize(
    "field",
    [
        "temperature",
        "max_num_seqs",
        "max_num_batched_tokens",
        "max_tokens",
        "top_p",
    ],
)
def test_missing_vllm_field(valid_config, field):
    del valid_config["vllm"][field]

    with pytest.raises(KeyError):
        vllm_generator.VLLMGenerator(valid_config)


def test_invalid_numeric_field(valid_config):
    valid_config["vllm"]["temperature"] = "bad"

    with pytest.raises(ValueError):
        vllm_generator.VLLMGenerator(valid_config)


def test_engine_initialisation_failure(monkeypatch, valid_config):
    monkeypatch.setattr(
        vllm_generator,
        "AsyncEngineArgs",
        lambda: SimpleNamespace(),
    )

    monkeypatch.setattr(
        vllm_generator.AsyncLLMEngine,
        "from_engine_args",
        lambda args: (_ for _ in ()).throw(RuntimeError("engine failed")),
    )

    with pytest.raises(RuntimeError):
        vllm_generator.VLLMGenerator(valid_config)


class FakeOutput:
    def __init__(self, text):
        self.text = text


class FakeResult:
    def __init__(self, text):
        self.outputs = [FakeOutput(text)]


@pytest.mark.asyncio
async def test_generate(monkeypatch):
    gen = vllm_generator.VLLMGenerator.__new__(vllm_generator.VLLMGenerator)

    gen.sampling_params = object()  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore
    gen.stopped = False

    async def fake_stream():
        yield FakeResult("first")
        yield FakeResult("second")
        yield FakeResult("final")

    class Engine:
        def generate(self, **kwargs):
            assert kwargs["prompt"] == "hello"
            assert kwargs["request_id"] == "123"
            return fake_stream()

    gen.vllmengine = Engine()  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore

    result = await gen.generate("hello", "123")

    assert result == "final"


@pytest.mark.asyncio
async def test_generate_empty_stream():
    gen = vllm_generator.VLLMGenerator.__new__(vllm_generator.VLLMGenerator)

    gen.sampling_params = object()  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore
    gen.stopped = False

    async def fake_stream():
        if False:
            yield

    class Engine:
        def generate(self, **kwargs):
            return fake_stream()

    gen.vllmengine = Engine()  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore

    result = await gen.generate("hello", "id")

    assert result == ""


@pytest.mark.asyncio
async def test_generate_exception(caplog):
    gen = vllm_generator.VLLMGenerator.__new__(vllm_generator.VLLMGenerator)

    gen.sampling_params = object()  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore
    gen.stopped = False

    class Engine:
        def generate(self, **kwargs):
            raise RuntimeError("generation failed")

    gen.vllmengine = Engine()  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore

    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError):
        await gen.generate("hello", "id")

    assert "during output generation" in caplog.text


@pytest.mark.asyncio
async def test_generate_when_stopped_raises():
    gen = vllm_generator.VLLMGenerator.__new__(vllm_generator.VLLMGenerator)

    gen.stopped = True

    with pytest.raises(RuntimeError, match="Generator is stopped."):
        await gen.generate("hello", "request-id")


def test_backend_cleanup():
    gen = vllm_generator.VLLMGenerator.__new__(vllm_generator.VLLMGenerator)

    called = []

    class Engine:
        def shutdown(self):
            called.append(True)

    gen.vllmengine = Engine()  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore

    gen._backend_cleanup()

    assert called == [True]


def test_cleanup(monkeypatch):
    gen = vllm_generator.VLLMGenerator.__new__(vllm_generator.VLLMGenerator)

    gen.stopped = False

    backend_called = []

    class Engine:
        def shutdown(self):
            backend_called.append(True)

    gen.vllmengine = Engine()  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore

    monkeypatch.setattr(
        backend.gc,
        "collect",
        lambda: backend_called.append("gc"),
    )

    gen.cleanup()

    assert gen.stopped is True
    assert backend_called == [True, "gc"]

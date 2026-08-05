import pytest
from fastapi import HTTPException, status

from src.api.schemas import UserRequest
from src.api.server import (
    agent_lifespan,
    app,
    generate_endpoint,
    get_version_from_toml,
    handle_user_prompt,
    invocations_endpoint,
    live_endpoint,
    liveness_check,
    main,
    ping_endpoint,
    rag_lifespan,
    readiness_check,
    ready_endpoint,
    startup_check,
    startup_endpoint,
)
from src.pipeline.agent_pipeline import AgentPipeline


def test_get_version_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    assert get_version_from_toml() == "Unknown"


def test_get_version_project(tmp_path, monkeypatch):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("""
[project]
version = "1.2.3"
""")

    monkeypatch.chdir(tmp_path)

    assert get_version_from_toml() == "1.2.3"


def test_get_version_poetry(tmp_path, monkeypatch):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("""
[tool.poetry]
version = "4.5.6"
""")

    monkeypatch.chdir(tmp_path)

    assert get_version_from_toml() == "4.5.6"


def test_get_version_unknown_table(tmp_path, monkeypatch):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("""
[foo]
bar = "baz"
""")

    monkeypatch.chdir(tmp_path)

    assert get_version_from_toml() == "Unknown"


@pytest.mark.asyncio
async def test_liveness_pipeline_missing():
    if hasattr(app.state, "rag_pipeline"):
        delattr(app.state, "rag_pipeline")

    response = await liveness_check()

    assert response.status == status.HTTP_500_INTERNAL_SERVER_ERROR


class DummyPipeline:
    def is_stopped(self):
        return True

    context_ingested = True


@pytest.mark.asyncio
async def test_liveness_pipeline_stopped():
    app.state.rag_pipeline = DummyPipeline()

    response = await liveness_check()

    assert response.status == status.HTTP_500_INTERNAL_SERVER_ERROR


class HealthyPipeline:
    context_ingested = True

    def is_stopped(self):
        return False

    async def generate_contextualized_output(self, queries):
        assert queries == ["hello"]
        return ["answer"]

    async def run(self, query):
        assert query == "hello"
        return "answer"


@pytest.mark.asyncio
async def test_liveness_success():
    app.state.rag_pipeline = HealthyPipeline()

    response = await live_endpoint()

    assert response.status == status.HTTP_200_OK
    assert response.generated_response == "alive"


class BrokenPipeline:
    @property
    def context_ingested(self):
        return True

    def is_stopped(self):
        raise RuntimeError("boom")


class BrokenGenPipeline:
    context_ingested = True

    def is_stopped(self):
        return False

    async def generate_contextualized_output(self, queries):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_liveness_exception():
    app.state.rag_pipeline = BrokenPipeline()

    with pytest.raises(HTTPException):
        await liveness_check()


@pytest.mark.asyncio
async def test_readiness_pipeline_missing():
    if hasattr(app.state, "rag_pipeline"):
        delattr(app.state, "rag_pipeline")

    response = await readiness_check()

    assert response.status == status.HTTP_503_SERVICE_UNAVAILABLE


@pytest.mark.asyncio
async def test_readiness_pipeline_stopped():
    app.state.rag_pipeline = DummyPipeline()

    response = await readiness_check()

    assert response.status == status.HTTP_503_SERVICE_UNAVAILABLE


@pytest.mark.asyncio
async def test_readiness_success():
    app.state.rag_pipeline = HealthyPipeline()

    response = await ready_endpoint()

    assert response.status == status.HTTP_200_OK
    assert response.generated_response == "ready"


@pytest.mark.asyncio
async def test_readiness_exception():
    app.state.rag_pipeline = BrokenPipeline()

    with pytest.raises(HTTPException):
        await ping_endpoint()


@pytest.mark.asyncio
async def test_readiness_not_ingested():
    class Pipeline:
        context_ingested = False

        def is_stopped(self):
            return False

    app.state.rag_pipeline = Pipeline()

    response = await readiness_check()

    assert response.status == status.HTTP_503_SERVICE_UNAVAILABLE


@pytest.mark.asyncio
async def test_startup_pipeline_missing():
    if hasattr(app.state, "rag_pipeline"):
        delattr(app.state, "rag_pipeline")

    response = await startup_check()

    assert response.status == status.HTTP_503_SERVICE_UNAVAILABLE


@pytest.mark.asyncio
async def test_startup_pipeline_stopped():
    app.state.rag_pipeline = DummyPipeline()

    response = await startup_check()

    assert response.status == status.HTTP_503_SERVICE_UNAVAILABLE


@pytest.mark.asyncio
async def test_startup_success():
    app.state.rag_pipeline = HealthyPipeline()

    response = await startup_endpoint()

    assert response.status == status.HTTP_200_OK
    assert response.generated_response == "started"


@pytest.mark.asyncio
async def test_startup_exception():
    app.state.rag_pipeline = BrokenPipeline()

    with pytest.raises(HTTPException):
        await startup_check()


@pytest.mark.asyncio
async def test_startup_not_ingested():
    class Pipeline:
        context_ingested = False

        def is_stopped(self):
            return False

    app.state.rag_pipeline = Pipeline()

    response = await startup_check()

    assert response.status == status.HTTP_503_SERVICE_UNAVAILABLE


@pytest.mark.asyncio
async def test_handle_prompt_pipeline_missing():
    if hasattr(app.state, "rag_pipeline"):
        delattr(app.state, "rag_pipeline")

    response = await handle_user_prompt(UserRequest(prompt="hello"))

    assert response.status == status.HTTP_503_SERVICE_UNAVAILABLE


@pytest.mark.asyncio
async def test_handle_prompt_pipeline_stopped():
    app.state.rag_pipeline = DummyPipeline()

    response = await handle_user_prompt(UserRequest(prompt="hello"))

    assert response.status == status.HTTP_503_SERVICE_UNAVAILABLE


@pytest.mark.asyncio
async def test_handle_prompt_success():
    app.state.rag_pipeline = HealthyPipeline()

    response = await generate_endpoint(UserRequest(prompt="hello"))

    assert response.status == status.HTTP_200_OK
    assert response.generated_response == "answer"


@pytest.mark.asyncio
async def test_handle_agent_prompt_success(monkeypatch):
    app.state.agent_pipeline = object.__new__(AgentPipeline)
    app.state.agent_pipeline.context_ingested = True

    def is_stopped(self):
        return False

    async def run(self, query):
        assert query == "hello"
        return {"answer": "answer", "iterations_used": 1, "scratchpad": []}

    monkeypatch.setattr("src.api.server.AgentPipeline.is_stopped", is_stopped)
    monkeypatch.setattr("src.api.server.AgentPipeline.run", run)

    response = await invocations_endpoint(UserRequest(prompt="hello"))

    assert response.status == status.HTTP_200_OK
    assert (
        response.generated_response
        == '{"answer":"answer","iterations_used":1,"scratchpad":[]}'
    )


@pytest.mark.asyncio
async def test_handle_prompt_not_ingested():
    class Pipeline:
        context_ingested = False

        def is_stopped(self):
            return False

    app.state.agent_pipeline = Pipeline()

    response = await handle_user_prompt(UserRequest(prompt="hello"))

    assert response.status == status.HTTP_503_SERVICE_UNAVAILABLE


@pytest.mark.asyncio
async def test_handle_prompt_generation_exception():
    app.state.agent_pipeline = BrokenGenPipeline()

    with pytest.raises(HTTPException):
        await invocations_endpoint(UserRequest(prompt="hello"))


@pytest.mark.asyncio
async def test_handle_prompt_readiness_exception():
    app.state.agent_pipeline = BrokenPipeline()

    with pytest.raises(HTTPException):
        await handle_user_prompt(UserRequest(prompt="hello"))


@pytest.mark.asyncio
async def test_rag_lifespan(monkeypatch):
    class FakePipeline:
        def __init__(self):
            self.ingested = False
            self.shutdown_called = False

        def ingest_data(self):
            self.ingested = True

        def shutdown(self):
            self.shutdown_called = True

    monkeypatch.setattr(
        "src.api.server.RAGPipeLine",
        FakePipeline,
    )

    async with rag_lifespan(app):
        assert app.state.rag_pipeline.ingested

    assert app.state.rag_pipeline.shutdown_called


@pytest.mark.asyncio
async def test_agent_lifespan(monkeypatch):
    class FakePipeline:
        def __init__(self):
            self.ingested = False
            self.shutdown_called = False

        def ingest_data(self):
            self.ingested = True

        def shutdown(self):
            self.shutdown_called = True

    monkeypatch.setattr(
        "src.api.server.AgentPipeline",
        FakePipeline,
    )

    async with agent_lifespan(app):
        assert app.state.agent_pipeline.ingested

    assert app.state.agent_pipeline.shutdown_called


def test_main(monkeypatch):
    called = {}

    def fake_run(app_, host, port, log_level):
        called["host"] = host
        called["port"] = port
        called["log_level"] = log_level

    monkeypatch.setattr("src.api.server.uvicorn.run", fake_run)

    main()

    assert called["host"] == "0.0.0.0"
    assert called["port"] == 8080
    assert called["log_level"] == "INFO"


def test_main_custom_env(monkeypatch):
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", "9000")
    monkeypatch.setenv("API_LOG_LEVEL", "DEBUG")

    args = {}

    def fake_run(app_, host, port, log_level):
        args["host"] = host
        args["port"] = port
        args["log_level"] = log_level

    monkeypatch.setattr("src.api.server.uvicorn.run", fake_run)

    main()

    assert args == {
        "host": "127.0.0.1",
        "port": 9000,
        "log_level": "DEBUG",
    }


def test_main_raises_on_invalid_lifespan(monkeypatch):
    monkeypatch.setenv("API_EXECUTION_MODE", "invalid")

    with pytest.raises(ValueError, match="API_EXECUTION_MODE"):
        main()


def test_main_uvicorn_exception(monkeypatch):
    def fake_run(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.api.server.uvicorn.run", fake_run)

    with pytest.raises(RuntimeError):
        main()

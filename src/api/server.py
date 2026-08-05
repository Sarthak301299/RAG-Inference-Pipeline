import logging
import os
import tomllib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, status

from src.api.schemas import AgentQueryResponse, UserRequest, UserResponse
from src.pipeline.agent_pipeline import AgentPipeline
from src.pipeline.rag_pipeline import RAGPipeLine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_version_from_toml() -> str:
    path = Path("pyproject.toml")
    if not path.exists():
        return "Unknown"

    with open(path, "rb") as f:
        data = tomllib.load(f)

    # Check standard PEP 621 [project] table first
    if "project" in data and "version" in data["project"]:
        return data["project"]["version"]

    # Fallback to Poetry [tool.poetry] table
    if "tool" in data and "poetry" in data.get("tool", {}):
        return data["tool"]["poetry"].get("version", "Unknown")

    return "Unknown"


@asynccontextmanager
async def rag_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Intialize RAG Modules and gracefully shutdown before exit."""
    app.state.rag_pipeline = RAGPipeLine()
    app.state.rag_pipeline.ingest_data()
    yield
    app.state.rag_pipeline.shutdown()


@asynccontextmanager
async def agent_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Intialize Agent Modules and gracefully shutdown before exit."""
    app.state.agent_pipeline = AgentPipeline()
    app.state.agent_pipeline.ingest_data()
    yield
    app.state.agent_pipeline.shutdown()


app = FastAPI(
    name="RAG Inference API",
    description="A FastAPI application that uses RAG to serve LLM inference.",
)
app.version = get_version_from_toml()


@app.get("/health/live")
async def live_endpoint() -> UserResponse:
    return await liveness_check()


@app.get("/health/ready")
async def ready_endpoint() -> UserResponse:
    return await readiness_check()


@app.get("/ping")
async def ping_endpoint() -> UserResponse:
    return await readiness_check()


@app.get("/health/startup")
async def startup_endpoint() -> UserResponse:
    return await startup_check()


@app.post("/generate", response_model=UserResponse)
async def generate_endpoint(request: UserRequest) -> UserResponse:
    return await handle_user_prompt(request)


@app.post("/invocations", response_model=UserResponse)
async def invocations_endpoint(request: UserRequest) -> UserResponse:
    return await handle_user_prompt(request)


async def liveness_check() -> UserResponse:
    try:
        pipeline: RAGPipeLine | AgentPipeline | None = getattr(
            app.state, "agent_pipeline", None
        )
        pipeline = pipeline or getattr(app.state, "rag_pipeline", None)
        if not pipeline:
            return UserResponse(
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                generated_response="RAG Pipeline not initialized.",
            )
        if pipeline.is_stopped():
            return UserResponse(
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                generated_response="RAG Pipeline is currently stopped.",
            )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Got Exception {e} while performing liveness check.",
        )
    return UserResponse(status=status.HTTP_200_OK, generated_response="alive")


async def readiness_check() -> UserResponse:
    try:
        pipeline: RAGPipeLine | AgentPipeline | None = getattr(
            app.state, "agent_pipeline", None
        )
        pipeline = pipeline or getattr(app.state, "rag_pipeline", None)
        if not pipeline:
            return UserResponse(
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
                generated_response="RAG Pipeline not initialized.",
            )
        if pipeline.is_stopped():
            return UserResponse(
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
                generated_response="RAG Pipeline is currently stopped.",
            )
        if not pipeline.context_ingested:
            return UserResponse(
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
                generated_response="RAG Pipeline has not ingested context.",
            )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Got Exception {e} while performing readiness check.",
        )
    return UserResponse(status=status.HTTP_200_OK, generated_response="ready")


async def startup_check() -> UserResponse:
    try:
        pipeline: RAGPipeLine | AgentPipeline | None = getattr(
            app.state, "agent_pipeline", None
        )
        pipeline = pipeline or getattr(app.state, "rag_pipeline", None)
        if not pipeline:
            return UserResponse(
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
                generated_response="RAG Pipeline not initialized.",
            )
        if pipeline.is_stopped():
            return UserResponse(
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
                generated_response="RAG Pipeline is currently stopped.",
            )
        if not pipeline.context_ingested:
            return UserResponse(
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
                generated_response="RAG Pipeline has not ingested context.",
            )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Got Exception {e} while performing startup check.",
        )
    return UserResponse(status=status.HTTP_200_OK, generated_response="started")


async def handle_user_prompt(request: UserRequest) -> UserResponse:
    try:
        pipeline: RAGPipeLine | AgentPipeline | None = getattr(
            app.state, "agent_pipeline", None
        )
        pipeline = pipeline or getattr(app.state, "rag_pipeline", None)
        if not pipeline:
            return UserResponse(
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
                generated_response="RAG Pipeline not initialized.",
            )
        if pipeline.is_stopped():
            return UserResponse(
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
                generated_response="RAG Pipeline is currently stopped.",
            )
        if not pipeline.context_ingested:
            return UserResponse(
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
                generated_response="RAG Pipeline has not ingested context.",
            )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Got Exception {e} while performing readiness check.",
        )
    try:
        if isinstance(pipeline, AgentPipeline):
            agent_result = await pipeline.run(request.prompt)
            prompt_output = AgentQueryResponse.model_validate(
                agent_result
            ).model_dump_json()
        else:
            prompt_output = await pipeline.generate_contextualized_output(
                [request.prompt]
            )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Got Exception {e} while generating response to query.",
        )

    if isinstance(prompt_output, str):
        return UserResponse(status=status.HTTP_200_OK, generated_response=prompt_output)
    else:
        return UserResponse(
            status=status.HTTP_200_OK, generated_response=prompt_output[0]
        )


def main():
    api_host = os.getenv("API_HOST", default="0.0.0.0")
    api_port = int(os.getenv("API_PORT", default="8080"))
    log_level = os.getenv("API_LOG_LEVEL", default="INFO")
    os.environ["CC"] = "gcc-14"
    os.environ["CXX"] = "g++-14"
    os.environ["CUDAHOSTCXX"] = "g++-14"
    mode = os.getenv("API_EXECUTION_MODE", default="agent")
    if mode not in ("agent", "rag"):
        raise ValueError("API_EXECUTION_MODE must be set to 'agent' or 'rag'.")
    app.state.mode = mode
    app.router.lifespan_context = agent_lifespan if mode == "agent" else rag_lifespan
    try:
        uvicorn.run(app, host=api_host, port=api_port, log_level=log_level)
    except Exception as e:
        logger.error(f"Got Exception {e} during API initialization.")
        raise


if __name__ == "__main__":
    main()  # pragma: no cover

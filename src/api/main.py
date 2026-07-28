from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Intialize Application Modules and gracefully shutdown before exit."""
    yield


app = FastAPI(
    name="RAG Inference API",
    description="A FastAPI application that uses RAG to serve LLM inference.",
    lifespan=lifespan,
)


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="INFO")


if __name__ == "__main__":
    main()  # pragma: no cover

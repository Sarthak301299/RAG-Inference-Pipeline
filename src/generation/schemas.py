from pydantic import BaseModel, Field


class RAGResponseSchema(BaseModel):
    thought_process: str = Field(
        description="Internal reasoning path before answering."
    )
    answer: str = Field(description="The factual answer based strictly on context.")
    sources: list[str] = Field(description="List of source document names used.")

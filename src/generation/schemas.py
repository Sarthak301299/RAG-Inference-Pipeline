from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


class BaseResponseSchema(BaseModel):
    thought_process: str = Field(...)
    answer: str = Field(...)
    sources: list[str] = Field(...)


def make_output_schema(
    max_thought_process_len: int, max_answer_len: int
) -> type[BaseModel]:
    class RAGResponseSchema(BaseModel):
        thought_process: Annotated[
            str, StringConstraints(max_length=max_thought_process_len)
        ] = Field(
            description="Internal reasoning path before answering. Keep it brief."
        )
        answer: Annotated[str, StringConstraints(max_length=max_answer_len)] = Field(
            description="The factual answer based strictly on context. Elaborate as much as needed unless the user asks otherwise."
        )
        sources: list[Annotated[str, StringConstraints(max_length=1000)]] = Field(
            description="List of source document names used. Only cite each source once."
        )

    return RAGResponseSchema

import hashlib
import json
from enum import Enum
from typing import Annotated

from langchain_core.documents import Document
from pydantic import BaseModel, Field, StringConstraints, create_model


class RAGInputPromptFormat(BaseModel):
    user_query: str = Field(..., description="The original input from the user.")
    chunks: list[Document] = Field(
        ..., description="List of highly relevant text chunks retrieved."
    )
    system_query: str = (
        "You are a helpful and accurate AI assistant. "
        "Your job is to answer the user's query using ONLY the factual context provided below.\n"
        "Rules:\n"
        "1. Rely entirely on the provided context. Do not use any outside knowledge or assumptions.\n"
        "2. If the answer cannot be found in the context, respond with: 'I cannot find the answer in the provided documents.'\n"
        "3. Do not speculate, extrapolate, or make up facts.\n"
        "4. Cite the specific source or document title when presenting a fact if available.\n"
        "5. DO NOT cite any sources that do not exist. DO NOT cite the same source multiple times.\n"
        "6. If no sources exist, clearly state 'None' in the sources."
        "7. If conflicting information appears across sources, highlight the discrepancy explicitly.\n"
    )

    def get_formatted_prompt(self) -> str:
        prompt = self.system_query
        for idx, chunk in enumerate(self.chunks, start=1):
            content = getattr(chunk, "page_content", "")
            metadata = getattr(chunk, "metadata", {})

            prompt += f"--- START CHUNK {idx}, metadata: {json.dumps(metadata)} ---\n"
            prompt += f"{content}\n"
            prompt += f"--- END CHUNK {idx} ---\n\n"

        prompt += f"User Query:\n{self.user_query}\nResponse: "
        return prompt


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


_FINAL_ANSWER_ACTION = "final_answer"


def make_agent_step_schema(
    tool_names: list[str], max_thought_len: int = 2500, max_action_input_len: int = 5000
) -> type[BaseModel]:
    if not tool_names:
        raise ValueError("At least one tool name must be provided.")
    if _FINAL_ANSWER_ACTION in tool_names:
        raise ValueError(
            f"{_FINAL_ANSWER_ACTION} is reserved and cannot be in tool_names."
        )

    action_values = list(tool_names) + [_FINAL_ANSWER_ACTION]
    schema_suffix = hashlib.sha1("|".join(sorted(action_values)).encode()).hexdigest()[
        :8
    ]
    ActionEnum = Enum(
        f"ActionEnum_{schema_suffix}",
        {f"Action_{i}": value for i, value in enumerate(action_values)},
    )

    try:
        AgentStepSchema = create_model(
            f"AgentStepSchema_{schema_suffix}",
            thought=(
                Annotated[
                    str, StringConstraints(min_length=1, max_length=max_thought_len)
                ],
                Field(
                    description="Brief internal reasoning before deciding on an action."
                ),
            ),
            action=(
                ActionEnum,
                Field(
                    description=f"The action to take. Must be one of {action_values}."
                ),
            ),
            action_input=(
                Annotated[
                    str,
                    StringConstraints(min_length=1, max_length=max_action_input_len),
                ],
                Field(
                    description="Input to the chosen tool, or the final answer text if action is 'final_answer'."
                ),
            ),
        )
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Error creating AgentStepSchema: {e}")

    return AgentStepSchema

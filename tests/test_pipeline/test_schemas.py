import json
from enum import Enum

import pytest
from langchain_core.documents import Document
from pydantic import BaseModel

from src.pipeline import schemas
from src.pipeline.schemas import RAGInputPromptFormat


def dummy_create_model(arg, **kwargs):
    class ActionEnum(str, Enum):
        calculator = "calculator"
        retrieve_documents = "retrieve_documents"
        final_answer = "final_answer"

    class Test(BaseModel):
        thought: str
        action: ActionEnum
        action_input: str

    return Test


def test_prompt_contains_user_query():
    prompt = RAGInputPromptFormat(
        user_query="What is MPI?",
        chunks=[],
    ).get_formatted_prompt()

    assert "User Query:" in prompt
    assert "What is MPI?" in prompt


def test_prompt_contains_chunk():
    chunk = Document(
        page_content="MPI is a message passing standard.",
        metadata={"source": "mpi.pdf"},
    )

    prompt = RAGInputPromptFormat(
        user_query="Explain MPI",
        chunks=[chunk],
    ).get_formatted_prompt()

    assert "START CHUNK 1" in prompt
    assert "END CHUNK 1" in prompt
    assert "MPI is a message passing standard." in prompt
    assert json.dumps({"source": "mpi.pdf"}) in prompt


def test_prompt_multiple_chunks():
    chunks = [
        Document(page_content="first", metadata={}),
        Document(page_content="second", metadata={}),
    ]

    prompt = RAGInputPromptFormat(
        user_query="query",
        chunks=chunks,
    ).get_formatted_prompt()

    assert "START CHUNK 1" in prompt
    assert "START CHUNK 2" in prompt
    assert "first" in prompt
    assert "second" in prompt


def test_prompt_empty_metadata():
    chunk = Document(
        page_content="abc",
        metadata={},
    )

    prompt = RAGInputPromptFormat(
        user_query="query",
        chunks=[chunk],
    ).get_formatted_prompt()

    assert "{}" in prompt


def test_prompt_no_chunks():
    prompt = RAGInputPromptFormat(
        user_query="query",
        chunks=[],
    ).get_formatted_prompt()

    assert "START CHUNK" not in prompt
    assert "query" in prompt


def test_system_prompt_present():
    prompt = RAGInputPromptFormat(
        user_query="query",
        chunks=[],
    ).get_formatted_prompt()

    assert "You are a helpful and accurate AI assistant." in prompt
    assert "Rules:" in prompt


class Dummy:
    pass


def test_make_output_schema(monkeypatch):
    monkeypatch.setattr(schemas, "BaseModel", Dummy)

    out = schemas.make_output_schema(2000, 14000)

    assert hasattr(out, "thought_process")
    assert hasattr(out, "answer")
    assert hasattr(out, "sources")


def test_make_agent_step_schema_accepts_valid_action(monkeypatch):
    monkeypatch.setattr("src.pipeline.schemas.create_model", dummy_create_model)
    schema = schemas.make_agent_step_schema(
        tool_names=["calculator", "retrieve_documents"]
    )

    instance = schema(
        thought="I should calculate this.",
        action="calculator",
        action_input="2 + 2",
    )

    assert instance.action.value == "calculator"  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore


def test_make_agent_step_schema_accepts_final_answer(monkeypatch):
    monkeypatch.setattr("src.pipeline.schemas.create_model", dummy_create_model)
    schema = schemas.make_agent_step_schema(tool_names=["calculator"])

    instance = schema(
        thought="I know the answer.", action="final_answer", action_input="42"
    )

    assert instance.action.value == "final_answer"  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore


def test_make_agent_step_schema_rejects_unknown_action(monkeypatch):
    monkeypatch.setattr("src.pipeline.schemas.create_model", dummy_create_model)
    schema = schemas.make_agent_step_schema(tool_names=["calculator"])

    with pytest.raises(ValueError):
        schema(thought="t", action="not_a_real_tool", action_input="x")


def test_make_agent_step_schema_no_tools_raises(monkeypatch):
    monkeypatch.setattr("src.pipeline.schemas.create_model", dummy_create_model)
    with pytest.raises(ValueError, match="At least one tool"):
        schemas.make_agent_step_schema(tool_names=[])


def test_make_agent_step_schema_reserved_name_raises(monkeypatch):
    monkeypatch.setattr("src.pipeline.schemas.create_model", dummy_create_model)
    with pytest.raises(ValueError, match="reserved"):
        schemas.make_agent_step_schema(tool_names=["final_answer"])


def test_make_agent_step_schema_create_model_failure(monkeypatch):
    def dummy_create_model(arg, **kwargs):
        raise RuntimeError("createfail")

    monkeypatch.setattr("src.pipeline.schemas.create_model", dummy_create_model)
    with pytest.raises(ValueError, match="createfail"):
        schemas.make_agent_step_schema(tool_names=["calculator"])


def test_make_agent_step_schema_json_schema_includes_enum(monkeypatch):
    monkeypatch.setattr("src.pipeline.schemas.create_model", dummy_create_model)
    schema = schemas.make_agent_step_schema(
        tool_names=["calculator", "retrieve_documents"]
    )

    json_schema = schema.model_json_schema()

    # The action field's constraint should be discoverable in the schema
    # so guided decoding can actually restrict generation to valid tools.
    schema_str = str(json_schema)
    assert "calculator" in schema_str
    assert "retrieve_documents" in schema_str
    assert "final_answer" in schema_str

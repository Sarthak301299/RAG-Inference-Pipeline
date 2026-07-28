import json

from langchain_core.documents import Document

from src.pipeline.schemas import RAGInputPromptFormat


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

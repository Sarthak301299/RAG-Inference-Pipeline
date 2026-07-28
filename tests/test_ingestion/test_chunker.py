# tests/test_chunker.py

import logging

import pytest
from langchain_core.documents import Document

from src.ingestion import chunker


@pytest.fixture
def valid_config():
    return {
        "strategy": "recursive",
        "chunk_size": "100",
        "chunk_overlap": "20",
    }


@pytest.fixture
def uninitialized_chunker():
    """
    Creates a Chunker instance without running __init__.

    Useful for testing generate_chunks() and cleanup() independently.
    """
    instance = chunker.Chunker.__new__(chunker.Chunker)
    instance.stopped = False
    instance.strategy = "recursive"
    instance.chunk_size = 100
    instance.chunk_overlap = 20
    return instance


@pytest.fixture
def documents():
    return [
        Document(
            page_content="First document content.",
            metadata={"source": "first.txt"},
        ),
        Document(
            page_content="Second document content.",
            metadata={"source": "second.txt"},
        ),
    ]


class TestChunkerConfiguration:
    def test_reads_configuration_correctly(
        self,
        valid_config,
    ):
        instance = chunker.Chunker(valid_config)

        assert instance.strategy == "recursive"
        assert instance.chunk_size == 100
        assert instance.chunk_overlap == 20
        assert instance.stopped is False

    @pytest.mark.parametrize(
        "strategy",
        [
            "fixed",
            "recursive",
            "auto",
        ],
    )
    def test_accepts_valid_strategies(
        self,
        strategy,
    ):
        instance = chunker.Chunker(
            {
                "strategy": strategy,
                "chunk_size": "100",
                "chunk_overlap": "20",
            }
        )

        assert instance.strategy == strategy
        assert instance.stopped is False

    @pytest.mark.parametrize(
        "strategy",
        [
            "invalid",
            "",
            "FIXED",
            "Recursive",
            "automatic",
        ],
    )
    def test_rejects_invalid_strategy(
        self,
        strategy,
        caplog,
    ):
        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(
                ValueError,
                match="Strategy must be fixed, recursive, or auto.",
            ),
        ):
            chunker.Chunker(
                {
                    "strategy": strategy,
                    "chunk_size": "100",
                    "chunk_overlap": "20",
                }
            )

        assert "reading configuration" in caplog.text

    @pytest.mark.parametrize(
        "missing_key",
        [
            "strategy",
            "chunk_size",
            "chunk_overlap",
        ],
    )
    def test_missing_configuration_key_raises_key_error(
        self,
        missing_key,
        caplog,
    ):
        config = {
            "strategy": "recursive",
            "chunk_size": "100",
            "chunk_overlap": "20",
        }

        del config[missing_key]

        with caplog.at_level(logging.ERROR), pytest.raises(KeyError):
            chunker.Chunker(config)

        assert "reading configuration" in caplog.text

    @pytest.mark.parametrize(
        "chunk_size",
        [
            "invalid",
            "",
            "1.5",
        ],
    )
    def test_invalid_chunk_size_raises_value_error(
        self,
        chunk_size,
        caplog,
    ):
        with caplog.at_level(logging.ERROR), pytest.raises(ValueError):
            chunker.Chunker(
                {
                    "strategy": "recursive",
                    "chunk_size": chunk_size,
                    "chunk_overlap": "20",
                }
            )

        assert "reading configuration" in caplog.text

    def test_chunk_overlap_is_converted_to_int(self):
        instance = chunker.Chunker(
            {
                "strategy": "recursive",
                "chunk_size": "100",
                "chunk_overlap": "20",
            }
        )

        assert instance.chunk_overlap == 20
        assert isinstance(instance.chunk_overlap, int)


class TestGenerateChunksStoppedState:
    def test_generate_chunks_raises_when_stopped(
        self,
        uninitialized_chunker,
        documents,
    ):
        uninitialized_chunker.stopped = True

        with pytest.raises(
            RuntimeError,
            match="Chunker is stopped.",
        ):
            uninitialized_chunker.generate_chunks(
                documents=documents,
                extension=".txt",
            )


class TestRecursiveStrategy:
    def test_uses_recursive_splitter(
        self,
        monkeypatch,
        uninitialized_chunker,
        documents,
    ):
        uninitialized_chunker.strategy = "recursive"

        captured = {}

        class FakeRecursiveSplitter:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def split_documents(self, documents):
                captured["documents"] = documents
                return ["recursive chunks"]

        monkeypatch.setattr(
            chunker,
            "RecursiveCharacterTextSplitter",
            FakeRecursiveSplitter,
        )

        result = uninitialized_chunker.generate_chunks(
            documents=documents,
            extension=".txt",
        )

        assert result == ["recursive chunks"]

        assert captured["chunk_size"] == 100
        assert captured["chunk_overlap"] == 20
        assert captured["documents"] is documents

    def test_recursive_strategy_does_not_use_secondary_splitter(
        self,
        monkeypatch,
        uninitialized_chunker,
        documents,
    ):
        uninitialized_chunker.strategy = "recursive"

        class FakeRecursiveSplitter:
            def __init__(self, **kwargs):
                pass

            def split_documents(self, documents):
                return ["chunks"]

        monkeypatch.setattr(
            chunker,
            "RecursiveCharacterTextSplitter",
            FakeRecursiveSplitter,
        )

        result = uninitialized_chunker.generate_chunks(
            documents=documents,
            extension=".md",
        )

        assert result == ["chunks"]


class TestFixedStrategy:
    def test_uses_character_text_splitter(
        self,
        monkeypatch,
        uninitialized_chunker,
        documents,
    ):
        uninitialized_chunker.strategy = "fixed"

        captured = {}

        class FakeCharacterSplitter:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def split_documents(self, documents):
                captured["documents"] = documents
                return ["fixed chunks"]

        monkeypatch.setattr(
            chunker,
            "CharacterTextSplitter",
            FakeCharacterSplitter,
        )

        result = uninitialized_chunker.generate_chunks(
            documents=documents,
            extension=".txt",
        )

        assert result == ["fixed chunks"]

        assert captured["chunk_size"] == 100
        assert captured["chunk_overlap"] == 20
        assert captured["documents"] is documents


class TestAutoRecursiveExtensions:
    @pytest.mark.parametrize(
        "extension",
        [
            ".pdf",
            ".txt",
        ],
    )
    def test_pdf_and_txt_use_recursive_splitter(
        self,
        monkeypatch,
        uninitialized_chunker,
        documents,
        extension,
    ):
        uninitialized_chunker.strategy = "auto"

        captured = {}

        class FakeRecursiveSplitter:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def split_documents(self, documents):
                captured["documents"] = documents
                return ["recursive chunks"]

        monkeypatch.setattr(
            chunker,
            "RecursiveCharacterTextSplitter",
            FakeRecursiveSplitter,
        )

        result = uninitialized_chunker.generate_chunks(
            documents=documents,
            extension=extension,
        )

        assert result == ["recursive chunks"]
        assert captured["chunk_size"] == 100
        assert captured["chunk_overlap"] == 20
        assert captured["documents"] is documents


class TestAutoLanguageExtensions:
    @pytest.mark.parametrize(
        "extension, expected_language",
        [
            (".c", chunker.Language.C),
            (".cpp", chunker.Language.CPP),
            (".py", chunker.Language.PYTHON),
        ],
    )
    def test_programming_languages_use_language_specific_splitter(
        self,
        monkeypatch,
        uninitialized_chunker,
        documents,
        extension,
        expected_language,
    ):
        uninitialized_chunker.strategy = "auto"

        captured = {}

        class FakeRecursiveSplitter:
            @classmethod
            def from_language(cls, **kwargs):
                captured.update(kwargs)
                return cls()

            def split_documents(self, documents):
                captured["documents"] = documents
                return ["language-specific chunks"]

        monkeypatch.setattr(
            chunker,
            "RecursiveCharacterTextSplitter",
            FakeRecursiveSplitter,
        )

        result = uninitialized_chunker.generate_chunks(
            documents=documents,
            extension=extension,
        )

        assert result == ["language-specific chunks"]

        assert captured["language"] == expected_language
        assert captured["chunk_size"] == 100
        assert captured["chunk_overlap"] == 20
        assert captured["documents"] is documents


class TestAutoMarkdown:
    def test_markdown_creates_header_splitter(
        self,
        monkeypatch,
        uninitialized_chunker,
    ):
        uninitialized_chunker.strategy = "auto"

        documents = [
            Document(
                page_content="# Main Header\nContent",
                metadata={"source": "document.md"},
            )
        ]

        captured = {}

        class FakeHeaderSplitter:
            def __init__(self, **kwargs):
                captured["headers_to_split_on"] = kwargs["headers_to_split_on"]

            def split_text(self, text):
                captured["input_text"] = text

                return [
                    Document(
                        page_content="Header chunk",
                        metadata={"Header_1": "Main Header"},
                    )
                ]

        class FakeMarkdownSplitter:
            def __init__(self, **kwargs):
                captured["chunk_size"] = kwargs["chunk_size"]
                captured["chunk_overlap"] = kwargs["chunk_overlap"]

            def split_documents(self, documents):
                captured["documents"] = documents
                return ["markdown chunks"]

        monkeypatch.setattr(
            chunker,
            "MarkdownHeaderTextSplitter",
            FakeHeaderSplitter,
        )

        monkeypatch.setattr(
            chunker,
            "MarkdownTextSplitter",
            FakeMarkdownSplitter,
        )

        result = uninitialized_chunker.generate_chunks(
            documents=documents,
            extension=".md",
        )

        assert result == ["markdown chunks"]

        assert captured["headers_to_split_on"] == [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ]

        assert captured["input_text"] == "# Main Header\nContent"

        assert captured["chunk_size"] == 100
        assert captured["chunk_overlap"] == 20


class TestMarkdownMetadata:
    def test_original_and_header_metadata_are_merged(
        self,
        monkeypatch,
        uninitialized_chunker,
    ):
        uninitialized_chunker.strategy = "auto"

        documents = [
            Document(
                page_content="# Header\nContent",
                metadata={
                    "source": "document.md",
                    "document_id": "123",
                    "original": "value",
                },
            )
        ]

        header_chunk = Document(
            page_content="Content",
            metadata={
                "Header_1": "Header",
                "original": "overridden",
            },
        )

        captured = {}

        class FakeHeaderSplitter:
            def __init__(self, **kwargs):
                pass

            def split_text(self, text):
                return [header_chunk]

        class FakeMarkdownSplitter:
            def __init__(self, **kwargs):
                pass

            def split_documents(self, documents):
                captured["documents"] = documents
                return ["final chunks"]

        monkeypatch.setattr(
            chunker,
            "MarkdownHeaderTextSplitter",
            FakeHeaderSplitter,
        )

        monkeypatch.setattr(
            chunker,
            "MarkdownTextSplitter",
            FakeMarkdownSplitter,
        )

        result = uninitialized_chunker.generate_chunks(
            documents=documents,
            extension=".md",
        )

        assert result == ["final chunks"]

        assert captured["documents"][0].metadata == {
            "source": "document.md",
            "document_id": "123",
            "original": "overridden",
            "Header_1": "Header",
        }

    def test_original_metadata_is_preserved(
        self,
        monkeypatch,
        uninitialized_chunker,
    ):
        uninitialized_chunker.strategy = "auto"

        original_document = Document(
            page_content="# Header\nContent",
            metadata={
                "source": "document.md",
                "custom": "metadata",
            },
        )

        header_chunk = Document(
            page_content="Content",
            metadata={
                "Header_1": "Header",
            },
        )

        captured = {}

        class FakeHeaderSplitter:
            def __init__(self, **kwargs):
                pass

            def split_text(self, text):
                return [header_chunk]

        class FakeMarkdownSplitter:
            def __init__(self, **kwargs):
                pass

            def split_documents(self, documents):
                captured["documents"] = documents
                return []

        monkeypatch.setattr(
            chunker,
            "MarkdownHeaderTextSplitter",
            FakeHeaderSplitter,
        )

        monkeypatch.setattr(
            chunker,
            "MarkdownTextSplitter",
            FakeMarkdownSplitter,
        )

        uninitialized_chunker.generate_chunks(
            documents=[original_document],
            extension=".md",
        )

        assert captured["documents"][0].metadata["source"] == "document.md"
        assert captured["documents"][0].metadata["custom"] == "metadata"
        assert captured["documents"][0].metadata["Header_1"] == "Header"


class TestAutoUnknownExtensions:
    @pytest.mark.parametrize(
        "extension",
        [
            ".json",
            ".xml",
            ".html",
            ".java",
            "",
            ".unknown",
        ],
    )
    def test_unknown_extensions_use_recursive_splitter(
        self,
        monkeypatch,
        uninitialized_chunker,
        documents,
        extension,
    ):
        uninitialized_chunker.strategy = "auto"

        captured = {}

        class FakeRecursiveSplitter:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def split_documents(self, documents):
                captured["documents"] = documents
                return ["fallback chunks"]

        monkeypatch.setattr(
            chunker,
            "RecursiveCharacterTextSplitter",
            FakeRecursiveSplitter,
        )

        result = uninitialized_chunker.generate_chunks(
            documents=documents,
            extension=extension,
        )

        assert result == ["fallback chunks"]
        assert captured["documents"] is documents


class TestSplitterErrors:
    def test_splitter_initialization_failure_is_logged(
        self,
        monkeypatch,
        uninitialized_chunker,
        documents,
        caplog,
    ):
        class FakeRecursiveSplitter:
            def __init__(self, **kwargs):
                raise RuntimeError("splitter initialization failed")

        monkeypatch.setattr(
            chunker,
            "RecursiveCharacterTextSplitter",
            FakeRecursiveSplitter,
        )

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(
                RuntimeError,
                match="splitter initialization failed",
            ),
        ):
            uninitialized_chunker.generate_chunks(
                documents=documents,
                extension=".txt",
            )

        assert "setting chunking strategy recursive" in caplog.text

    def test_split_documents_failure_is_logged(
        self,
        monkeypatch,
        uninitialized_chunker,
        documents,
        caplog,
    ):
        class FakeRecursiveSplitter:
            def __init__(self, **kwargs):
                pass

            def split_documents(self, documents):
                raise RuntimeError("splitting failed")

        monkeypatch.setattr(
            chunker,
            "RecursiveCharacterTextSplitter",
            FakeRecursiveSplitter,
        )

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(
                RuntimeError,
                match="splitting failed",
            ),
        ):
            uninitialized_chunker.generate_chunks(
                documents=documents,
                extension=".txt",
            )

        assert "splitting documents with extension .txt" in caplog.text


class TestInputIterables:
    def test_accepts_list_of_documents(
        self,
        monkeypatch,
        uninitialized_chunker,
        documents,
    ):
        class FakeRecursiveSplitter:
            def __init__(self, **kwargs):
                pass

            def split_documents(self, documents):
                return list(documents)

        monkeypatch.setattr(
            chunker,
            "RecursiveCharacterTextSplitter",
            FakeRecursiveSplitter,
        )

        result = uninitialized_chunker.generate_chunks(
            documents=documents,
            extension=".txt",
        )

        assert result == documents

    def test_accepts_generator_of_documents(
        self,
        monkeypatch,
        uninitialized_chunker,
        documents,
    ):
        captured = {}

        class FakeRecursiveSplitter:
            def __init__(self, **kwargs):
                pass

            def split_documents(self, documents):
                captured["documents"] = documents
                return list(documents)

        monkeypatch.setattr(
            chunker,
            "RecursiveCharacterTextSplitter",
            FakeRecursiveSplitter,
        )

        document_generator = (document for document in documents)

        result = uninitialized_chunker.generate_chunks(
            documents=document_generator,
            extension=".txt",
        )

        assert result == documents


class TestCleanup:
    def test_cleanup_marks_chunker_as_stopped(
        self,
        uninitialized_chunker,
    ):
        uninitialized_chunker.stopped = False

        uninitialized_chunker.cleanup()

        assert uninitialized_chunker.stopped is True

    def test_cleanup_is_idempotent(
        self,
        uninitialized_chunker,
    ):
        uninitialized_chunker.cleanup()
        uninitialized_chunker.cleanup()

        assert uninitialized_chunker.stopped is True

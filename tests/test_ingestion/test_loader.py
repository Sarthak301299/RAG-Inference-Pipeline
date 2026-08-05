# tests/test_loader.py

import logging
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.ingestion import loader


class TestDynamicDirectoryUnstructuredLoader:
    def test_initializes_with_root_directory_and_glob(
        self,
        monkeypatch,
        tmp_path,
    ):
        captured_kwargs = {}

        def fake_init(self, file_path, **kwargs):
            captured_kwargs["file_path"] = file_path
            captured_kwargs["kwargs"] = kwargs

        monkeypatch.setattr(
            loader.UnstructuredLoader,
            "__init__",
            fake_init,
        )

        instance = loader.DynamicDirectoryUnstructuredLoader(
            root_dir_path=str(tmp_path),
            glob="**/*",
            mode="elements",
        )

        assert instance.root_dir_path == Path(tmp_path)
        assert instance.rglob == "**/*"
        assert instance.loader_kwargs == {
            "mode": "elements",
        }

        assert captured_kwargs["file_path"] == []
        assert captured_kwargs["kwargs"] == {
            "mode": "elements",
        }

    def test_get_current_files_returns_files_recursively(
        self,
        tmp_path,
    ):
        root_file = tmp_path / "root.txt"
        nested_dir = tmp_path / "nested" / "deep"
        nested_dir.mkdir(parents=True)
        nested_file = nested_dir / "nested.pdf"

        root_file.write_text("root")
        nested_file.write_text("nested")

        instance = loader.DynamicDirectoryUnstructuredLoader.__new__(
            loader.DynamicDirectoryUnstructuredLoader
        )
        instance.root_dir_path = tmp_path
        instance.rglob = "**/*"

        files = instance._get_current_files()

        assert set(files) == {
            str(root_file),
            str(nested_file),
        }

    def test_get_current_files_returns_empty_list_for_missing_directory(
        self,
        tmp_path,
    ):
        missing_dir = tmp_path / "does-not-exist"

        instance = loader.DynamicDirectoryUnstructuredLoader.__new__(
            loader.DynamicDirectoryUnstructuredLoader
        )
        instance.root_dir_path = missing_dir
        instance.rglob = "**/*"

        assert instance._get_current_files() == []

    def test_get_current_files_skips_directories(
        self,
        tmp_path,
    ):
        nested_dir = tmp_path / "nested"
        nested_dir.mkdir()

        instance = loader.DynamicDirectoryUnstructuredLoader.__new__(
            loader.DynamicDirectoryUnstructuredLoader
        )
        instance.root_dir_path = tmp_path
        instance.rglob = "**/*"

        files = instance._get_current_files()

        assert files == []

    def test_get_current_files_respects_glob_pattern(
        self,
        tmp_path,
    ):
        txt_file = tmp_path / "document.txt"
        pdf_file = tmp_path / "document.pdf"

        txt_file.write_text("text")
        pdf_file.write_text("pdf")

        instance = loader.DynamicDirectoryUnstructuredLoader.__new__(
            loader.DynamicDirectoryUnstructuredLoader
        )
        instance.root_dir_path = tmp_path
        instance.rglob = "**/*.pdf"

        files = instance._get_current_files()

        assert files == [str(pdf_file)]

    def test_get_current_files_detects_files_created_after_initialization(
        self,
        tmp_path,
    ):
        instance = loader.DynamicDirectoryUnstructuredLoader.__new__(
            loader.DynamicDirectoryUnstructuredLoader
        )
        instance.root_dir_path = tmp_path
        instance.rglob = "**/*"

        assert instance._get_current_files() == []

        new_file = tmp_path / "new.txt"
        new_file.write_text("created later")

        assert instance._get_current_files() == [str(new_file)]

    def test_lazy_load_refreshes_file_path_before_loading(
        self,
        monkeypatch,
        tmp_path,
    ):
        document = Document(
            page_content="content",
            metadata={"source": str(tmp_path / "document.txt")},
        )

        source_file = tmp_path / "document.txt"
        source_file.write_text("content")

        instance = loader.DynamicDirectoryUnstructuredLoader.__new__(
            loader.DynamicDirectoryUnstructuredLoader
        )
        instance.root_dir_path = tmp_path
        instance.rglob = "**/*"
        instance.file_path = []

        captured_file_paths = []

        def fake_lazy_load(self):
            captured_file_paths.append(self.file_path)
            yield document

        monkeypatch.setattr(
            loader.UnstructuredLoader,
            "lazy_load",
            fake_lazy_load,
        )

        result = list(instance.lazy_load())

        assert result == [document]
        assert captured_file_paths == [
            [str(source_file)],
        ]
        assert instance.file_path == [str(source_file)]

    def test_lazy_load_exits_with_empty_when_no_files_exist(
        self,
        tmp_path,
    ):
        instance = loader.DynamicDirectoryUnstructuredLoader.__new__(
            loader.DynamicDirectoryUnstructuredLoader
        )
        instance.root_dir_path = tmp_path
        instance.rglob = "**/*"
        instance.file_path = []

        out = list(instance.lazy_load())

        assert out == []

    def test_lazy_load_exits_with_empty_for_missing_directory(
        self,
        tmp_path,
    ):
        missing_dir = tmp_path / "missing"

        instance = loader.DynamicDirectoryUnstructuredLoader.__new__(
            loader.DynamicDirectoryUnstructuredLoader
        )
        instance.root_dir_path = missing_dir
        instance.rglob = "**/*"
        instance.file_path = []

        out = list(instance.lazy_load())

        assert out == []

    def test_lazy_load_uses_files_added_since_previous_load(
        self,
        monkeypatch,
        tmp_path,
    ):
        first_file = tmp_path / "first.txt"
        first_file.write_text("first")

        instance = loader.DynamicDirectoryUnstructuredLoader.__new__(
            loader.DynamicDirectoryUnstructuredLoader
        )
        instance.root_dir_path = tmp_path
        instance.rglob = "**/*"
        instance.file_path = []

        observed_file_paths = []

        def fake_lazy_load(self):
            observed_file_paths.append(list(self.file_path))
            yield Document(
                page_content="content",
                metadata={},
            )

        monkeypatch.setattr(
            loader.UnstructuredLoader,
            "lazy_load",
            fake_lazy_load,
        )

        list(instance.lazy_load())

        second_file = tmp_path / "second.txt"
        second_file.write_text("second")

        list(instance.lazy_load())

        assert observed_file_paths[0] == [
            str(first_file),
        ]

        assert set(observed_file_paths[1]) == {
            str(first_file),
            str(second_file),
        }

    def test_load_returns_flat_list_of_documents(
        self,
        monkeypatch,
        tmp_path,
    ):
        document_1 = Document(
            page_content="first",
            metadata={},
        )
        document_2 = Document(
            page_content="second",
            metadata={},
        )

        source_file = tmp_path / "document.txt"
        source_file.write_text("content")

        instance = loader.DynamicDirectoryUnstructuredLoader.__new__(
            loader.DynamicDirectoryUnstructuredLoader
        )
        instance.root_dir_path = tmp_path
        instance.rglob = "**/*"
        instance.file_path = []

        monkeypatch.setattr(
            loader.DynamicDirectoryUnstructuredLoader,
            "lazy_load",
            lambda self: iter([document_1, document_2]),
        )

        result = instance.load()

        assert result == [
            document_1,
            document_2,
        ]

    def test_load_propagates_lazy_load_exception(
        self,
        monkeypatch,
        tmp_path,
    ):
        instance = loader.DynamicDirectoryUnstructuredLoader.__new__(
            loader.DynamicDirectoryUnstructuredLoader
        )
        instance.root_dir_path = tmp_path
        instance.rglob = "**/*"

        def raise_error():
            raise RuntimeError("lazy loading failed")

        monkeypatch.setattr(
            instance,
            "lazy_load",
            raise_error,
        )

        with pytest.raises(
            RuntimeError,
            match="lazy loading failed",
        ):
            instance.load()


class TestLoader:
    def test_initializes_successfully(
        self,
        monkeypatch,
    ):
        captured_args = {}

        class FakeDynamicLoader:
            def __init__(self, **kwargs):
                captured_args.update(kwargs)

        monkeypatch.setattr(
            loader,
            "DynamicDirectoryUnstructuredLoader",
            FakeDynamicLoader,
        )

        instance = loader.Loader(
            {
                "source_dir": "/data/documents",
            },
        )

        assert instance.source_dir == "/data/documents"
        assert instance.stopped is False
        assert isinstance(
            instance.docloader,
            FakeDynamicLoader,
        )

        assert captured_args == {
            "root_dir_path": "/data/documents",
            "glob": "**/*",
        }

    def test_missing_source_dir_raises_key_error(
        self,
    ):
        with pytest.raises(KeyError):
            loader.Loader({})

    def test_source_directory_configuration_exception_is_logged(
        self,
        caplog,
    ):
        class InvalidConfig:
            def __getitem__(self, key):
                raise RuntimeError("configuration failure")

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(RuntimeError, match="configuration failure"),
        ):
            loader.Loader(InvalidConfig())  # type: ignore # pyright: ignore[reportArgumentType] # fmt: ignore

        assert "while setting source directory" in caplog.text

    def test_dynamic_loader_initialization_exception_is_logged(
        self,
        monkeypatch,
        caplog,
    ):
        def raise_error(**kwargs):
            raise RuntimeError("loader initialization failed")

        monkeypatch.setattr(
            loader,
            "DynamicDirectoryUnstructuredLoader",
            raise_error,
        )

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(
                RuntimeError,
                match="loader initialization failed",
            ),
        ):
            loader.Loader(
                {
                    "source_dir": "/data/documents",
                }
            )

        assert "initializing the directory loader" in caplog.text

    def test_parse_docs_raises_when_stopped(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            loader,
            "DynamicDirectoryUnstructuredLoader",
            lambda **kwargs: object(),
        )

        instance = loader.Loader(
            {
                "source_dir": "/data/documents",
            }
        )

        instance.stopped = True

        with pytest.raises(
            RuntimeError,
            match="Indexer is stopped.",
        ):
            instance.parse_docs()

    def test_parse_docs_groups_documents_by_extension(
        self,
        monkeypatch,
    ):
        documents = [
            Document(
                page_content="PDF 1",
                metadata={"source": "/data/a.pdf"},
            ),
            Document(
                page_content="PDF 2",
                metadata={"source": "/data/b.pdf"},
            ),
            Document(
                page_content="Text",
                metadata={"source": "/data/c.txt"},
            ),
        ]

        class FakeDynamicLoader:
            def __init__(self, **kwargs):
                pass

            def load(self):
                return documents

        monkeypatch.setattr(
            loader,
            "DynamicDirectoryUnstructuredLoader",
            FakeDynamicLoader,
        )

        instance = loader.Loader(
            {
                "source_dir": "/data/documents",
            },
            save_documents=True,
        )

        result = instance.parse_docs()

        assert result == {
            ".pdf": [
                documents[0],
                documents[1],
            ],
            ".txt": [
                documents[2],
            ],
        }

    def test_parse_docs_groups_unknown_filetypes(
        self,
        monkeypatch,
    ):
        documents = [
            Document(
                page_content="No extension",
                metadata={"source": "/data/README"},
            ),
            Document(
                page_content="Missing source",
                metadata={},
            ),
            Document(
                page_content="Empty source",
                metadata={"source": ""},
            ),
        ]

        class FakeDynamicLoader:
            def __init__(self, **kwargs):
                pass

            def load(self):
                return documents

        monkeypatch.setattr(
            loader,
            "DynamicDirectoryUnstructuredLoader",
            FakeDynamicLoader,
        )

        instance = loader.Loader(
            {
                "source_dir": "/data/documents",
            }
        )

        result = instance.parse_docs()

        assert result["Unknown"] == documents

    def test_parse_docs_preserves_document_order(
        self,
        monkeypatch,
    ):
        documents = [
            Document(
                page_content="First",
                metadata={"source": "/data/first.pdf"},
            ),
            Document(
                page_content="Second",
                metadata={"source": "/data/second.pdf"},
            ),
            Document(
                page_content="Third",
                metadata={"source": "/data/third.pdf"},
            ),
        ]

        class FakeDynamicLoader:
            def __init__(self, **kwargs):
                pass

            def load(self):
                return documents

        monkeypatch.setattr(
            loader,
            "DynamicDirectoryUnstructuredLoader",
            FakeDynamicLoader,
        )

        instance = loader.Loader(
            {
                "source_dir": "/data/documents",
            }
        )

        result = instance.parse_docs()

        assert result[".pdf"] == documents

    def test_parse_docs_returns_empty_dict_for_no_documents(
        self,
        monkeypatch,
    ):
        class FakeDynamicLoader:
            def __init__(self, **kwargs):
                pass

            def load(self):
                return []

        monkeypatch.setattr(
            loader,
            "DynamicDirectoryUnstructuredLoader",
            FakeDynamicLoader,
        )

        instance = loader.Loader(
            {
                "source_dir": "/data/documents",
            }
        )

        assert instance.parse_docs() == {}

    def test_document_loading_exception_is_logged_and_reraised(
        self,
        monkeypatch,
        caplog,
    ):
        class FakeDynamicLoader:
            def __init__(self, **kwargs):
                pass

            def load(self):
                raise RuntimeError("loading failed")

        monkeypatch.setattr(
            loader,
            "DynamicDirectoryUnstructuredLoader",
            FakeDynamicLoader,
        )

        instance = loader.Loader(
            {
                "source_dir": "/data/documents",
            }
        )

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(
                RuntimeError,
                match="loading failed",
            ),
        ):
            instance.parse_docs()

        assert "loading the data for ingestion" in caplog.text

    def test_grouping_exception_is_logged_and_reraised(
        self,
        monkeypatch,
        caplog,
    ):
        class InvalidDocument:
            @property
            def metadata(self):
                raise RuntimeError("metadata failure")

        class FakeDynamicLoader:
            def __init__(self, **kwargs):
                pass

            def load(self):
                return [InvalidDocument()]

        monkeypatch.setattr(
            loader,
            "DynamicDirectoryUnstructuredLoader",
            FakeDynamicLoader,
        )

        instance = loader.Loader(
            {
                "source_dir": "/data/documents",
            }
        )

        with (
            caplog.at_level(logging.ERROR),
            pytest.raises(
                RuntimeError,
                match="metadata failure",
            ),
        ):
            instance.parse_docs()

        assert "grouping the filetypes" in caplog.text

    def test_parse_docs_does_not_load_when_stopped(
        self,
        monkeypatch,
    ):
        load_called = False

        class FakeDynamicLoader:
            def __init__(self, **kwargs):
                pass

            def load(self):
                nonlocal load_called
                load_called = True
                return []

        monkeypatch.setattr(
            loader,
            "DynamicDirectoryUnstructuredLoader",
            FakeDynamicLoader,
        )

        instance = loader.Loader(
            {
                "source_dir": "/data/documents",
            }
        )

        instance.stopped = True

        with pytest.raises(
            RuntimeError,
            match="Indexer is stopped.",
        ):
            instance.parse_docs()

        assert load_called is False

    def test_cleanup_stops_loader(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            loader,
            "DynamicDirectoryUnstructuredLoader",
            lambda **kwargs: object(),
        )

        instance = loader.Loader(
            {
                "source_dir": "/data/documents",
            }
        )

        assert instance.stopped is False

        instance.cleanup()

        assert instance.stopped is True

    def test_cleanup_is_idempotent(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            loader,
            "DynamicDirectoryUnstructuredLoader",
            lambda **kwargs: object(),
        )

        instance = loader.Loader(
            {
                "source_dir": "/data/documents",
            }
        )

        instance.cleanup()
        instance.cleanup()

        assert instance.stopped is True

    def test_parse_docs_fails_after_cleanup(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            loader,
            "DynamicDirectoryUnstructuredLoader",
            lambda **kwargs: object(),
        )

        instance = loader.Loader(
            {
                "source_dir": "/data/documents",
            }
        )

        instance.cleanup()

        with pytest.raises(
            RuntimeError,
            match="Indexer is stopped.",
        ):
            instance.parse_docs()

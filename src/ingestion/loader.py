import logging
import os
from collections.abc import Iterator
from pathlib import Path

from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DynamicDirectoryUnstructuredLoader(UnstructuredLoader):
    """
    Initializes configuration once, but dynamically scans the target
    root directory for updated files at the exact moment of execution.
    """

    def __init__(self, root_dir_path: str, glob: str, **kwargs):
        self.root_dir_path = Path(root_dir_path)
        # Store configuration options to apply at runtime
        self.loader_kwargs = kwargs
        self.rglob = glob

        # Initialize the base class with empty fields to bypass setup validation
        super().__init__(file_path=[], **kwargs)

    def _get_current_files(self) -> list[str]:
        """Scans the directory for files actively on disk."""
        if not self.root_dir_path.exists():
            return []

        # Pulls all actual files dynamically (skipping directory structures)
        return [str(f) for f in self.root_dir_path.rglob(self.rglob) if f.is_file()]

    def lazy_load(self) -> Iterator[Document]:
        """Intercepts lazy_load to inject the freshly scanned file list."""
        # 1. Update the instance's target file paths right before execution
        self.file_path = self._get_current_files()

        if not self.file_path:
            logger.warning(
                f"Warning: No files found in {self.root_dir_path} at load time."
            )
            return []
        else:
            # 2. Yield documents using Unstructured's built-in processing pipeline
            yield from super().lazy_load()

    def load(self) -> list[Document]:
        """Collates the lazy generator items into a flat list layout."""
        return list(self.lazy_load())


class Loader:
    def __init__(self, config: dict[str, str]) -> None:
        self.stopped = True
        try:
            self.source_dir = config["source_dir"]
        except Exception as e:
            logger.error(f"Got Exception {e} while setting source directory.")
            raise
        try:
            self.docloader = DynamicDirectoryUnstructuredLoader(
                root_dir_path=self.source_dir, glob="**/*"
            )
        except Exception as e:
            logger.error(f"Got Exception {e} initializing the directory loader.")
            raise
        self.stopped = False

    def parse_docs(self) -> dict[str, list[Document]]:
        if self.stopped:
            raise RuntimeError("Indexer is stopped.")
        try:
            documents: list[Document] = self.docloader.load()
        except Exception as e:
            logger.error(f"Got Exception {e} loading the data for ingestion.")
            raise
        grouped_docs: dict[str, list[Document]] = {}
        try:
            for doc in documents:
                filepath = doc.metadata.get("source", "")
                _, ext = os.path.splitext(filepath)
                filetype = ext if ext else "Unknown"
                grouped_docs.setdefault(filetype, []).append(doc)
        except Exception as e:
            logger.error(f"Got Exception {e} grouping the filetypes.")
            raise
        return grouped_docs

    def cleanup(self) -> None:
        self.stopped = True

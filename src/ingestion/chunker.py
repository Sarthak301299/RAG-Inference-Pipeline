import logging
from collections.abc import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Chunker:
    def __init__(self, config: dict[str, str]) -> None:
        self.stopped: bool = True
        try:
            self.strategy: str = config["strategy"]
            if self.strategy not in ["fixed", "recursive", "auto"]:
                raise ValueError("Strategy must be fixed, recursive, or auto.")
            self.chunk_size: int = int(config["chunk_size"])
            self.chunk_overlap: int = int(config["chunk_overlap"])
        except Exception as e:
            logger.error(f"Got Exception {e} reading configuration.")
            raise
        self.stopped = False

    def generate_chunks(
        self, documents: Iterable[Document], extension: str
    ) -> list[Document]:
        if self.stopped:
            raise RuntimeError("Chunker is stopped.")
        try:
            if self.strategy == "recursive":
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                )
                secondary_splitter = None
            elif self.strategy == "fixed":
                splitter = CharacterTextSplitter(
                    chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                )
                secondary_splitter = None
            else:
                if extension in [".pdf", ".txt"]:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                    )
                    secondary_splitter = None
                elif extension == ".c":
                    splitter = RecursiveCharacterTextSplitter.from_language(
                        language=Language.C,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    )
                    secondary_splitter = None
                elif extension == ".cpp":
                    splitter = RecursiveCharacterTextSplitter.from_language(
                        language=Language.CPP,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    )
                    secondary_splitter = None
                elif extension == ".py":
                    splitter = RecursiveCharacterTextSplitter.from_language(
                        language=Language.PYTHON,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    )
                    secondary_splitter = None
                elif extension == ".md":
                    headers_to_split_on = [
                        ("#", "Header_1"),
                        ("##", "Header_2"),
                        ("###", "Header_3"),
                    ]
                    secondary_splitter = MarkdownHeaderTextSplitter(
                        headers_to_split_on=headers_to_split_on
                    )
                    splitter = MarkdownTextSplitter(
                        chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                    )
                else:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                    )
                    secondary_splitter = None
        except Exception as e:
            logger.error(
                f"Got Exception {e} while setting chunking strategy {self.strategy}"
            )
            raise

        try:
            if secondary_splitter is not None:
                docchunks = []
                for doc in documents:
                    headerchunks = secondary_splitter.split_text(doc.page_content)
                    for chunk in headerchunks:
                        chunk.metadata = {**doc.metadata, **chunk.metadata}
                        docchunks.append(chunk)

                chunks = splitter.split_documents(docchunks)
            else:
                chunks = splitter.split_documents(documents=documents)
        except Exception as e:
            logger.error(
                f"Got Exception {e} while splitting documents with extension {extension}"
            )
            raise

        return chunks

    def cleanup(self):
        self.stopped = True

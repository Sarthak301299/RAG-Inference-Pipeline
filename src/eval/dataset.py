import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EvalExample(BaseModel):
    """A single query/ground-truth pair used to evaluate the RAG pipeline."""

    query: str = Field(..., description="The user query to evaluate.")
    expected_answer: str = Field(
        ..., description="A reference answer used for faithfulness comparison."
    )
    relevant_doc_ids: list[str] = Field(
        ...,
        description=(
            "Source identifiers (matching Document.metadata['source']) that are "
            "considered relevant/correct for this query. Used to score retrieval."
        ),
    )


def load_eval_dataset(path: str) -> list[EvalExample]:
    """Loads a JSONL file of EvalExample records.

    Each line must be a JSON object with keys: query, expected_answer,
    relevant_doc_ids. Blank lines are skipped.
    """
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Eval dataset not found at {path}")

    examples: list[EvalExample] = []
    try:
        with open(dataset_path, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                    examples.append(EvalExample(**record))
                except Exception as e:
                    logger.error(
                        f"Got Exception {e} parsing eval dataset line {line_num}"
                    )
                    raise
    except Exception as e:
        logger.error(f"Got Exception {e} reading eval dataset at {path}")
        raise

    if not examples:
        raise ValueError(f"Eval dataset at {path} contained no valid examples.")

    return examples

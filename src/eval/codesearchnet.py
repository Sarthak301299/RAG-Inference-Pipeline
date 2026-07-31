import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

from datasets import load_dataset

from src.eval.dataset import EvalExample

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _load_hf_dataset(
    dataset_name: str, config_name: str, split: str, num_examples: int, seed: int
) -> Iterable[dict[str, Any]]:
    try:
        dataset = load_dataset(dataset_name, config_name, split=split)
        dataset = dataset.shuffle(seed=seed)
        if num_examples > 0:
            dataset = dataset.select(range(min(num_examples, len(dataset))))
        dataset = cast(Iterable[dict[str, Any]], dataset)
    except Exception as e:
        logger.error(
            f"Got Exception {e} loading dataset {dataset_name}/{config_name}"
            f"[{split}] from Hugging Face."
        )
        raise
    return dataset


def _row_to_file_and_example(
    row: dict[str, Any],
    index: int,
    corpus_dir: Path,
    docstring_field: str,
    code_field: str,
    path_field: str,
) -> EvalExample | None:
    """Writes a single CodeSearchNet row to disk as a .py file and builds
    the matching EvalExample. Returns None (and logs a warning) for rows
    missing a usable docstring or code body, rather than raising -- a
    handful of malformed rows shouldn't abort the whole build.
    """
    docstring = str(row.get(docstring_field, "")).strip()
    code = str(row.get(code_field, "")).strip()

    if not docstring or not code:
        logger.warning(f"Skipping row {index}: missing docstring or code.")
        return None

    original_path = str(row.get(path_field, "")).strip()
    safe_suffix = original_path.replace("/", "__").replace("\\", "__") or "unknown"
    file_name = f"{index:05d}_{safe_suffix}"
    if not file_name.endswith(".py"):
        file_name += ".py"

    file_path = corpus_dir / file_name
    try:
        file_path.write_text(code, encoding="utf-8")
    except Exception as e:
        logger.error(f"Got Exception {e} writing {file_path}")
        raise

    return EvalExample(
        query=docstring,
        expected_answer=docstring,
        relevant_doc_ids=[str(file_path)],
    )


def build_codesearchnet_eval_set(
    output_dir: str,
    dataset_name: str = "code_search_net",
    config_name: str = "python",
    split: str = "test",
    num_examples: int = 20,
    seed: int = 42,
    docstring_field: str = "func_documentation_string",
    code_field: str = "func_code_string",
    path_field: str = "func_path_in_repository",
) -> tuple[str, str]:
    """Downloads a CodeSearchNet subset, writes each function as a .py
    file under `<output_dir>/corpus/`, and writes a matching
    qa_set-compatible JSONL to `<output_dir>/qa_set.jsonl`.

    Returns (corpus_dir, qa_set_path) as strings.

    The written corpus directory is intended to be pointed at directly
    via `ingestion.loading.source_dir` in config/config.yml before
    running the pipeline's ingestion + eval steps.
    """
    output_path = Path(output_dir)
    corpus_dir = output_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_hf_dataset(
        dataset_name=dataset_name,
        config_name=config_name,
        split=split,
        num_examples=num_examples,
        seed=seed,
    )

    examples: list[EvalExample] = []
    for index, row in enumerate(rows):
        example = _row_to_file_and_example(
            row=row,
            index=index,
            corpus_dir=corpus_dir,
            docstring_field=docstring_field,
            code_field=code_field,
            path_field=path_field,
        )
        if example is not None:
            examples.append(example)

    if not examples:
        raise ValueError(
            "No usable CodeSearchNet examples were found -- check "
            "docstring_field/code_field/path_field against the actual "
            "dataset schema."
        )

    qa_set_path = output_path / "qa_set.jsonl"
    try:
        with open(qa_set_path, "w", encoding="utf-8") as file:
            for example in examples:
                file.write(example.model_dump_json() + "\n")
    except Exception as e:
        logger.error(f"Got Exception {e} writing {qa_set_path}")
        raise

    logger.info(
        f"Wrote {len(examples)} CodeSearchNet examples to {corpus_dir} "
        f"and {qa_set_path}"
    )
    return str(corpus_dir), str(qa_set_path)


def main() -> None:
    import yaml

    with open("config/config.yml", "r") as file:
        config = yaml.safe_load(file)

    csn_config = config.get("eval", {}).get("codesearchnet")
    if csn_config is None:
        raise KeyError("No 'eval.codesearchnet' section found in config/config.yml.")

    build_codesearchnet_eval_set(
        output_dir=csn_config["output_dir"],
        dataset_name=csn_config.get("dataset_name", "code_search_net"),
        config_name=csn_config.get("config_name", "python"),
        split=csn_config.get("split", "test"),
        num_examples=int(csn_config.get("num_examples", 20)),
        seed=int(csn_config.get("seed", 42)),
        docstring_field=csn_config.get("docstring_field", "func_documentation_string"),
        code_field=csn_config.get("code_field", "func_code_string"),
        path_field=csn_config.get("path_field", "func_path_in_repository"),
    )


if __name__ == "__main__":
    main()  # pragma: no cover

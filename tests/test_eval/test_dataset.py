import json

import pytest
from pydantic import ValidationError

from src.eval.dataset import EvalExample, load_eval_dataset


def _write_jsonl(path, records) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(record) + "\n" for record in records)


def test_load_eval_dataset_valid(tmp_path):
    dataset_path = tmp_path / "qa_set.jsonl"
    _write_jsonl(
        dataset_path,
        [
            {
                "query": "What is X?",
                "expected_answer": "X is Y.",
                "relevant_doc_ids": ["doc1.py"],
            },
            {
                "query": "What is Z?",
                "expected_answer": "Z is W.",
                "relevant_doc_ids": ["doc2.py", "doc3.py"],
            },
        ],
    )

    examples = load_eval_dataset(str(dataset_path))

    assert len(examples) == 2
    assert isinstance(examples[0], EvalExample)
    assert examples[0].query == "What is X?"
    assert examples[1].relevant_doc_ids == ["doc2.py", "doc3.py"]


def test_load_eval_dataset_skips_blank_lines(tmp_path):
    dataset_path = tmp_path / "qa_set.jsonl"
    dataset_path.write_text(
        '{"query": "q", "expected_answer": "a", "relevant_doc_ids": ["d"]}\n'
        "\n"
        "   \n"
    )

    examples = load_eval_dataset(str(dataset_path))

    assert len(examples) == 1


def test_load_eval_dataset_missing_file():
    with pytest.raises(FileNotFoundError):
        load_eval_dataset("does/not/exist.jsonl")


def test_load_eval_dataset_malformed_json(tmp_path):
    dataset_path = tmp_path / "qa_set.jsonl"
    dataset_path.write_text("{not valid json\n")

    with pytest.raises(json.decoder.JSONDecodeError):
        load_eval_dataset(str(dataset_path))


def test_load_eval_dataset_missing_required_field(tmp_path):
    dataset_path = tmp_path / "qa_set.jsonl"
    _write_jsonl(dataset_path, [{"query": "q", "expected_answer": "a"}])

    with pytest.raises(ValidationError):
        load_eval_dataset(str(dataset_path))


def test_load_eval_dataset_empty_file_raises(tmp_path):
    dataset_path = tmp_path / "qa_set.jsonl"
    dataset_path.write_text("")

    with pytest.raises(ValueError):
        load_eval_dataset(str(dataset_path))

import json
from pathlib import Path

import pytest
import yaml

from src.eval import codesearchnet as csn_module
from src.eval.codesearchnet import (
    _load_hf_dataset,
    _row_to_file_and_example,
    build_codesearchnet_eval_set,
    main,
)
from src.eval.dataset import EvalExample

FAKE_ROWS = [
    {
        "func_documentation_string": "Adds two numbers together.",
        "func_code_string": "def add(a, b):\n    return a + b\n",
        "func_path_in_repository": "mathlib/ops.py",
    },
    {
        "func_documentation_string": "Subtracts b from a.",
        "func_code_string": "def subtract(a, b):\n    return a - b\n",
        "func_path_in_repository": "mathlib/ops.py",  # duplicate path, different fn
    },
]


def test_row_to_file_and_example_writes_file_and_builds_example(tmp_path):
    example = _row_to_file_and_example(
        row=FAKE_ROWS[0],
        index=0,
        corpus_dir=tmp_path,
        docstring_field="func_documentation_string",
        code_field="func_code_string",
        path_field="func_path_in_repository",
    )

    assert example is not None
    assert isinstance(example, EvalExample)
    assert example.query == "Adds two numbers together."
    assert len(example.relevant_doc_ids) == 1

    written_path = tmp_path / "00000_mathlib__ops.py"
    assert written_path.exists()
    assert "def add" in written_path.read_text()
    assert example.relevant_doc_ids[0] == str(written_path)


def test_row_to_file_raises_on_invalid_path():
    with pytest.raises(FileNotFoundError):
        _ = _row_to_file_and_example(
            row=FAKE_ROWS[0],
            index=0,
            corpus_dir=Path("invalid_path"),
            docstring_field="func_documentation_string",
            code_field="func_code_string",
            path_field="func_path_in_repository",
        )


def test_row_to_file_and_example_missing_docstring_returns_none(tmp_path):
    row = {
        "func_documentation_string": "",
        "func_code_string": "def f(): pass",
        "func_path_in_repository": "a.py",
    }

    example = _row_to_file_and_example(
        row=row,
        index=0,
        corpus_dir=tmp_path,
        docstring_field="func_documentation_string",
        code_field="func_code_string",
        path_field="func_path_in_repository",
    )

    assert example is None
    assert list(tmp_path.iterdir()) == []


def test_row_to_file_and_example_missing_code_returns_none(tmp_path):
    row = {
        "func_documentation_string": "does something",
        "func_code_string": "   ",
        "func_path_in_repository": "a.py",
    }

    example = _row_to_file_and_example(
        row=row,
        index=0,
        corpus_dir=tmp_path,
        docstring_field="func_documentation_string",
        code_field="func_code_string",
        path_field="func_path_in_repository",
    )

    assert example is None


def test_row_to_file_and_example_missing_path_field_falls_back_to_unknown(tmp_path):
    row = {
        "func_documentation_string": "does something",
        "func_code_string": "def f(): pass",
    }

    example = _row_to_file_and_example(
        row=row,
        index=3,
        corpus_dir=tmp_path,
        docstring_field="func_documentation_string",
        code_field="func_code_string",
        path_field="func_path_in_repository",
    )

    assert example is not None
    written_path = tmp_path / "00003_unknown.py"
    assert written_path.exists()


def test_build_codesearchnet_eval_set_success(tmp_path, monkeypatch):
    monkeypatch.setattr(
        csn_module,
        "_load_hf_dataset",
        lambda **kwargs: iter(FAKE_ROWS),
    )

    corpus_dir, qa_set_path = build_codesearchnet_eval_set(
        output_dir=str(tmp_path), num_examples=2
    )

    corpus_files = list((tmp_path / "corpus").glob("*.py"))
    assert len(corpus_files) == 2

    with open(qa_set_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    assert len(lines) == 2
    assert lines[0]["query"] == "Adds two numbers together."
    assert lines[1]["query"] == "Subtracts b from a."
    assert corpus_dir == str(tmp_path / "corpus")


def test_build_codesearchnet_eval_set_raises_on(tmp_path, monkeypatch):
    monkeypatch.setattr(
        csn_module,
        "_load_hf_dataset",
        lambda **kwargs: iter(FAKE_ROWS),
    )

    def dummy_open(fname, mode, encoding):
        raise RuntimeError

    monkeypatch.setattr("builtins.open", dummy_open)

    with pytest.raises(RuntimeError):
        _, _ = build_codesearchnet_eval_set(output_dir=str(tmp_path), num_examples=2)


def test_build_codesearchnet_eval_set_skips_bad_rows(tmp_path, monkeypatch):
    rows = FAKE_ROWS + [
        {
            "func_documentation_string": "",
            "func_code_string": "def broken(): pass",
            "func_path_in_repository": "broken.py",
        }
    ]
    monkeypatch.setattr(csn_module, "_load_hf_dataset", lambda **kwargs: iter(rows))

    _, _ = build_codesearchnet_eval_set(output_dir=str(tmp_path))

    corpus_files = list((tmp_path / "corpus").glob("*.py"))
    assert len(corpus_files) == 2  # the malformed row was skipped, not written


def test_build_codesearchnet_eval_set_all_rows_invalid_raises(tmp_path, monkeypatch):
    bad_rows = [
        {
            "func_documentation_string": "",
            "func_code_string": "",
            "func_path_in_repository": "x.py",
        }
    ]
    monkeypatch.setattr(csn_module, "_load_hf_dataset", lambda **kwargs: iter(bad_rows))

    with pytest.raises(ValueError, match="No usable CodeSearchNet examples"):
        build_codesearchnet_eval_set(output_dir=str(tmp_path))


def test_build_codesearchnet_eval_set_creates_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        csn_module, "_load_hf_dataset", lambda **kwargs: iter(FAKE_ROWS)
    )
    nested_output = tmp_path / "does" / "not" / "exist"

    _, _ = build_codesearchnet_eval_set(output_dir=str(nested_output))

    assert (nested_output / "corpus").exists()
    assert (nested_output / "qa_set.jsonl").exists()


def test_main_reads_config_and_builds(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "config.yml"
    output_dir = tmp_path / "csn_output"
    config_path.write_text(
        yaml.dump(
            {
                "eval": {
                    "codesearchnet": {
                        "output_dir": str(output_dir),
                        "num_examples": 2,
                    }
                }
            }
        )
    )

    monkeypatch.chdir(tmp_path)

    called_kwargs = {}

    def fake_build(**kwargs):
        called_kwargs.update(kwargs)
        return (str(output_dir / "corpus"), str(output_dir / "qa_set.jsonl"))

    monkeypatch.setattr(csn_module, "build_codesearchnet_eval_set", fake_build)

    main()

    assert called_kwargs["output_dir"] == str(output_dir)
    assert called_kwargs["num_examples"] == 2
    assert called_kwargs["dataset_name"] == "code_search_net"


def test_main_missing_config_section_raises(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "config.yml"
    config_path.write_text(yaml.dump({"eval": {}}))

    monkeypatch.chdir(tmp_path)

    with pytest.raises(KeyError):
        main()


class FakeDataset:
    def __init__(self, items):
        self.items = list(items)
        self.shuffle_seed = None
        self.selected = None

    def shuffle(self, seed):
        self.shuffle_seed = seed
        return self

    def select(self, indices):
        self.selected = list(indices)
        self.items = [self.items[i] for i in indices]
        return self

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


def test_load_dataset_without_selection(monkeypatch):
    dataset = FakeDataset([{"id": 1}, {"id": 2}, {"id": 3}])

    monkeypatch.setattr(
        csn_module,
        "load_dataset",
        lambda *args, **kwargs: dataset,
    )

    result = _load_hf_dataset(
        "dataset",
        "config",
        "train",
        0,
        123,
    )

    assert result is dataset
    assert dataset.shuffle_seed == 123
    assert dataset.selected is None


def test_load_dataset_with_selection(monkeypatch):
    dataset = FakeDataset([{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}])

    monkeypatch.setattr(
        csn_module,
        "load_dataset",
        lambda *args, **kwargs: dataset,
    )

    result = _load_hf_dataset(
        "dataset",
        "config",
        "train",
        2,
        42,
    )

    assert result is dataset
    assert dataset.shuffle_seed == 42
    assert dataset.selected == [0, 1]
    assert list(result) == [{"id": 1}, {"id": 2}]


def test_load_dataset_selection_larger_than_dataset(monkeypatch):
    dataset = FakeDataset([{"id": 1}, {"id": 2}])

    monkeypatch.setattr(
        csn_module,
        "load_dataset",
        lambda *args, **kwargs: dataset,
    )

    result = _load_hf_dataset(
        "dataset",
        "config",
        "train",
        10,
        7,
    )

    assert result is dataset
    assert dataset.selected == [0, 1]


def test_load_dataset_failure(monkeypatch):
    def fake_load_dataset(*args, **kwargs):
        raise RuntimeError("load failed")

    monkeypatch.setattr(
        csn_module,
        "load_dataset",
        fake_load_dataset,
    )

    with pytest.raises(RuntimeError, match="load failed"):
        _load_hf_dataset(
            "dataset",
            "config",
            "train",
            1,
            0,
        )

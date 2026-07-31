import json

import pytest
import yaml

from src.eval import run_eval as run_eval_module
from src.eval.dataset import EvalExample
from src.eval.run_eval import (
    _aggregate,
    _extract_answer,
    _load_eval_config,
    main,
    run_eval,
)


def test_extract_answer_valid_json():
    raw = json.dumps({"thought_process": "t", "answer": "the answer", "sources": []})

    assert _extract_answer(raw) == "the answer"


def test_extract_answer_malformed_json_falls_back_to_raw():
    raw = "not json"

    assert _extract_answer(raw) == "not json"


def test_load_eval_config_valid(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        yaml.dump(
            {
                "eval": {
                    "dataset_path": "data/eval/qa_set.jsonl",
                    "k_values": [3, 5],
                    "output_dir": "results/eval",
                }
            }
        )
    )

    eval_config = _load_eval_config(str(config_path))

    assert eval_config["dataset_path"] == "data/eval/qa_set.jsonl"
    assert eval_config["k_values"] == [3, 5]


def test_load_eval_config_missing_section_raises(tmp_path):
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.dump({"project": {"name": "x"}}))

    with pytest.raises(KeyError):
        _load_eval_config(str(config_path))


def test_load_eval_config_invalid_file():
    config_path = "dummyconfig"

    with pytest.raises(FileNotFoundError):
        _load_eval_config(str(config_path))


def test_aggregate_computes_averages():
    per_example = [
        {
            "precision_at_k": {3: 1.0},
            "recall_at_k": {3: 0.5},
            "mrr": 1.0,
            "faithfulness": 1.0,
        },
        {
            "precision_at_k": {3: 0.0},
            "recall_at_k": {3: 0.5},
            "mrr": 0.0,
            "faithfulness": 0.0,
        },
    ]

    summary = _aggregate(per_example, k_values=[3])

    assert summary["num_examples"] == 2
    assert summary["precision_at_k"][3] == pytest.approx(0.5)
    assert summary["recall_at_k"][3] == pytest.approx(0.5)
    assert summary["mrr"] == pytest.approx(0.5)
    assert summary["faithfulness"] == pytest.approx(0.5)


class FakePipeline:
    def __init__(self):
        self.ingested = False
        self.shut_down = False
        self.generator = object()

    def ingest_data(self):
        self.ingested = True

    def shutdown(self):
        self.shut_down = True

    async def generate_with_retrieval(self, queries):
        return [
            {
                "query": queries[0],
                "retrieved_ids": ["doc1.py"],
                "context_chunks": ["some context"],
                "response": json.dumps(
                    {
                        "thought_process": "t",
                        "answer": "the answer",
                        "sources": ["doc1.py"],
                    }
                ),
            }
        ]


@pytest.mark.asyncio
async def test_run_eval_end_to_end(tmp_path, monkeypatch):
    output_dir = tmp_path / "results"

    monkeypatch.setattr(
        run_eval_module,
        "_load_eval_config",
        lambda: {
            "dataset_path": "unused",
            "k_values": [1],
            "output_dir": str(output_dir),
        },
    )
    monkeypatch.setattr(
        run_eval_module,
        "load_eval_dataset",
        lambda path: [
            EvalExample(
                query="What is X?",
                expected_answer="X is Y.",
                relevant_doc_ids=["doc1.py"],
            )
        ],
    )
    monkeypatch.setattr(run_eval_module, "RAGPipeLine", FakePipeline)

    async def fake_faithfulness_score(question, answer, context_chunks, judge):
        return 1.0

    monkeypatch.setattr(run_eval_module, "faithfulness_score", fake_faithfulness_score)

    summary = await run_eval()

    assert summary["num_examples"] == 1
    assert summary["precision_at_k"][1] == 1.0
    assert summary["faithfulness"] == 1.0
    assert (output_dir / "summary.json").exists()


@pytest.mark.asyncio
async def test_run_eval_shuts_down_pipeline_on_failure(tmp_path, monkeypatch):
    output_dir = tmp_path / "results"

    class FailingPipeline(FakePipeline):
        async def generate_with_retrieval(self, queries):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        run_eval_module,
        "_load_eval_config",
        lambda: {
            "dataset_path": "unused",
            "k_values": [1],
            "output_dir": str(output_dir),
        },
    )
    monkeypatch.setattr(
        run_eval_module,
        "load_eval_dataset",
        lambda path: [
            EvalExample(query="q", expected_answer="a", relevant_doc_ids=["d"])
        ],
    )

    failing_pipeline = FailingPipeline()
    monkeypatch.setattr(run_eval_module, "RAGPipeLine", lambda: failing_pipeline)

    with pytest.raises(RuntimeError, match="boom"):
        await run_eval()

    assert failing_pipeline.shut_down is True


def test_main(monkeypatch):
    class DummyAsyncio:
        ran = False

        def run(self, input_coro):
            self.ran = True
            input_coro.close()

    obj = DummyAsyncio()
    monkeypatch.setattr(run_eval_module, "asyncio", obj)

    main()

    assert obj.ran

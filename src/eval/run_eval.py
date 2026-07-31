import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, cast

import anyio
import yaml

from src.eval.dataset import EvalExample, load_eval_dataset
from src.eval.generation_metrics import faithfulness_score
from src.eval.retrieval_metrics import average_metric, mrr, precision_at_k, recall_at_k
from src.pipeline.rag_pipeline import RAGPipeLine

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _load_eval_config(config_path: str = "config/config.yml") -> dict[str, Any]:
    try:
        with open(config_path, "r") as file:
            full_config = yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Got Exception {e} parsing configuration file")
        raise

    eval_config = full_config.get("eval")
    if eval_config is None:
        raise KeyError(
            "No 'eval' section found in config/config.yml. See "
            "README for the expected format."
        )
    return eval_config


def _extract_answer(raw_response: str) -> str:
    """Pulls the 'answer' field out of the generator's structured JSON output.

    Falls back to the raw string if parsing fails, so a malformed response
    still gets scored (as unfaithful) rather than crashing the eval run.
    """
    try:
        parsed = json.loads(raw_response)
        return str(parsed.get("answer", raw_response))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Got Exception {e} parsing generator response as JSON.")
        return raw_response


async def _evaluate_example(
    pipeline: RAGPipeLine, example: EvalExample, k_values: list[int]
) -> dict[str, Any]:
    result = (await pipeline.generate_with_retrieval([example.query]))[0]
    retrieved_ids: list[str] = cast(list[str], result["retrieved_ids"])
    context_chunks: list[str] = cast(list[str], result["context_chunks"])
    answer = _extract_answer(str(result["response"]))

    per_k_precision = {
        k: precision_at_k(retrieved_ids, example.relevant_doc_ids, k) for k in k_values
    }
    per_k_recall = {
        k: recall_at_k(retrieved_ids, example.relevant_doc_ids, k) for k in k_values
    }
    reciprocal_rank = mrr(retrieved_ids, example.relevant_doc_ids)

    faithfulness = await faithfulness_score(
        question=example.query,
        answer=answer,
        context_chunks=context_chunks,
        judge=pipeline.generator,
    )

    return {
        "query": example.query,
        "answer": answer,
        "retrieved_ids": retrieved_ids,
        "relevant_doc_ids": example.relevant_doc_ids,
        "precision_at_k": per_k_precision,
        "recall_at_k": per_k_recall,
        "mrr": reciprocal_rank,
        "faithfulness": faithfulness,
    }


def _aggregate(
    per_example: list[dict[str, Any]], k_values: list[int]
) -> dict[str, Any]:
    return {
        "num_examples": len(per_example),
        "precision_at_k": {
            k: average_metric([ex["precision_at_k"][k] for ex in per_example])
            for k in k_values
        },
        "recall_at_k": {
            k: average_metric([ex["recall_at_k"][k] for ex in per_example])
            for k in k_values
        },
        "mrr": average_metric([ex["mrr"] for ex in per_example]),
        "faithfulness": average_metric([ex["faithfulness"] for ex in per_example]),
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"\nEvaluated {summary['num_examples']} examples\n")
    print("Retrieval:")
    for k, score in summary["precision_at_k"].items():
        print(f"  precision@{k}: {score:.3f}")
    for k, score in summary["recall_at_k"].items():
        print(f"  recall@{k}:    {score:.3f}")
    print(f"  MRR:           {summary['mrr']:.3f}")
    print("\nGeneration:")
    print(f"  Faithfulness:  {summary['faithfulness']:.3f}")


async def run_eval() -> dict[str, Any]:
    eval_config = _load_eval_config()
    dataset_path = eval_config["dataset_path"]
    k_values = [int(k) for k in eval_config["k_values"]]
    output_dir = Path(eval_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_eval_dataset(dataset_path)
    logger.info(f"Loaded {len(examples)} eval examples from {dataset_path}")

    pipeline = RAGPipeLine()
    pipeline.ingest_data()
    try:
        per_example: list[dict[str, Any]] = []
        for example in examples:
            result = await _evaluate_example(pipeline, example, k_values)
            per_example.append(result)
            logger.info(f"Evaluated: {example.query[:60]!r}")

        summary = _aggregate(per_example, k_values)
    finally:
        pipeline.shutdown()

    output_path = output_dir / "summary.json"
    json_data = json.dumps({"summary": summary, "examples": per_example}, indent=2)
    async with await anyio.open_file(output_path, mode="w") as file:
        await file.write(json_data)
    logger.info(f"Wrote eval results to {output_path}")

    _print_summary(summary)
    return summary


def main() -> None:
    os.environ["CC"] = "gcc-14"
    os.environ["CXX"] = "g++-14"
    os.environ["CUDAHOSTCXX"] = "g++-14"
    asyncio.run(run_eval())


if __name__ == "__main__":
    main()  # pragma: no cover

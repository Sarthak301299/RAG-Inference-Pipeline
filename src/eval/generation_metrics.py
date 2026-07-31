import json
import logging
from typing import Protocol
from uuid import uuid4

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Maps a judge verdict to a numeric score. A 3-point scale is intentionally
# simple -- it is far easier to get a judge model to reliably pick one of
# three labels than to produce a stable continuous score.
_VERDICT_SCORES: dict[str, float] = {
    "yes": 1.0,
    "partial": 0.5,
    "no": 0.0,
}

_JUDGE_SYSTEM_PROMPT = (
    "You are a strict, impartial evaluator. You will be given a QUESTION, "
    "an ANSWER produced by an AI assistant, and the CONTEXT the assistant "
    "was given to answer from.\n"
    "Judge whether the ANSWER is faithful to the CONTEXT -- i.e. whether "
    "every claim in the ANSWER is actually supported by the CONTEXT, with "
    "no fabricated or unsupported information.\n"
    "Respond with exactly one word in the 'answer' field: 'yes' if fully "
    "supported, 'partial' if some claims are unsupported, or 'no' if the "
    "answer is largely unsupported or contradicts the context.\n"
)


class JudgeGenerator(Protocol):
    """Structural type for anything that can act as an LLM judge.

    Matches the existing src.generation.backend.Generator interface, so a
    VLLMGenerator instance (or any other Generator subclass) can be passed
    in directly without a concrete import/dependency here.
    """

    async def generate(self, prompt: str, request_id: str) -> str: ...


def _build_judge_prompt(question: str, answer: str, context: str) -> str:
    prompt = _JUDGE_SYSTEM_PROMPT
    prompt += f"--- QUESTION ---\n{question}\n\n"
    prompt += f"--- CONTEXT ---\n{context}\n\n"
    prompt += f"--- ANSWER ---\n{answer}\n\n"
    prompt += "Response: "
    return prompt


def _parse_verdict(raw_output: str) -> str:
    """Extracts the verdict label from a generator response.

    The generator backend returns a JSON string matching the pipeline's
    structured RAG output schema (thought_process/answer/sources), so the
    verdict is expected in the 'answer' field.
    """
    try:
        parsed = json.loads(raw_output)
        verdict = str(parsed.get("answer", "")).strip().lower()
    except Exception as e:  # noqa: BLE001
        logger.error(f"Got Exception {e} parsing judge output, treating as 'no'.")
        return "no"

    for label in _VERDICT_SCORES:
        if label in verdict:
            return label

    logger.warning(f"Unrecognized judge verdict '{verdict}', treating as 'no'.")
    return "no"


async def faithfulness_score(
    question: str, answer: str, context_chunks: list[str], judge: JudgeGenerator
) -> float:
    """Scores how well `answer` is supported by `context_chunks`.

    Returns a float in {0.0, 0.5, 1.0} corresponding to no/partial/yes.
    """
    context = "\n\n".join(context_chunks)
    prompt = _build_judge_prompt(question=question, answer=answer, context=context)

    try:
        raw_output = await judge.generate(prompt=prompt, request_id=str(uuid4()))
    except Exception as e:
        logger.error(f"Got Exception {e} while querying judge model.")
        raise

    verdict = _parse_verdict(raw_output)
    return _VERDICT_SCORES[verdict]

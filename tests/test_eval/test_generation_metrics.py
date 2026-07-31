import json

import pytest

from src.eval.generation_metrics import faithfulness_score


class FakeJudge:
    def __init__(self, verdict: str):
        self.verdict = verdict
        self.last_prompt: str | None = None

    async def generate(self, prompt: str, request_id: str) -> str:
        self.last_prompt = prompt
        return json.dumps(
            {
                "thought_process": "checked claims against context",
                "answer": self.verdict,
                "sources": [],
            }
        )


@pytest.mark.asyncio
async def test_faithfulness_score_yes():
    judge = FakeJudge("yes")

    score = await faithfulness_score(
        question="What is X?",
        answer="X is Y.",
        context_chunks=["X is Y, according to the docs."],
        judge=judge,
    )

    assert score == 1.0
    assert judge.last_prompt is not None
    assert "What is X?" in judge.last_prompt


@pytest.mark.asyncio
async def test_faithfulness_score_partial():
    judge = FakeJudge("partial")

    score = await faithfulness_score(
        question="q", answer="a", context_chunks=["c1", "c2"], judge=judge
    )

    assert score == 0.5


@pytest.mark.asyncio
async def test_faithfulness_score_no():
    judge = FakeJudge("no")

    score = await faithfulness_score(
        question="q", answer="a", context_chunks=["c"], judge=judge
    )

    assert score == 0.0


@pytest.mark.asyncio
async def test_faithfulness_score_unparseable_output_defaults_to_zero():
    class BrokenJudge:
        async def generate(self, prompt: str, request_id: str) -> str:
            return "not json at all"

    score = await faithfulness_score(
        question="q", answer="a", context_chunks=["c"], judge=BrokenJudge()
    )

    assert score == 0.0


@pytest.mark.asyncio
async def test_faithfulness_score_unrecognized_verdict_defaults_to_zero():
    judge = FakeJudge("maybe")

    score = await faithfulness_score(
        question="q", answer="a", context_chunks=["c"], judge=judge
    )

    assert score == 0.0


@pytest.mark.asyncio
async def test_faithfulness_score_judge_raises_propagates():
    class RaisingJudge:
        async def generate(self, prompt: str, request_id: str) -> str:
            raise RuntimeError("judge unavailable")

    with pytest.raises(RuntimeError, match="judge unavailable"):
        await faithfulness_score(
            question="q", answer="a", context_chunks=["c"], judge=RaisingJudge()
        )

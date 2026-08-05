import json

import pytest

from src.agent.agent import Agent, MaxIterationsExceeded
from src.agent.tools import CalculatorTool, Tool


class _EchoTool(Tool):
    name = "echo"
    description = "Returns its input unchanged."

    def run(self, tool_input: str) -> str:
        return f"echoed: {tool_input}"


class _ScriptedGenerator:
    """Fake generator returning a pre-programmed sequence of AgentStep
    JSON strings, one per call -- lets us test the loop's control flow
    without any real model."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.calls: list[dict] = []
        self._index = 0

    async def generate(self, prompt, request_id, max_tokens=None):
        self.calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
            }
        )
        if self._index >= len(self.responses):
            raise AssertionError("Generator called more times than scripted.")
        response = self.responses[self._index]
        self._index += 1
        return response


def _step(thought: str, action: str, action_input: str) -> str:
    return json.dumps(
        {"thought": thought, "action": action, "action_input": action_input}
    )


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_agent_requires_at_least_one_tool():
    with pytest.raises(ValueError, match="at least one tool"):
        Agent(step_generator=_ScriptedGenerator([]), tools=[])


def test_agent_requires_positive_max_iterations():
    with pytest.raises(ValueError, match="max_iterations"):
        Agent(
            step_generator=_ScriptedGenerator([]), tools=[_EchoTool()], max_iterations=0
        )


# ---------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_immediate_final_answer():
    generator = _ScriptedGenerator([_step("I already know.", "final_answer", "42")])
    agent = Agent(step_generator=generator, tools=[_EchoTool()])

    result = await agent.run("what is the answer?")

    assert result["answer"] == "42"
    assert result["scratchpad"] == []
    assert result["iterations_used"] == 1


@pytest.mark.asyncio
async def test_agent_single_tool_call_then_final_answer():
    generator = _ScriptedGenerator(
        [
            _step("Let me echo this.", "echo", "hello"),
            _step("Now I know.", "final_answer", "echoed: hello"),
        ]
    )
    agent = Agent(step_generator=generator, tools=[_EchoTool()])

    result = await agent.run("echo hello please")

    assert result["answer"] == "echoed: hello"
    assert isinstance(result["scratchpad"], list)
    assert len(result["scratchpad"]) == 1
    assert result["scratchpad"][0]["action"] == "echo"
    assert result["scratchpad"][0]["observation"] == "echoed: hello"
    assert result["iterations_used"] == 2


@pytest.mark.asyncio
async def test_agent_multi_tool_routing():
    generator = _ScriptedGenerator(
        [
            _step("Let me calculate.", "calculator", "2 + 2"),
            _step("Now let me echo the result.", "echo", "4"),
            _step("Done.", "final_answer", "The answer is 4."),
        ]
    )
    agent = Agent(step_generator=generator, tools=[CalculatorTool(), _EchoTool()])

    result = await agent.run("what is 2+2, then echo it")

    assert isinstance(result["scratchpad"], list)
    assert len(result["scratchpad"]) == 2
    assert result["scratchpad"][0]["action"] == "calculator"
    assert result["scratchpad"][0]["observation"] == "4"
    assert result["scratchpad"][1]["action"] == "echo"
    assert result["scratchpad"][1]["observation"] == "echoed: 4"
    assert result["answer"] == "The answer is 4."


@pytest.mark.asyncio
async def test_agent_scratchpad_accumulates_into_subsequent_prompts():
    generator = _ScriptedGenerator(
        [
            _step("First step.", "echo", "one"),
            _step("Second step.", "final_answer", "done"),
        ]
    )
    agent = Agent(step_generator=generator, tools=[_EchoTool()])

    await agent.run("query")

    # The second prompt should contain the first step's full record --
    # this is the actual state-accumulation mechanism (stateless model,
    # state reconstructed via prompt growth).
    second_prompt = generator.calls[1]["prompt"]
    assert "First step." in second_prompt
    assert "echoed: one" in second_prompt


# ---------------------------------------------------------------------
# Termination / halting behavior
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_raises_when_never_reaching_final_answer():
    generator = _ScriptedGenerator(
        [
            _step("Loop.", "echo", "a"),
            _step("Loop again.", "echo", "b"),
        ]
    )
    agent = Agent(step_generator=generator, tools=[_EchoTool()], max_iterations=2)

    with pytest.raises(MaxIterationsExceeded) as exc_info:
        await agent.run("never-ending query")

    assert len(exc_info.value.scratchpad) == 2


@pytest.mark.asyncio
async def test_agent_max_iterations_exceeded_preserves_partial_scratchpad():
    generator = _ScriptedGenerator([_step("Working on it.", "echo", "partial")])
    agent = Agent(step_generator=generator, tools=[_EchoTool()], max_iterations=1)

    with pytest.raises(MaxIterationsExceeded) as exc_info:
        await agent.run("query")

    assert exc_info.value.scratchpad[0]["observation"] == "echoed: partial"


# ---------------------------------------------------------------------
# Error handling / robustness
# ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_unparseable_step_does_not_crash_the_run():
    generator = _ScriptedGenerator(
        [
            "this is not valid json at all",
            _step("Recovered.", "final_answer", "ok"),
        ]
    )
    agent = Agent(step_generator=generator, tools=[_EchoTool()], max_iterations=3)

    result = await agent.run("query")

    assert result["answer"] == "ok"
    assert isinstance(result["scratchpad"], list)
    assert result["scratchpad"][0]["action"] == "(none)"
    assert "could not be parsed" in result["scratchpad"][0]["observation"]


@pytest.mark.asyncio
async def test_agent_repeated_unparseable_steps_eventually_hit_max_iterations():
    generator = _ScriptedGenerator(["garbage", "still garbage"])
    agent = Agent(step_generator=generator, tools=[_EchoTool()], max_iterations=2)

    with pytest.raises(MaxIterationsExceeded) as exc_info:
        await agent.run("query")

    assert len(exc_info.value.scratchpad) == 2
    assert all(step["action"] == "(none)" for step in exc_info.value.scratchpad)


@pytest.mark.asyncio
async def test_agent_repeated_same_actions_trigger_error_observation():
    generator = _ScriptedGenerator(
        [
            _step("Loop.", "echo", "a"),
            _step("Loop.", "echo", "a"),
            _step("Loop.", "echo", "a"),
            _step("Loop.", "echo", "a"),
            _step("Loop.", "echo", "a"),
            _step("Loop.", "echo", "a"),
        ]
    )
    agent = Agent(step_generator=generator, tools=[_EchoTool()], max_iterations=6)

    with pytest.raises(MaxIterationsExceeded) as exc_info:
        await agent.run("query")

    assert len(exc_info.value.scratchpad) == 6
    assert "DIFFERENT" in exc_info.value.scratchpad[-1]["observation"]


@pytest.mark.asyncio
async def test_agent_unknown_action_from_model_is_handled_as_observation():
    # Simulates the model naming a tool that isn't registered -- in
    # practice guided decoding should prevent this, but the loop
    # shouldn't crash even if it happens (e.g. a bug in schema wiring).
    generator = _ScriptedGenerator(
        [
            _step("Trying a tool that doesn't exist.", "nonexistent_tool", "x"),
            _step("Giving up on that.", "final_answer", "fallback answer"),
        ]
    )
    agent = Agent(step_generator=generator, tools=[_EchoTool()], max_iterations=3)

    result = await agent.run("query")

    assert isinstance(result["scratchpad"], list)
    assert "Error: unknown tool" in result["scratchpad"][0]["observation"]
    assert result["answer"] == "fallback answer"


@pytest.mark.asyncio
async def test_agent_tool_exception_becomes_observation_not_crash():
    class ExplodingTool(Tool):
        name = "exploding"
        description = "Always raises."

        def run(self, tool_input: str) -> str:
            raise RuntimeError("boom")

    generator = _ScriptedGenerator(
        [
            _step("Use the exploding tool.", "exploding", "x"),
            _step("That failed, giving up.", "final_answer", "could not complete"),
        ]
    )
    agent = Agent(step_generator=generator, tools=[ExplodingTool()], max_iterations=3)

    result = await agent.run("query")

    assert isinstance(result["scratchpad"], list)
    assert "Error running tool" in result["scratchpad"][0]["observation"]
    assert "boom" in result["scratchpad"][0]["observation"]
    assert result["answer"] == "could not complete"


@pytest.mark.asyncio
async def test_agent_generator_failure_propagates():
    class RaisingGenerator:
        async def generate(self, prompt, request_id, max_tokens=None):
            raise RuntimeError("engine unavailable")

    agent = Agent(step_generator=RaisingGenerator(), tools=[_EchoTool()])

    with pytest.raises(RuntimeError, match="engine unavailable"):
        await agent.run("query")


def test_cleanup_marks_agent_as_stopped():
    agent = object.__new__(Agent)
    agent.stopped = False

    agent.cleanup()

    assert agent.stopped is True


def test_cleanup_is_idempotent():
    agent = object.__new__(Agent)
    agent.stopped = False

    agent.cleanup()
    agent.cleanup()

    assert agent.stopped is True

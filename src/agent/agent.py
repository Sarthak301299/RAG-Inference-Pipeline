import json
import logging
from typing import Protocol
from uuid import uuid4

from src.agent.tools import Tool
from src.pipeline.schemas import _FINAL_ANSWER_ACTION

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from typing import TypedDict


class AgentStepDict(TypedDict):
    thought: str
    action: str
    action_input: str
    observation: str


class AgentRunResult(TypedDict):
    answer: str
    iterations_used: int
    scratchpad: list[AgentStepDict]


class StepGenerator(Protocol):
    async def generate(
        self,
        prompt: str,
        request_id: str,
        max_tokens: int | None = None,
    ) -> str: ...


class MaxIterationsExceeded(Exception):
    """Raised when the maximum number of iterations is exceeded."""

    def __init__(self, message: str, scratchpad: list[AgentStepDict]) -> None:
        super().__init__(message)
        self.scratchpad = scratchpad


class Agent:
    def __init__(
        self,
        tools: list[Tool],
        step_generator: StepGenerator,
        max_iterations: int = 6,
    ) -> None:
        self.stopped = True
        if tools is None or len(tools) == 0:
            raise ValueError("Agent must be initialized with at least one tool.")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer.")
        self.tools_by_name: dict[str, Tool] = {tool.name: tool for tool in tools}
        self.step_generator = step_generator
        self.max_iterations = max_iterations
        self.stopped = False

    def _build_prompt(self, query: str, scratchpad: list[AgentStepDict]) -> str:
        tool_descriptions = "\n".join(
            f"- {name}: {tool.description}" for name, tool in self.tools_by_name.items()
        )
        prompt = (
            "You are an agent that answers questions by reasoning step by step and using tools when needed.\n\n"
            f"Available tools:\n{tool_descriptions}\n\n"
            "At each step, respond with a thought, an action to take (a tool name or 'final_answer'), and the action input. "
            "When you have enough information, use action 'final_answer' with the answer text as action_input. "
            "Ensure that all the requirements (units, numeric form, etc.) in the question are satisfied when using 'final_answer'."
            "Before choosing an action, check the steps already taken below. "
            "If a previous non-erroneous observation already contains the needed information,"
            "use that information directly instead of repeating the same action and input."
            "If the last step returned an error in the observation, read the error message and "
            "make appropriate changes instead of repeating the same action and input."
            "Repeating the same action with the same input will not give a different result.\n\n"
            f"Question: {query}\n\n"
        )
        for i, step in enumerate(scratchpad, start=1):
            prompt += (
                f"Step {i}:\n"
                f"Thought: {step['thought']}\n"
                f"Action: {step['action']}\n"
                f"Action Input: {step['action_input']}\n"
                f"Observation: {step['observation']}\n\n"
            )
        prompt += "Next step:\n"
        return prompt

    def _is_consecutive_repeat(
        self, action: str, action_input: str, scratchpad: list[AgentStepDict]
    ) -> bool:
        return (
            scratchpad != []
            and scratchpad[-1]["action"] == action
            and scratchpad[-1]["action_input"] == action_input
        )

    def _parse_step(self, raw_step: str) -> tuple[str, str, str] | None:
        try:
            parsed = json.loads(raw_step)
            thought = str(parsed["thought"])
            action = str(parsed["action"])
            action_input = str(parsed["action_input"])
        except Exception as e:  # noqa: BLE001
            logger.error(f"Got Exception {e} parsing agent step output.")
            return None
        return thought, action, action_input

    async def run(self, query: str) -> AgentRunResult:
        scratchpad: list[AgentStepDict] = []

        for iteration in range(self.max_iterations):
            prompt = self._build_prompt(query, scratchpad)
            try:
                raw_step = await self.step_generator.generate(
                    prompt=prompt,
                    request_id=str(uuid4()),
                )
            except Exception as e:
                logger.error(f"Got Exception {e} generating agent step {iteration}.")
                raise
            parsed = self._parse_step(raw_step)
            if parsed is None:
                scratchpad.append(
                    {
                        "thought": "(Unparsable agent step output)",
                        "action": "(none)",
                        "action_input": "(none)",
                        "observation": (
                            "Error: your last response could not be parsed. "
                            "Ensure that your response is valid JSON and follows the specified schema."
                        ),
                    }
                )
                continue

            thought, action, action_input = parsed

            if action == _FINAL_ANSWER_ACTION:
                return {
                    "answer": action_input,
                    "iterations_used": iteration + 1,
                    "scratchpad": scratchpad,
                }

            if self._is_consecutive_repeat(action, action_input, scratchpad):
                observation = (
                    f"Error: you already took the action '{action}' with "
                    f"input '{action_input}' in the last step."
                    "See the matching Observation above for its result. "
                    "Repeating it will not produce a new result. Use the "
                    "result already shown above to take a DIFFERENT next "
                    "step, or respond with 'final_answer' if you already "
                    "have enough information to answer the question."
                )
            else:
                tool = self.tools_by_name.get(action)
                if tool is None:
                    observation = f"Error: unknown tool '{action}'."
                else:
                    try:
                        observation = tool.run(action_input)
                    except Exception as e:  # noqa: BLE001
                        logger.error(
                            f"Got Exception {e} running tool '{action}' with input '{action_input}'."
                        )
                        observation = f"Error running tool '{action}': {e}"
            scratchpad.append(
                {
                    "thought": thought,
                    "action": action,
                    "action_input": action_input,
                    "observation": observation,
                }
            )
        raise MaxIterationsExceeded(
            f"Maximum iterations of {self.max_iterations} exceeded without reaching a final answer.",
            scratchpad=scratchpad,
        )

    def cleanup(self):
        self.stopped = True

from pydantic import BaseModel


class UserRequest(BaseModel):
    prompt: str


class UserResponse(BaseModel):
    status: int
    generated_response: str


class AgentStepRecord(BaseModel):
    thought: str
    action: str
    action_input: str
    observation: str


class AgentQueryResponse(BaseModel):
    answer: str
    iterations_used: int
    scratchpad: list[AgentStepRecord]

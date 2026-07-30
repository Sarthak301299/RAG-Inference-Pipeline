from pydantic import BaseModel


class UserRequest(BaseModel):
    prompt: str


class UserResponse(BaseModel):
    status: int
    generated_response: str

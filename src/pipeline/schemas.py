import json

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class RAGInputPromptFormat(BaseModel):
    user_query: str = Field(..., description="The original input from the user.")
    chunks: list[Document] = Field(
        ..., description="List of highly relevant text chunks retrieved."
    )
    system_query: str = (
        "You are a helpful and accurate AI assistant. "
        "Your job is to answer the user's query using ONLY the factual context provided below.\n"
        "Rules:\n"
        "1. Rely entirely on the provided context. Do not use any outside knowledge or assumptions.\n"
        "2. If the answer cannot be found in the context, respond with: 'I cannot find the answer in the provided documents.'\n"
        "3. Do not speculate, extrapolate, or make up facts.\n"
        "4. Cite the specific source or document title when presenting a fact if available.\n"
        "5. If conflicting information appears across sources, highlight the discrepancy explicitly.\n"
    )

    def get_formatted_prompt(self) -> str:
        prompt = self.system_query
        for idx, chunk in enumerate(self.chunks, start=1):
            content = getattr(chunk, "page_content", "")
            metadata = getattr(chunk, "metadata", {})

            prompt += f"--- START CHUNK {idx}, metadata: {json.dumps(metadata)} ---\n"
            prompt += f"{content}\n"
            prompt += f"--- END CHUNK {idx} ---\n\n"

        prompt += f"User Query:\n{self.user_query}"
        return prompt

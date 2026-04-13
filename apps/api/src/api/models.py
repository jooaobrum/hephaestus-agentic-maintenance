from typing import Union

from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    query: str


class RAGUsedContext(BaseModel):
    id: Union[int, str] = Field(description="ID of the intervention")
    machine: str = Field(description="Machine of the intervention")
    date_start: str = Field(description="Date of the intervention")
    summary: str = Field(description="Summary of the intervention")


class RAGResponse(BaseModel):
    answer: str
    references: list[RAGUsedContext] = []

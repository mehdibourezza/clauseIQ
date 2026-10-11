"""HTTP contracts of the API: what the frontend sends and receives."""

from datetime import datetime

from pydantic import BaseModel


class ClauseTypeOut(BaseModel):
    name: str
    description: str


class FindingOut(BaseModel):
    clause_type: str
    quote: str
    start_char: int | None
    end_char: int | None
    verified: bool


class ContractSummary(BaseModel):
    id: str
    filename: str
    status: str                  # processing | done | failed
    created_at: datetime
    completed_at: datetime | None
    clauses_found: int           # distinct clause types with at least one verified excerpt


class ContractDetail(ContractSummary):
    text: str
    error: str | None
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    findings: list[FindingOut]

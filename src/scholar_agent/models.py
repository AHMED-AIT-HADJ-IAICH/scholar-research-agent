from __future__ import annotations

from datetime import date
from pydantic import BaseModel, Field, field_validator, model_validator


class ResearchRequest(BaseModel):
    question: str = Field(min_length=15, max_length=1500)
    min_year: int = Field(ge=1900, le=date.today().year)
    paper_count: int = Field(ge=3, le=50)
    max_queries: int = Field(default=4, ge=2, le=6)
    language: str = Field(default="French", pattern="^(French|English)$")


class SearchPlan(BaseModel):
    concepts: list[str] = Field(min_length=2, max_length=8)
    synonyms: dict[str, list[str]] = Field(default_factory=dict)
    queries: list[str] = Field(min_length=2, max_length=6)
    exclusions: list[str] = Field(default_factory=list)
    rationale: str

    @field_validator("queries")
    @classmethod
    def valid_queries(cls, value: list[str]) -> list[str]:
        queries = list(dict.fromkeys(q.strip() for q in value if q.strip()))
        if len(queries) < 2:
            raise ValueError("At least two distinct queries are required")
        return queries


class ReviewDecision(BaseModel):
    approved: bool
    concepts: list[str] | None = None
    queries: list[str] | None = None
    exclusions: list[str] | None = None


class Paper(BaseModel):
    id: str
    title: str
    year: int
    doi: str | None = None
    url: str
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    abstract: str | None = None
    cited_by_count: int = 0
    is_retracted: bool = False
    is_oa: bool = False
    pdf_url: str | None = None
    queries: list[str] = Field(default_factory=list)
    score: float = 0.0
    score_breakdown: dict[str, float] = Field(default_factory=dict)


class Evidence(BaseModel):
    paper_id: str
    relevance: int = Field(ge=0, le=3)
    contribution: str
    method: str | None = None
    limitation: str | None = None
    relevant: bool = True


class Analysis(BaseModel):
    evidence: list[Evidence]


class Synthesis(BaseModel):
    overview: str
    themes: list[str]
    gaps: list[str]
    next_questions: list[str]


class Result(BaseModel):
    question: str
    plan: SearchPlan
    requested_count: int
    min_year: int
    papers: list[Paper]
    evidence: list[Evidence]
    synthesis: Synthesis | None
    warnings: list[str] = Field(default_factory=list)
    searched_queries: list[str] = Field(default_factory=list)
    candidate_count: int = 0

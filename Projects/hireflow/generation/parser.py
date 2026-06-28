from typing import List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class CandidateEvaluation(BaseModel):
    """
    Structured output from LLM evaluation.
    One instance per candidate.
    """
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="Match score between 0.0 and 1.0"
    )
    verdict: str = Field(
        description="One of: Hire / Maybe / Reject"
    )
    strengths: List[str] = Field(
        description="List of candidate strengths relevant to the JD"
    )
    gaps: List[str] = Field(
        description="List of skill or experience gaps"
    )
    reasoning: str = Field(
        description="2-3 sentence explanation of the verdict"
    )
    candidate_id: str = Field(
        default="",
        description="Candidate email / unique ID"
    )
    file_name: str = Field(
        default="",
        description="Source resume file name"
    )


def get_parser() -> PydanticOutputParser:
    """
    Returns a LangChain PydanticOutputParser for CandidateEvaluation.
    Automatically generates format_instructions for the prompt.
    Handles JSON parsing + validation internally.
    """
    return PydanticOutputParser(pydantic_object=CandidateEvaluation)
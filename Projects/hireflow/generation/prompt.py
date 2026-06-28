from langchain_core.prompts import ChatPromptTemplate
from generation.parser import get_parser

SYSTEM_PROMPT = """You are an expert technical recruiter with deep knowledge of evaluating candidates.

Your job is to evaluate a candidate's resume against a job description and provide a structured assessment.

Scoring guide:
- 0.8 to 1.0 → Hire   (strong match, meets most requirements)
- 0.5 to 0.7 → Maybe  (partial match, worth interviewing)
- 0.0 to 0.4 → Reject (poor match, missing critical requirements)

{format_instructions}"""

HUMAN_PROMPT = """Job Description:
{jd_text}

---

Candidate Resume:
{resume_text}

---

Evaluate this candidate against the job description."""


def get_eval_prompt() -> ChatPromptTemplate:
    """
    Returns evaluation prompt with format_instructions
    automatically injected from PydanticOutputParser.

    Variables needed at runtime: jd_text, resume_text
    format_instructions is injected via partial_variables.
    """
    parser = get_parser()

    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ]).partial(
        format_instructions=parser.get_format_instructions()
    )
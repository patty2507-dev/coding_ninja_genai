import json
from typing import List, AsyncGenerator

from generation.prompt import get_eval_prompt
from generation.llm import get_llm
from generation.parser import get_parser, CandidateEvaluation
from retrieval.chain import retrieve


async def evaluate_candidate(
    jd_text: str,
    resume_text: str,
    candidate_id: str = "",
    file_name: str = "",
) -> CandidateEvaluation:
    """
    Evaluate a single candidate against the JD.
    Uses LangChain LCEL: prompt | llm | parser
    Parser handles JSON extraction + Pydantic validation automatically.
    """
    prompt = get_eval_prompt()
    llm = get_llm()
    parser = get_parser()

    # LCEL chain — LangChain pipe operator
    chain = prompt | llm | parser

    evaluation: CandidateEvaluation = await chain.ainvoke({
        "jd_text": jd_text,
        "resume_text": resume_text,
    })

    # Attach metadata not in LLM output
    evaluation.candidate_id = candidate_id
    evaluation.file_name = file_name

    return evaluation


async def evaluate_candidates_stream(
    jd_text: str,
    top_k: int = 5,
) -> AsyncGenerator[str, None]:
    """
    Full pipeline — streaming version:
        JD text
            → retrieve top candidates (Pinecone + reranker)
            → evaluate each with GPT-4o-mini via LCEL chain
            → yield SSE-ready JSON events

    Note: PydanticOutputParser does not support astream()
    so we stream tokens manually then parse the complete response.
    This gives us SSE status events while keeping structured output.
    """
    prompt = get_eval_prompt()
    llm = get_llm()
    parser = get_parser()

    # Step 1 — Retrieve candidates
    candidates = retrieve(jd_text=jd_text, top_k=top_k)

    if not candidates:
        yield json.dumps({"error": "No candidates found in database"})
        return

    # Step 2 — Evaluate each candidate
    for i, candidate in enumerate(candidates):
        resume_text = candidate.get("text", "")
        candidate_id = candidate.get("candidate_id", "")
        file_name = candidate.get("file_name", "")

        # Status event — UI shows "Evaluating candidate X..."
        yield json.dumps({
            "event": "evaluating",
            "rank": i + 1,
            "candidate_id": candidate_id,
            "file_name": file_name,
            "total": len(candidates),
        })

        try:
            # Stream tokens from LLM, collect full response
            # PydanticOutputParser needs full text to parse
            formatted = await prompt.ainvoke({
                "jd_text": jd_text,
                "resume_text": resume_text,
            })

            full_response = ""
            async for chunk in llm.astream(formatted):
                token = chunk.content
                if token:
                    full_response += token

            # Parse via LangChain PydanticOutputParser
            evaluation: CandidateEvaluation = parser.parse(full_response)
            evaluation.candidate_id = candidate_id
            evaluation.file_name = file_name

            yield json.dumps({
                "event": "result",
                "rank": i + 1,
                "data": evaluation.model_dump(),
            })

        except Exception as e:
            yield json.dumps({
                "event": "error",
                "rank": i + 1,
                "candidate_id": candidate_id,
                "error": str(e),
            })
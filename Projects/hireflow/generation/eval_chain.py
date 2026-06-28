import json
from typing import List

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
    llm    = get_llm()
    parser = get_parser()

    # LCEL chain — LangChain pipe operator
    chain = prompt | llm | parser

    evaluation: CandidateEvaluation = await chain.ainvoke({
        "jd_text":     jd_text,
        "resume_text": resume_text,
    })

    # Attach metadata not in LLM output
    evaluation.candidate_id = candidate_id
    evaluation.file_name    = file_name

    return evaluation


# ─────────────────────────────────────────────
# REPLACED: evaluate_candidates_stream (old)
#   - AsyncGenerator  → removed
#   - yield           → removed
#   - llm.astream()   → removed
#   - SSE events      → removed
#
# NEW: evaluate_all (simple)
#   - Normal async function
#   - Returns a plain list
#   - chain.ainvoke() — one call per candidate
# ─────────────────────────────────────────────

async def evaluate_all(
    jd_text: str,
    top_k: int = 5,
) -> List[dict]:
    """
    Full pipeline — simple version:
        JD text
            → retrieve top candidates (Pinecone + reranker)
            → evaluate each with GPT-4o-mini via LCEL chain
            → return list of results
    """
    prompt = get_eval_prompt()
    llm    = get_llm()
    parser = get_parser()

    chain = prompt | llm | parser

    # Step 1 — Retrieve top-K candidates from Pinecone
    candidates = retrieve(jd_text=jd_text, top_k=top_k)

    if not candidates:
        return []

    results = []

    # Step 2 — Evaluate each candidate one by one
    for i, candidate in enumerate(candidates):
        resume_text  = candidate.get("text", "")
        candidate_id = candidate.get("candidate_id", "")
        file_name    = candidate.get("file_name", "")

        print(f"[{i+1}/{len(candidates)}] Evaluating: {file_name}")

        try:
            evaluation: CandidateEvaluation = await chain.ainvoke({
                "jd_text":     jd_text,
                "resume_text": resume_text,
            })

            evaluation.candidate_id = candidate_id
            evaluation.file_name    = file_name

            results.append({
                "rank": i + 1,
                "data": evaluation.model_dump(),
            })

        except Exception as e:
            results.append({
                "rank":         i + 1,
                "error":        str(e),
                "candidate_id": candidate_id,
            })

    return results

# jd="""
# We are looking for a Senior Accountant to join our finance team.
# Requirements:
# - 3+ years of accounting experience
# - Strong knowledge of financial reporting and general ledger
# - Experience with month-end closing and bank reconciliation
# - Proficiency in accounting software (SAP, NetSuite, or similar)
# - CPA certification preferred
# - Experience with variance analysis and budget planning
# Responsibilities:
# - Prepare and review financial statements
# - Manage accounts receivable and payable
# - Perform month-end and year-end close procedures
# - Ensure compliance with accounting standards


# """

# output = evaluate_all(jd, top_k=4)
# print(output)

# Reranker model loaded.
# Reranked 2 → top 2 candidates
#   1. Andrew_Green_Resume_27.pdf          rerank_score: -0.0281
#   2. Angela_Lewis_Resume_09.pdf          rerank_score: -0.6854
# [{'id': 'andrew.green@email.com_1', 'score': 0.5335, 'candidate_id': 'andrew.green@email.com', 'text': 'PROFESSIONAL EXPERIENCE\nAccounting Intern | Premier Financial Advisors | Summer 2023\n\x7f Assisted with month-end closing procedures and financial reporting\n\x7f Processed accounts payable invoices and vendor payments\n\x7f Performed bank reconciliations and account analysis\n\x7f Maintained general ledger accounts and prepared journal entries\n\x7f Created Excel spreadsheets for financial analysis and reporting\nEDUCATION\nMaster of Accountancy\nColumbia University | 2022\nGPA: 3.3/4.0\nCERTIFICATIONS\n\x7f PMP\n\x7f CGA\nACHIEVEMENTS\n{generate_achievements(experience_years)}\nREFERENCES\nAvailable upon request', 'file_name': 'Andrew_Green_Resume_27.pdf', 'chunk_index': 1, 'chunk_total': 2, 'rerank_score': -0.0281}, {'id': 'angela.lewis@email.com_1', 'score': 0.5492, 'candidate_id': 'angela.lewis@email.com', 'text': 'PROFESSIONAL EXPERIENCE\nFinancial Planning Analyst | Professional Accounting Partners | 2021 - Present\n\x7f Led financial reporting and analysis for $50M+ revenue company\n\x7f Managed month-end closing process and prepared financial statements\n\x7f Collaborated with cross-functional teams on strategic initiatives\n\x7f Implemented process improvements resulting in 20% efficiency gains\n\x7f Mentored junior staff and provided training on new procedures\nFixed Asset Accountant | Premier Financial Advisors | 2020 - 2021\n\x7f Prepared monthly financial reports and variance analysis\n\x7f Processed journal entries and maintained general ledger\n\x7f Assisted with budget planning and forecasting processes\n\x7f Reconciled bank statements and credit card accounts\n\x7f Supported audit preparation and documentation\nAccounting Manager | Excellence in Finance | 2019 - 2021\n\x7f Prepared monthly financial reports and variance analysis\n\x7f Processed journal entries and maintained general ledger\n\x7f Assisted with budget planning and forecast', 'file_name': 'Angela_Lewis_Resume_09.pdf', 'chunk_index': 1, 'chunk_total': 3, 'rerank_score': -0.6854}]
# <coroutine object evaluate_all at 0x1603d9380>
# /Users/rahultiwari/Documents/02_Freelancing/coding_ninja_fresh/hire-env/bin/python: Error while
# finding module specification for 'generation.eval_chain.py' (ModuleNotFoundError: __path__ attribute not found on
#                                                               'generation.eval_chain' while trying to find 'generation.eval_chain.py'). Try using 'generation.eval_chain' instead of 'generation.eval_chain.py' as the module name.
# <sys>:0: RuntimeWarning: coroutine 'evaluate_all' was never awaited
from typing import List, Dict, Any
from pinecone import Pinecone
from config import settings
_index = None

def _get_index():
    global _index
    if _index is None:
        pc = Pinecone(api_key=settings.pinecone_api_key)
        _index = pc.Index(settings.pinecone_index_name)
    return _index


def retrieve_candidates(
    query_embedding: List[float],
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Query Pinecone with a dense embedding vector.
    Returns top_k most similar resume chunks.

    We fetch top_k=20 here (more than needed)
    so the reranker has enough candidates to work with.
    Final top-k is applied after reranking.

    Returns list of dicts:
        {
            id, score, candidate_id,
            text, file_name, chunk_index
        }
    """
    index = _get_index()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )
   

    candidates = []
    seen_candidate_ids = set()

    for match in results.matches:
        meta = match.metadata or {}
        candidate_id = meta.get("candidate_id", match.id)

        # Deduplicate — keep only best chunk per candidate
        if candidate_id in seen_candidate_ids:
            continue
        seen_candidate_ids.add(candidate_id)

        candidates.append({
            "id": match.id,
            "score": round(match.score, 4),
            "candidate_id": candidate_id,
            "text": meta.get("text", ""),
            "file_name": meta.get("file_name", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "chunk_total": meta.get("chunk_total", 0),
        })

    print(f"Retrieved {len(candidates)} unique candidate(s) from Pinecone")
    return candidates

jd = """
We are looking for a Senior Accountant to join our finance team.
Requirements:
- 3+ years of accounting experience
- Strong knowledge of financial reporting and general ledger
- Experience with month-end closing and bank reconciliation
- Proficiency in accounting software (SAP, NetSuite, or similar)
- CPA certification preferred
- Experience with variance analysis and budget planning
Responsibilities:
- Prepare and review financial statements
- Manage accounts receivable and payable
- Perform month-end and year-end close procedures
- Ensure compliance with accounting standards


"""

# from retrieval.query_embedder import embed_query
# query_embedded =embed_query(jd)
# result = retrieve_candidates(query_embedded,top_k=4)
# print(result)


# [{'id': 'angela.lewis@email.com_1', 'score': 0.5492, 
#   'candidate_id': 'angela.lewis@email.com',
#     'text': 'PROFESSIONAL EXPERIENCE\nFinancial Planning Analyst | Professional Accounting Partners | 2021 - Present\n\x7f Led financial reporting and analysis for $50M+ revenue company\n\x7f '
#     'Managed month-end closing process and prepared financial statements\n\x7f Collaborated with cross-functional teams on strategic initiatives\n\x7f Implemented process improvements resulting in 20% efficiency gains\n\x7f Mentored junior staff and provided training on new procedures\nFixed Asset Accountant | Premier Financial Advisors | 2020 - 2021\n\x7f Prepared monthly financial '
#     'reports and variance analysis\n\x7f Processed journal entries and maintained'
#     ' general ledger\n\x7f Assisted with budget planning and forecasting processes\n\x7f '
#     'Reconciled bank statements and credit card accounts\n\x7f Supported audit preparation a'
#     'nd documentation\nAccounting Manager | Excellence in Finance | 2019 - 2021\n\x7f'
#     ' Prepared monthly financial reports and variance analysis\n\x7f '
#     'Processed journal entries and maintained general ledger\n\x7f '
#     'Assisted with budget planning and forecast', 'file_name':
#       'Angela_Lewis_Resume_09.pdf', 'chunk_index': 1, 'chunk_total': 3},
#         {'id': 'andrew.green@email.com_1', 'score': 0.5334, 'candidate_id': 'andrew.green@email.com', 'text': 'PROFESSIONAL EXPERIENCE\nAccounting Intern | Premier Financial Advisors | Summer 2023\n\x7f Assisted with month-end closing procedures and financial reporting\n\x7f Processed accounts payable invoices and vendor payments\n\x7f Performed bank reconciliations and account analysis\n\x7f Maintained general ledger accounts and prepared journal entries\n\x7f Created Excel spreadsheets for financial analysis and reporting\nEDUCATION\nMaster of Accountancy\nColumbia University | 2022\nGPA: 3.3/4.0\nCERTIFICATIONS\n\x7f PMP\n\x7f CGA\nACHIEVEMENTS\n{generate_achievements(experience_years)}\nREFERENCES\nAvailable upon request', 'file_name': 'Andrew_Green_Resume_27.pdf', 'chunk_index': 1, 'chunk_total': 2}]



# QueryResponse(matches=[ScoredVector(id='angela.lewis@email.com_1', score=0.549211, values=[],
#                                      metadata={'candidate_id': 'angela.lewis@email.com', 
#                                                'chunk_index': 1, 'chunk_total': 3, 
#                                                'file_name': 'Angela_Lewis_Resume_09.pdf', 
#                                                'page_label': '2', 
#                                                'text': 'PROFESSIONAL EXPERIENCE\nFinancial Planning Analyst | Professional Accounting Partners | 2021 - Present\n\x7f Led financial reporting and analysis for $50M+ revenue company\n\x7f Managed month-end closing process and prepared financial statements\n\x7f Collaborated with cross-functional teams on strategic initiatives\n\x7f '
#                                                'Implemented process improvements resulting in 20% efficiency gains\n\x7f '
#                                                'Mentored junior staff and provided training on new procedures\nFixed Asset Accountant'
#                                                ' | Premier Financial Advisors | 2020 - 2021\n\x7f Prepared monthly financial reports'
#                                                ' and variance analysis\n\x7f Processed journal entries and maintained general ledger\n\x7f Assisted with budget planning and forecasting processes\n\x7f Reconciled bank statements and credit card accounts\n\x7f Supported audit preparation and documentation\nAccounting Manager | Excellence in Finance | 2019 - 2021\n\x7f Prepared monthly financial reports and variance analysis\n\x7f Processed journal entries and maintained general ledger\n\x7f Assisted with budget planning and forecast'}), ScoredVector(id='andrew.green@email.com_1', score=0.533403516, values=[], metadata={'candidate_id': 'andrew.green@email.com', 'chunk_index': 1, 'chunk_total': 2, 'file_name': 'Andrew_Green_Resume_27.pdf', 'page_label': '2', 'text': 'PROFESSIONAL EXPERIENCE\nAccounting Intern | Premier Financial Advisors | Summer 2023\n\x7f Assisted with month-end closing procedures and financial reporting\n\x7f Processed accounts payable invoices and vendor payments\n\x7f Performed bank reconciliations and account analysis\n\x7f Maintained general ledger accounts and prepared journal entries\n\x7f Created Excel spreadsheets for financial analysis and reporting\nEDUCATION\nMaster of Accountancy\nColumbia University | 2022\nGPA: 3.3/4.0\nCERTIFICATIONS\n\x7f PMP\n\x7f CGA\nACHIEVEMENTS\n{generate_achievements(experience_years)}\nREFERENCES\nAvailable upon request'}), ScoredVector(id='andrew.green@email.com_0', score=0.458282381, values=[], metadata={'candidate_id': 'andrew.green@email.com', 'chunk_index': 0, 'chunk_total': 2, 'file_name': 'Andrew_Green_Resume_27.pdf', 'page_label': '1', 'text': 'ANDREW GREEN\nRecent Graduate\nContact Information:\nEmail: andrew.green@email.com\nPhone: (714) 826-3519\nLocation: Long Beach, CA\nLinkedIn: linkedin.com/in/andrewgreen\nPROFESSIONAL SUMMARY\nRecent accounting graduate with strong academic foundation and internship experience. Eager to\nbegin career in accounting with focus on financial reporting and analysis. Detail-oriented, organized,\nand committed to professional growth and development.\nTECHNICAL SKILLS\n\x7f Tax Preparation\n\x7f Financial Reporting\n\x7f Fixed Asset Management\n\x7f Payroll Processing\n\x7f Accounts Payable\n\x7f Year-End Closing\n\x7f Bank Reconciliation\n\x7f Variance Analysis\nSOFTWARE PROFICIENCY\n\x7f PowerPoint\n\x7f Microsoft Excel\n\x7f Adobe Acrobat\n\x7f SQL\n\x7f Outlook\n\x7f Sage'}), ScoredVector(id='angela.lewis@email.com_2', score=0.456184238, values=[], metadata={'candidate_id': 'angela.lewis@email.com', 'chunk_index': 2, 'chunk_total': 3, 'file_name': 'Angela_Lewis_Resume_09.pdf', 'page_label': '3', 'text': 'EDUCATION\nBachelor of Science in Finance\nPrinceton University | 2019\nGPA: 3.5/4.0\nCERTIFICATIONS\n\x7f CIA\n\x7f QuickBooks ProAdvisor\n\x7f CMA\nACHIEVEMENTS\n{generate_achievements(experience_years)}\nREFERENCES\nAvailable upon request'})], namespace='', usage=Usage(read_units=1, write_units=None), response_info=ResponseInfo(raw_headers={'date': 'Sun, 28 Jun 2026 07:19:10 GMT', 'content-type': 'application/json', 'content-length': '3710', 'connection': 'keep-alive', 'x-pinecone-max-indexed-lsn': '2', 'x-pinecone-request-latency-ms': '297', 'x-envoy-upstream-service-time': '111', 'x-pinecone-response-duration-ms': '299', 'grpc-status': '0', 'server': 'envoy'}))
# (hire-env) Rahuls-MacBook-Air:hireflow rahultiwari$ 
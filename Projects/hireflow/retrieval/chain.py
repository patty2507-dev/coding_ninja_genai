from typing import List, Dict, Any

from retrieval.query_embedder import embed_query
from retrieval.retriever import retrieve_candidates
from retrieval.reranker import rerank_candidates
from config import settings


def retrieve(
    jd_text: str,
    top_k: int = None):
    # step 1 : embedded the JD
    query_vector = embed_query(jd_text)
    # step 2 : Pinecone search ( top 20 candiated based on cosine similirty)
    candidates = retrieve_candidates(query_vector)

    if not candidates:
        print("No candidates found")
        return []

    # step 3 : rerank the candidates
    top_candidates = rerank_candidates(jd_text, candidates, top_k=top_k)

    return top_candidates
    
    
# from retrieval.chain import retrieve
# jd = """
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

# test = retrieve(jd, top_k=4)
# print(test)
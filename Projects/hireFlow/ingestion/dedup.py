from typing import Optional

from pinecone import Pinecone
from config import settings

_index = None


def _get_index():
    global _index
    if _index is None:
        pc = Pinecone(api_key=settings.pinecone_api_key)
        _index = pc.Index(settings.pinecone_index_name)
    return _index


def candidate_exists(candidate_id: str) -> bool:
    """
    Check if a candidate_id (email) already exists in Pinecone.
    Uses fetch by ID prefix — looks for any vector with this candidate_id.
    Returns True if found, False if new candidate.
    """
    index = _get_index()

    try:
        # List vectors with this candidate_id prefix
        result = index.list(prefix=candidate_id, limit=1)
        ids = list(result)
        return len(ids) > 0
    except Exception:
        return False


def delete_existing_candidate(candidate_id: str) -> int:
    """
    Delete all vectors for an existing candidate.
    Called before re-upserting updated resume.
    Returns count of deleted vectors.
    """
    index = _get_index()

    try:
        # Collect all vector IDs for this candidate
        all_ids = []
        for id_batch in index.list(prefix=candidate_id):
            if isinstance(id_batch, list):
                all_ids.extend(id_batch)
            else:
                all_ids.append(id_batch)

        if all_ids:
            index.delete(ids=all_ids)
            print(f"  Deleted {len(all_ids)} existing vector(s) for: {candidate_id}")
            return len(all_ids)

        return 0

    except Exception as e:
        print(f"  Warning: Could not delete existing vectors for {candidate_id}: {e}")
        return 0


def handle_deduplication(candidate_id: str) -> bool:
    """
    Main deduplication entry point.
    - If candidate exists → delete old vectors, return True (was existing)
    - If new candidate   → do nothing,          return False (is new)
    """
    if candidate_exists(candidate_id):
        print(f"  Existing candidate found: {candidate_id} — overwriting...")
        delete_existing_candidate(candidate_id)
        return True

    print(f"  New candidate: {candidate_id}")
    return False
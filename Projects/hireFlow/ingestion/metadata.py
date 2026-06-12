import re
from typing import Optional

from openai import OpenAI
from config import settings

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def extract_email_regex(text: str) -> Optional[str]:
    """
    Primary method — fast regex email extraction.
    Returns first email found or None.
    """
    pattern = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[0].lower().strip()
    return None


def extract_email_llm(text: str) -> Optional[str]:
    """
    Fallback — use GPT-4o-mini to find email when regex fails.
    Only called when regex returns None.
    Keeps prompt minimal to reduce token cost.
    """
    # Only send first 1000 chars — email is usually at the top
    snippet = text[:1000]

    client = _get_client()
    response = client.chat.completions.create(
        model=settings.openai_llm_model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract the email address from this resume text. "
                    "Reply with ONLY the email address, nothing else. "
                    "If no email is found, reply with: NONE\n\n"
                    f"{snippet}"
                ),
            }
        ],
        max_tokens=20,
        temperature=0,
    )

    result = response.choices[0].message.content.strip()

    if result.upper() == "NONE" or "@" not in result:
        return None

    return result.lower().strip()


def extract_email(text: str) -> Optional[str]:
    """
    Main entry point.
    1. Try regex first (fast, free)
    2. Fallback to LLM if regex fails
    """
    email = extract_email_regex(text)
    if email:
        return email

    print("Regex found no email — trying LLM fallback...")
    email = extract_email_llm(text)
    return email


def build_candidate_id(text: str, file_name: str) -> str:
    """
    Build a unique candidate_id from:
    1. Email (preferred)
    2. Hash of filename as last resort
    Returns candidate_id string.
    """
    email = extract_email(text)

    if email:
        print(f"  candidate_id: {email}")
        return email

    # Last resort — hash of filename
    import hashlib
    fallback_id = hashlib.md5(file_name.encode()).hexdigest()[:16]
    print(f"  No email found. Using file hash as candidate_id: {fallback_id}")
    return fallback_id
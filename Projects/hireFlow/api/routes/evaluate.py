import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from generation.eval_chain import evaluate_candidates_stream
from config import settings

router = APIRouter()


class EvaluateRequest(BaseModel):
    jd_text: str
    top_k: int = None


async def sse_generator(jd_text: str, top_k: int):
    """
    Async generator that yields SSE-formatted events.
    SSE format: data: <json>\n\n
    """
    async for event_json in evaluate_candidates_stream(
        jd_text=jd_text,
        top_k=top_k,
    ):
        # SSE format — each event prefixed with "data: "
        yield f"data: {event_json}\n\n"

    # Send done event so client knows stream is complete
    yield f"data: {json.dumps({'event': 'done'})}\n\n"


@router.post("/evaluate")
async def evaluate(request: EvaluateRequest):
    """
    POST /evaluate

    Body:
        {
            "jd_text": "We are looking for a Senior Accountant...",
            "top_k": 5
        }

    Returns: Server-Sent Events (SSE) stream
    Each event is a JSON object:
        - {"event": "evaluating", "rank": 1, "candidate_id": "...", ...}
        - {"event": "result",     "rank": 1, "data": {...}}
        - {"event": "error",      "rank": 1, "error": "..."}
        - {"event": "done"}
    """
    top_k = request.top_k or settings.top_k

    return StreamingResponse(
        sse_generator(
            jd_text=request.jd_text,
            top_k=top_k,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
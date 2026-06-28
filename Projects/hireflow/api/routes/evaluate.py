from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from generation.eval_chain import evaluate_all   # changed
from config import settings

router = APIRouter()


class EvaluateRequest(BaseModel):
    jd_text: str
    top_k: int = None


# sse_generator — REMOVED
# StreamingResponse — REMOVED


@router.post("/evaluate")
async def evaluate(request: EvaluateRequest):
    top_k = request.top_k or settings.top_k

    results = await evaluate_all(
        jd_text=request.jd_text,
        top_k=top_k,
    )

    if not results:
        return JSONResponse(
            status_code=404,
            content={"message": "No candidates found in database."}
        )

    return JSONResponse(content={"total": len(results), "results": results})
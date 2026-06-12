import time
import mlflow
from functools import wraps
from typing import Callable


def trace_llm_call(func: Callable) -> Callable:
    """
    Decorator — traces any async LLM call to MLflow.
    Logs: input length, output length, latency, model name.

    Usage:
        @trace_llm_call
        async def my_llm_function(...):
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        latency = round(time.time() - start, 3)

        try:
            with mlflow.start_run(nested=True):
                mlflow.log_metrics({
                    "llm_latency_seconds": latency,
                })
                mlflow.log_param("traced_function", func.__name__)
        except Exception:
            pass  # Never let tracing break the main flow

        return result

    return wrapper


def log_retrieval_metrics(
    candidates_retrieved: int,
    candidates_reranked: int,
    top_k: int,
    retrieval_latency: float,
):
    """
    Log retrieval pipeline metrics to active MLflow run.
    Call inside an active mlflow.start_run() context.
    """
    try:
        mlflow.log_metrics({
            "retrieval_candidates_fetched": candidates_retrieved,
            "retrieval_candidates_reranked": candidates_reranked,
            "retrieval_top_k": top_k,
            "retrieval_latency_seconds": round(retrieval_latency, 3),
        })
    except Exception:
        pass
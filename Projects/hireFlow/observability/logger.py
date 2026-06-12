import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import mlflow

from config import settings


def log_eval_run(
    jd_text: str,
    results: List[Dict[str, Any]],
    ragas_scores: Dict[str, Optional[float]],
    top_k: int,
    prompt_template: str = "",
) -> str:
    """
    Log a complete HireFlow evaluation run to MLflow.
    Uses sqlite:///mlflow.db backend.

    Logs:
        params   — top_k, model, embedding model
        metrics  — RAGAS scores + eval metrics
        artifacts — output JSON, prompt version
    """
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # ── Params ────────────────────────────────────────────────
        mlflow.log_params({
            "top_k": top_k,
            "llm_model": settings.openai_llm_model,
            "embedding_model": settings.openai_embedding_model,
            "total_candidates": len(results),
        })

        # ── RAGAS metrics — only log non-None values ──────────────
        ragas_logged = 0
        for metric_name, score in ragas_scores.items():
            if score is not None:
                mlflow.log_metric(f"ragas_{metric_name}", round(score, 4))
                ragas_logged += 1

        if ragas_logged == 0:
            mlflow.log_metric("ragas_available", 0)
        else:
            mlflow.log_metric("ragas_available", 1)

        # ── Candidate eval metrics ────────────────────────────────
        if results:
            avg_score    = sum(r.get("confidence_score", 0) for r in results) / len(results)
            hire_count   = sum(1 for r in results if r.get("verdict") == "Hire")
            maybe_count  = sum(1 for r in results if r.get("verdict") == "Maybe")
            reject_count = sum(1 for r in results if r.get("verdict") == "Reject")

            mlflow.log_metrics({
                "avg_confidence_score": round(avg_score, 4),
                "hire_count":           hire_count,
                "maybe_count":          maybe_count,
                "reject_count":         reject_count,
            })

        # ── Artifacts ─────────────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Evaluation output JSON
            output = {
                "jd_preview": jd_text[:300],
                "results": results,
                "ragas_scores": ragas_scores,
            }
            output_file = tmp / "eval_output.json"
            output_file.write_text(json.dumps(output, indent=2))
            mlflow.log_artifact(str(output_file), artifact_path="outputs")

            # Prompt template
            if prompt_template:
                prompt_file = tmp / "prompt_template.txt"
                prompt_file.write_text(prompt_template)
                mlflow.log_artifact(str(prompt_file), artifact_path="prompts")

        print(f"MLflow run logged: {run_id}")
        print(f"  Params: top_k={top_k}, model={settings.openai_llm_model}")
        print(f"  Metrics: {len(results)} candidates, {ragas_logged} RAGAS scores")
        print(f"  View at: http://localhost:5000")
        return run_id


def log_recruiter_feedback(
    run_id: str,
    candidate_id: str,
    action: str,
):
    """
    Log recruiter feedback against an existing MLflow run.
    action: shortlist | reject | thumbs_up | thumbs_down
    """
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param(f"feedback_{candidate_id}", action)
    print(f"Feedback logged: {candidate_id} → {action}")
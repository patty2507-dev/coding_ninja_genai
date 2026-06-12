from typing import List, Dict, Any
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset


def build_ragas_dataset(
    jd_text: str,
    candidates: List[Dict[str, Any]],
    evaluations: List[Dict[str, Any]],
) -> EvaluationDataset:
    """
    Build RAGAS EvaluationDataset from HireFlow eval results.
    Uses ragas.dataset_schema — correct for ragas 0.2.x

    Fields per SingleTurnSample:
        user_input         — JD (what we searched for)
        retrieved_contexts — resume chunks retrieved
        response           — LLM evaluation output
        reference          — expected answer (verdict as proxy)
    """
    samples = []

    for candidate, evaluation in zip(candidates, evaluations):
        verdict   = evaluation.get("verdict", "Reject")
        score     = evaluation.get("confidence_score", 0)
        reasoning = evaluation.get("reasoning", "")

        response = (
            f"Verdict: {verdict}. "
            f"Score: {score}. "
            f"Reasoning: {reasoning}"
        )

        sample = SingleTurnSample(
            user_input=jd_text,
            retrieved_contexts=[candidate.get("text", "")],
            response=response,
            reference=f"Candidate verdict should be {verdict}",
        )
        samples.append(sample)

    dataset = EvaluationDataset(samples=samples)
    print(f"Built RAGAS dataset with {len(samples)} sample(s)")
    return dataset
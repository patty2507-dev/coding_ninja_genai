from typing import Dict, Optional

from ragas.dataset_schema import EvaluationDataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextRecall,
    LLMContextPrecisionWithReference,
)
from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import settings


def run_ragas_evaluation(dataset: EvaluationDataset) -> Dict[str, Optional[float]]:
    """
    Run RAGAS evaluation using ragas 0.2.x API.
    Dynamically reads column names from result DataFrame.
    """
    print(f"\nRunning RAGAS evaluation ({len(dataset.samples)} sample(s))...")

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.openai_llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
    )

    judge_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
    )

    metrics = [
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        Faithfulness(),
        ResponseRelevancy(),
    ]

    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
        )

        # Convert to DataFrame and inspect actual column names
        result_df = result.to_pandas()
        print(f"\nRAGAS result columns: {list(result_df.columns)}")

        # Map expected names to actual column names dynamically
        scores = {}
        col_map = {
            "context_precision": ["LLMContextPrecisionWithReference", "context_precision"],
            "context_recall":    ["LLMContextRecall", "context_recall"],
            "faithfulness":      ["Faithfulness", "faithfulness"],
            "answer_relevancy":  ["ResponseRelevancy", "answer_relevancy", "response_relevancy"],
        }

        for metric_name, possible_cols in col_map.items():
            score = None
            for col in possible_cols:
                if col in result_df.columns:
                    score = round(float(result_df[col].mean()), 4)
                    break
            scores[metric_name] = score

        print("\nRAGAS scores:")
        for metric, score in scores.items():
            if score is not None:
                bar = "█" * int(score * 20)
                print(f"  {metric:<25} {score:.4f}  {bar}")
            else:
                print(f"  {metric:<25} None")

        return scores

    except Exception as e:
        print(f"RAGAS evaluation error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "context_precision": None,
            "context_recall":    None,
            "faithfulness":      None,
            "answer_relevancy":  None,
        }
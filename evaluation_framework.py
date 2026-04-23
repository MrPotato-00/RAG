from deepeval.models import LiteLLMModel
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
import json
import os
import time


class DeepEvalRAGEvaluator:
    def __init__(self, rag_pipeline, api_key: str, model_name: str):
        self.llm = LiteLLMModel(
            model=model_name,
            api_key=api_key
        )
        self.rag = rag_pipeline
        self.faithfulness        = FaithfulnessMetric(model=self.llm)
        self.answer_relevancy    = AnswerRelevancyMetric(model=self.llm)
        self.contextual_relevancy = ContextualRelevancyMetric(model=self.llm)

    def evaluate_dataset(self, dataset: list[dict], free_api=True) -> list[dict]:
        all_results = []
        for sample in dataset:
            query        = sample["query"]
            ground_truth = sample["grounded_answer"]
            domain       = sample.get("domain", "unknown")

            rag_out  = self.rag.get_output(query, reranked=True, dense_top_k= 10, sparse_top_k=10, reranked_topk= 3)
            answer   = rag_out["message"]
            context  = rag_out["context"]
            citations = rag_out["citations"]

            if rag_out["ood"] or not context:
                all_results.append({
                    "query": query, "domain": domain,
                    "grounded_answer": ground_truth,
                    "generated_answer": answer,
                    "citations": citations,
                    "metrics": {"faithfulness": None,
                                "faithfulness_reason": None,
                                "answer_relevancy": None,
                                "answer_relevancy_reason": None,
                                "contextual_relevancy": None,
                                "contextual_relevancy_reason": None},
                    "ood": True
                })
                continue

            faithfulness_test_case = LLMTestCase(
                input=query,
                actual_output=answer,
                retrieval_context=context,
            )
            answer_relevancy_test_case = LLMTestCase(
                input=query,
                actual_output=answer
            )
            contextual_relevancy_test_case = LLMTestCase(
                input=query,
                actual_output=answer,
                retrieval_context=context
            )

            self.faithfulness.measure(faithfulness_test_case)
            self.answer_relevancy.measure(answer_relevancy_test_case)
            self.contextual_relevancy.measure(contextual_relevancy_test_case)

            all_results.append({
                "query":            query,
                "domain":           domain,
                "grounded_answer": ground_truth,
                "generated_answer": answer,
                "metrics": {
                    "faithfulness":         self.faithfulness.score,
                    "faithfulness_reason":  self.faithfulness.reason,
                    "answer_relevancy":     self.answer_relevancy.score,
                    "answer_relevancy_reason": self.answer_relevancy.reason,
                    "contextual_relevancy": self.contextual_relevancy.score,
                    "contextual_relevancy_reason": self.contextual_relevancy.reason,
                },
                "ood": False
            })

            if free_api:
                print("Sleeping for 10 seconds")
                time.sleep(10)

        return all_results


def create_evaluator(rag_pipeline, config: dict) -> DeepEvalRAGEvaluator:
    eval_config = config.get("eval_config", {})
    api_key= os.getenv("EVAL_API_KEY")
    
    
    if not api_key:
        raise ValueError("API key not found. Set EVAL_API_KEY or OPENROUTER_API_KEY environment variable.")
    
    return DeepEvalRAGEvaluator(
        rag_pipeline=rag_pipeline,
        api_key=api_key,
        model_name=eval_config.get("model_name", "mistral/mistral-small"),
        
    )
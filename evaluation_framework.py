from deepeval.models import LiteLLMModel
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from rag import rag_pipeline
import os

class DeepEvalRAGEvaluator:
    def __init__(self, rag_pipeline, api_key: str | None):
        if api_key is None:
            raise ValueError("EVAL_API_KEY environment variable not set. ")
        
        self.llm = LiteLLMModel(
            model="mistral/mistral-small",
            api_key=api_key
        )
        self.rag = rag_pipeline
        self.faithfulness        = FaithfulnessMetric(model=self.llm)
        self.answer_relevancy    = AnswerRelevancyMetric(model=self.llm)
        self.contextual_relevancy = ContextualRelevancyMetric(model=self.llm)

    def evaluate_dataset(self, dataset: list[dict], limit:int | None) -> list[dict]:
        all_results = []
        if limit is not None and limit>0:
            dataset= dataset[:limit]
        for sample in dataset:
            query        = sample["query"]
            ground_truth = sample["grounded_answer"]
            domain       = sample.get("domain", "unknown")

            rag_out  = self.rag.get_output(query, reranked=True, top_k= 20, reranked_topk= 3)
            answer   = rag_out["message"]
            context  = rag_out["context"]
            citations = rag_out["citations"]

            if rag_out["ood"] or not context:
                all_results.append({
                    "query": query, "domain": domain,
                    "ground_truth": ground_truth,
                    "generated_answer": answer,
                    "citations": citations,
                    "metrics": {"faithfulness": None,
                                "answer_relevancy": None,
                                "contextual_relevancy": None},
                    "ood": True
                })
                continue

            test_case = LLMTestCase(
                input=query,
                actual_output=answer,
                retrieval_context=context,
                expected_output=ground_truth
            )

            self.faithfulness.measure(test_case)
            self.answer_relevancy.measure(test_case)
            self.contextual_relevancy.measure(test_case)

            all_results.append({
                "query":            query,
                "domain":           domain,
                "ground_truth":     ground_truth,
                "generated_answer": answer,
                "citations":        citations,
                "metrics": {
                    "faithfulness":         self.faithfulness.score,
                    "answer_relevancy":     self.answer_relevancy.score,
                    "contextual_relevancy": self.contextual_relevancy.score,
                },
                "ood": False
            })
        return all_results
    


rageval = DeepEvalRAGEvaluator(
    rag_pipeline=rag_pipeline,
    api_key=os.getenv("EVAL_API_KEY")
)

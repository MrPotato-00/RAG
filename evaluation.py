import json
from evaluation_framework import create_evaluator


def evaluate_document(doc_config: dict, evaluator, free_api: bool,
                      default_eval_limit: int | None = None):
    eval_data = json.load(open(doc_config['eval_data_path']))
    
    eval_limit = doc_config.get('eval_limit', default_eval_limit)
    
    
    if eval_limit:
        eval_data = eval_data[:eval_limit]
    
    results = evaluator.evaluate_dataset(eval_data, free_api=free_api)
    
    result_dict = []
    for r in results:
        result_dict.append({
            "Query": r['query'],
            "Ground Truth": r['grounded_answer'],
            "Generated Answer": r.get('generated_answer'),
            "OOD": r['ood'],
            "Faithfulness": r['metrics'].get('faithfulness'),
            "Faithfulness Reason": r['metrics'].get('faithfulness_reason'),
            "Answer Relevancy": r['metrics'].get('answer_relevancy'),
            "Answer Relevancy Reason": r['metrics'].get('answer_relevancy_reason'),
            "Contextual Relevancy": r['metrics'].get('contextual_relevancy'),
            "Contextual Relevancy Reason": r['metrics'].get('contextual_relevancy_reason'),
        })
    
    output_path = doc_config['eval_output_path']
    json.dump(result_dict, open(output_path, 'w'), indent=2)
    print(f"Evaluation for {doc_config['name']} complete. Results saved to {output_path}")


def evaluate(config_path: str = "config.json"):
    from rag import create_pipeline

    config = json.load(open(config_path))
    free_api = config.get('free_api', True)
    rag_pipeline = create_pipeline(config_path, evaluation=True)
    evaluator = create_evaluator(rag_pipeline, config)
    
    documents = config.get('documents', [])
    if not documents:
        print("No documents configured for evaluation.")
        return
    
    for doc_config in documents:
        print(f"\n{'='*50}")
        print(f"Evaluating: {doc_config['name']}")
        print(f"{'='*50}")
        evaluate_document(
            doc_config,
            evaluator,
            free_api,
            default_eval_limit=config.get('eval_limit'),
        )

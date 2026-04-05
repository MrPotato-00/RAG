from evaluation_framework import rageval
import json

config_file= json.load(open('config.json'))

def evaluate():
    result_dict= []
    results= rageval.evaluate_dataset(config_file['eval_datapath'], limit= config_file['eval_limit'])

    for r in results:

        result_dict.append({
            "Query": r['query'],
            "Ground Truth": r['ground_truth'],
            "OOD": r['ood'],
            "Faithfulness": r['metrics']['faithfulness'],
            "Answer Relevancy": r['metrics']['answer_relevancy'],
            "Contextual Relevancy": r['metrics']['contextual_relevancy'],
            "Citations": [c['source']+' p.'+str(c['page']) for c in r['citations']]
        })

    json.dump(result_dict, open(config_file['eval_output_path'], 'w'))
    print("Evaluation complete. Results saved to", config_file['eval_output_path'])


from rag import rag_pipeline
import sys

def ask(query: str, reranked: bool = True, top_k: int = 20, reranked_topk: int = 5):
    out  = rag_pipeline.get_output(query, reranked=True, dense_top_k= 10, sparse_top_k=10, reranked_topk= 3)

    print(f"Q: {query}")
    print(f"\nA: {out['message']}")
    print(f"\nContext: {out['context']}")
    if out['citations']:
        print("\n── Sources ──")
        for c in out['citations']:
            print(f"  [{c['ref']}] {c['source']} | p.{c['page']} | {c['section']}")
    else:
        print("  (no sources — out-of-domain or no match)")
    print()

if __name__=='__main__':
    if len(sys.argv) > 1:
        query= " ".join(sys.argv[1:])

    else:
        print("Follow this format to ask a question." \
        "FORMAT: python3 ask_rag.py <query>")

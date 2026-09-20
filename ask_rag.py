
import sys

def ask(query: str, pipeline=None):
    if pipeline is None:
        from rag import create_pipeline
        pipeline = create_pipeline()

    out = pipeline.get_output(query, reranked=True)

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
        ask(query)
    else:
        print("Follow this format to ask a question." \
        "FORMAT: python3 ask_rag.py <query>")

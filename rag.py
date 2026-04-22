import os
import warnings
import torch
from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from unsloth import FastLanguageModel
from process_document import ingest_papers
import json
warnings.filterwarnings("ignore", category=FutureWarning)



## Generation model. 
quant_model, quant_tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True
)
FastLanguageModel.for_inference(quant_model)


## reranker model
reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    trust_remote_code=True,
    activation_fn=torch.nn.Sigmoid()
)


class RAGPipeline:
    def __init__(self, inference_model, tokenizer, vectorstore, reranker, bm25_retriever, all_docs,
                 reranker_ood_threshold: float = 0.10):
        self.inference_model = inference_model
        self.tokenizer = tokenizer
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.bm25_retriever = bm25_retriever
        self.all_docs = all_docs
        # If the best reranker score is below this, treat query as out-of-domain
        self.reranker_ood_threshold = reranker_ood_threshold
        print(self.reranker_ood_threshold)

    # ── Retrieval ──────────────────────────────────────────────────────────────
    def similarity_searchdb(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        return self.vectorstore.similarity_search_with_score(query, k=top_k)
       
        ''' ##mmr implementation
        docs= self.vectorstore.max_marginal_relevance_search(
            query=query, 
            k=top_k,
            fetch_k= top_k*4,
            lambda_mult= 0.6
        )

        return [(doc, 0.0) for doc in docs]
        '''

    #BM25 Retrieval method
    def bm25_search(self, query: str, top_k: int) -> list[Document]:
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25_retriever.get_scores(tokenized_query)
        scored_docs = []
        for i, score in enumerate(bm25_scores):
            if score > 0: # Only consider documents with a positive score
                scored_docs.append((score, self.all_docs[i]))

        # Sort by score in descending order and get top_k documents
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:top_k]]

    #Hybrid Retrieval (combining dense and sparse)
    def _hybrid_retrieve(self, query: str, dense_top_k: int, sparse_top_k: int) -> list[Document]:
        # Dense retrieval
        dense_results_with_scores = self.similarity_searchdb(query, dense_top_k)
        dense_docs = [doc for doc, _ in dense_results_with_scores]

        # Sparse retrieval
        sparse_docs = self.bm25_search(query, sparse_top_k)

        # Combine and deduplicate
        combined_docs_map = {doc.page_content: doc for doc in dense_docs}
        for doc in sparse_docs:
            combined_docs_map[doc.page_content] = doc # Add or overwrite (if content is the same)

        # Convert back to list
        combined_docs = list(combined_docs_map.values())
        return combined_docs

    def reranked_context(self,
                         query: str,
                         docs: list[Document],
                         top_k: int) -> tuple[list[Document], float]:
        """
        Returns (reranked_docs[:top_k], best_score).
        best_score is used for out-of-domain detection.
        """
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        best_score = ranked[0][0] if ranked else 0.0
        filtered_docs= [doc for doc_score, doc in ranked if doc_score>=0.25]
        #return [doc for _, doc in ranked[:top_k]], float(best_score)
        return filtered_docs[:min(top_k, len(filtered_docs))], float(best_score)


    def _inference(self, messages: list[dict]) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.inference_model.device)
        outputs = self.inference_model.generate(
            **inputs, max_new_tokens=400, temperature=0.1, do_sample=True, max_length= None
        )
        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

    def get_message(self, query: str, formatted_context: str) -> list[dict]:
        return [
    {
      "role":"system",
    "content": (
    "You are a precise research assistant. Answer questions using ONLY the "
    "provided sources.\n\n"

    "Example:\n"
    "  Q: What optimizer was used?\n"
    "  A: The model was trained with the Adam optimizer with a warmup learning "
    "rate schedule that increased linearly for the first 4000 steps, then "
    "decreased proportionally to the inverse square root of the step number.\n\n"

    "Answering rules:\n"
    "1. Write clean prose — no inline citations, no bracket numbers in the answer text.\n"
    "2. Use ONLY information explicitly present in the sources.\n"
    "3. Always answer in a complete sentence. Never responsd with a standalone number, symbol, or single word.\n"
    "4. DO NOT paraphrase or generalize beyond what is written.\n"
    "5. Include ALL components of the answer if they appear across multiple sources.\n"
    "6. Do NOT use your training knowledge to fill gaps- if it not in the sources word-for-word, do not include it.\n"
    "7. If the question asks about a specific named external system or model "
    "(e.g., GPT-4, DeepSeek) that is NOT mentioned by that exact name in the "
    "sources, respond with exactly: 'Answer not found in the provided documents.' "
    "Do NOT apply this rule to concepts or components from the indexed papers.\n"
    "8. Do not add qualifiers, synonyms, or elaborations absent from the sources.\n"
    "9. Do not open with 'Based on the context' or similar phrases.\n"
    "10. Do NOT introduce explanations not explicitly present in the context. If a concept is not directly stated, do NOT infer it.\n"
    "11. Answer using exact statements from the context. Do NOT rephrase technical reasoning.\n"
    "12. Do not use external knowledge.\n"
    "13. If the answer is not present in the sources, respond with exactly:\n"
    "   'Answer not found in the provided documents.'\n"
)
    },
    {
      "role": "user",
      "content": f'''

      Context:
        {formatted_context}

      Question:
        {query}
    '''
    }
    ]

    #Context window truncation
    def _truncate_to_context_budget(self, docs: list[Document]) -> list[Document]:
        dummy_messages = self.get_message("q", "")
        dummy_text = self.tokenizer.apply_chat_template(
            dummy_messages, tokenize=False, add_generation_prompt=True
        )
        base_tokens = len(self.tokenizer.encode(dummy_text, add_special_tokens=False))
        capacity = self.inference_model.max_seq_length - 400 - base_tokens
        if capacity <= 0:
            return []

        kept, used = [], 0
        for doc in docs:
            n = len(self.tokenizer.encode(doc.page_content, add_special_tokens=False))
            if used + n + 2 <= capacity:
                kept.append(doc)
                used += n + 2
            else:
                remaining = capacity - used - 2
                if remaining > 0:
                    truncated_tokens = self.tokenizer.encode(
                        doc.page_content, add_special_tokens=False
                    )[:remaining]
                    truncated_text = self.tokenizer.decode(
                        truncated_tokens, skip_special_tokens=True
                    )
                    # Create a new Document with truncated content but same metadata
                    kept.append(Document(
                        page_content=truncated_text,
                        metadata=doc.metadata
                    ))
                break
        return kept

    @staticmethod
    def _format_context(docs: list[Document]) -> tuple[str, list[dict]]:
        """
        Returns:
          formatted_str  — numbered context string for the prompt
          citations      — list of citation dicts for the final output
        """
        lines = []
        citations = []
        for i, doc in enumerate(docs, 1):
            m = doc.metadata
            source_label = (
                f"{m.get('source', 'Unknown')} "
                f"(p.{m.get('page', '?')}, {m.get('section', 'Body')})"
            )
            lines.append(f"[{i}] {source_label}\n{doc.page_content}")
            citations.append({
                "ref":     i,
                "source":  m.get("source", "Unknown"),
                #"authors": m.get("authors", "Unknown"),
                "page":    m.get("page", "?"),
                "section": m.get("section", "Body"),
                "chunk_id":m.get("chunk_id", ""),
                "text":    doc.page_content,
            })
        return "\n\n".join(lines), citations
  

    
    def get_output(self,
                   query: str,
                   dense_top_k: int = 20,
                   sparse_top_k: int = 10,
                   reranked: bool = False,
                   reranked_topk: int = 8) -> dict:
        """
        Returns a dict with keys:
          message    — LLM answer (with inline [N] citations)
          context    — list of raw chunk strings (for Ragas compatibility)
          citations  — list of citation metadata dicts
          ood        — True if query appears to be out-of-domain
        """
        # Changed retrieval to hybrid
        raw_results = self._hybrid_retrieve(query, dense_top_k, sparse_top_k)

        if not raw_results:
            return {"message": "No relevant documents retrieved.",
                    "context": [], "citations": [], "ood": True}

        docs = raw_results # raw_results are now Documents, not (Document, score) tuples

        best_score = 0.0
        if reranked and self.reranker:
            k = min(len(docs), reranked_topk)
            docs, best_score = self.reranked_context(query, docs, k)
        else:
            # Score the top-1 doc to gauge OOD even without full reranking
            if self.reranker and docs: # Check if docs is not empty
                s = self.reranker.predict([[query, docs[0].page_content]])
                best_score = float(s[0])

        # Out-of-domain gate
        #ood = len(docs)==0 or best_score < self.reranker_ood_threshold

        if not docs:
            return {
                "message": "Answer not found in the provided documents.",
                "context": [],
                "citations": [],
                "ood": True
            }



        # Truncate to context budget
        docs = self._truncate_to_context_budget(docs)

        # Build formatted context with numbered citations
        formatted_context, citations = self._format_context(docs)


        messages = self.get_message(query, formatted_context)
        answer = self._inference(messages)


        return {
            "message":   answer,
            "context":   [doc.page_content for doc in docs],  # Ragas-compatible
            "citations": citations,
            "ood":       False
        }

    
vectorstore, bm25_retriever, all_docs= ingest_papers(
    pdf_paths_config= json.load(open("config.json"))['document_configs'],
    db_name= "my_chroma_db"
)
rag_pipeline = RAGPipeline(
    quant_model,
    quant_tokenizer,
    vectorstore,
    reranker,
    bm25_retriever,
    all_docs,
    reranker_ood_threshold=0.35
)


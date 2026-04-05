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
    def __init__(self, inference_model, tokenizer, vectorstore, reranker,
                 reranker_ood_threshold: float = 0.10):
        self.inference_model = inference_model
        self.tokenizer = tokenizer
        self.vectorstore = vectorstore
        self.reranker = reranker
        #
        self.reranker_ood_threshold = reranker_ood_threshold

    def similarity_searchdb(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        return self.vectorstore.similarity_search_with_score(query=query, k=top_k)

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
        return [doc for _, doc in ranked[:top_k]], float(best_score)

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
        return [{
            "role":"system",
            "content": (
            "You are a research assistant answering questions strictly based on the provided context.\n\n"

            "Instructions:\n"
            "1. Identify exact supporting statements from the context.\n"
            "2. Use only those statements to construct your answer.\n"
            "3. Do NOT use external knowledge.\n"
            "4. Do NOT infer beyond what is clearly supported.\n"
            "5. Avoid generic explanations.\n\n"
            "6. Do NOT include additional details beyond what is explicitly asked.\n\n"
            "7. Focus only on the part of the context that directly answers the question. Ignore unrelated details."
            "8. Do NOT add qualifiers or extra descriptors not explicitly required (e.g., avoid words like 'solely', 'significant', 'multi-headed' unless directly needed)."
            "9. Prefer the simpler expression of the core concepts."
            "10. Ensure all key aspects of the answer are included if present in the context."
            "11. If multiple components define the answer, include all of them."

            "Answering Rules:\n"
            "- Provide a concise answer.\n"
            "- Do NOT include phrases like 'Based on the context'.\n"
            "- Every statement MUST be supported with citation [N]."
            "- If sufficient evidence is not present, return exactly:\n"
            "  'Answer not found in the provided documents.'\n\n"

            "Important:\n"
            "Every part of your answer must be grounded in the context."

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
                "page":    m.get("page", "?"),
                "section": m.get("section", "Body"),
                "chunk_id":m.get("chunk_id", ""),
                "text":    doc.page_content,
            })
        return "\n\n".join(lines), citations

    def get_output(self,
                   query: str,
                   top_k: int = 20,
                   reranked: bool = False,
                   reranked_topk: int = 8) -> dict:
        """
        Returns a dict with keys:
          message    — LLM answer (with inline [N] citations)
          context    — list of raw chunk strings (for Ragas compatibility)
          citations  — list of citation metadata dicts
          ood        — True if query appears to be out-of-domain
        """
        raw_results = self.similarity_searchdb(query, top_k)

        if not raw_results:
            return {"message": "No relevant documents retrieved.",
                    "context": [], "citations": [], "ood": True}

        docs = [doc for doc, _ in raw_results]

        best_score = 0.0
        if reranked and self.reranker:
            k = min(len(docs), reranked_topk)
            docs, best_score = self.reranked_context(query, docs, k)
        else:
            # Score the top-1 doc to gauge OOD even without full reranking
            if self.reranker:
                s = self.reranker.predict([[query, docs[0].page_content]])
                best_score = float(s[0])

        # Out-of-domain gate
        ood = best_score < self.reranker_ood_threshold
        if ood:
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
            "context":   [doc.page_content for doc in docs],
            "citations": citations,
            "ood":       False
        }

    
vectorstore= ingest_papers(
    pdf_paths= json.load(open("config.json"))['document_paths'],
    db_name= "my_chroma_db"
)
rag_pipeline = RAGPipeline(
    quant_model,
    quant_tokenizer,
    vectorstore,
    reranker,
    reranker_ood_threshold=0.05
)


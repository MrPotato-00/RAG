# RAG Pipeline — Design Decisions

This document records every architectural decision made during the development
of this pipeline, including what was tried, what failed, and why each final
choice was made. It is intended as both an engineering reference and an
interview preparation document.

Every decision here was arrived at empirically — through evaluation rounds,
observed failures, and measured metric changes. Nothing was decided speculatively.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [PDF Processing & Chunking](#pdf-processing--chunking)
3. [Embedding Model](#embedding-model)
4. [Retrieval Strategy](#retrieval-strategy)
5. [Reranking & OOD Detection](#reranking--ood-detection)
6. [Generation & Prompting](#generation--prompting)
7. [Evaluation Framework](#evaluation-framework)
8. [What Was Tried and Reverted](#what-was-tried-and-reverted)
9. [Metric Interpretation Guide](#metric-interpretation-guide)
10. [Key Insight Summary](#key-insight-summary)

---

## System Overview

This is a **research paper RAG pipeline** built for multi-paper corpora.
The full pipeline is:

```
PDF → Chunking (with metadata) → Chroma (dense) + BM25 (sparse) index
                                              ↓
Query → Hybrid Retrieval → Reranker (+ OOD gate) → LLM Generation → Answer + Citations
                                              ↓
                                   Evaluation (DeepEval)
```

**Papers indexed at time of final evaluation:**
- Vaswani et al., "Attention Is All You Need" (Transformer paper)
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers"

**Hardware:** Google Colab T4 GPU (14.5 GB VRAM)

---

## PDF Processing & Chunking

### Decision: Per-paper chunk sizes, not a unified size

**Final config:**
```python
PAPER_CHUNK_CONFIGS = {
    "transformer_paper.pdf": (550, 100),
    "BERT.pdf":              (500, 100),
}
DEFAULT_CHUNK_CONFIG = (500, 100)
```

**How this was arrived at:**

Six evaluation rounds were run with different chunk sizes. Results showed a
consistent pattern — what fixed one paper broke the other.

| chunk_size | Transformer faithfulness | BERT faithfulness | Key observation |
|---|---|---|---|
| 800 | Variable | — | Too large, pulled in noise |
| 450 | 0.67 | — | Explanation chains split |
| 500 (unified) | 0.60 on scaling query | 1.00 on main idea | BERT fixed, Transformer broke |
| 550 (Transformer) + 500 (BERT) | 1.00 | 1.00 | Both stable |

**Root cause of Transformer failure at chunk_size=500:**
The Transformer paper's technical explanations are dense and span multiple
consecutive sentences. The scaling rationale, for example, reads:
*"for large values of dk, dot products grow large in magnitude → softmax pushed
into low-gradient regions → we scale by 1/√dk to counteract."*
At chunk_size=500, this causal chain was split across two chunks. The model
retrieved the conclusion (*"scale by 1/√dk"*) without the reasoning, then
filled in the gap from parametric knowledge — producing a wrong explanation
about *"different lengths of vectors"* instead of the correct softmax gradient
reasoning. This hallucination persisted across multiple prompt-only attempts to
fix it. Only fixing the chunk boundary resolved it.

**Root cause of BERT improvement at chunk_size=650:**
The BERT paper contains a sentence comparing BERT to OpenAI GPT in a way that
is easy to misread: *"designed to be minimally compared."* At chunk_size=500
this sentence appeared near the edge of a chunk with insufficient surrounding
context. The model consistently distorted it to *"designed to closely resemble
GPT for direct comparison"* — a subtle but meaningfully different claim. This
hallucination appeared in four consecutive evaluation rounds and was resolved
only by increasing chunk size so the sentence had more surrounding context,
reducing the model's ability to fill in gaps with parametric knowledge.

**Interview framing:** *"Optimal chunk size is document-dependent, not
pipeline-dependent. A single unified chunk size is a convenience, not a best
practice. The right chunk size is the one that keeps complete explanations
intact within a single chunk — and that boundary differs across papers with
different prose densities."*

---

### Decision: chunk_overlap=100, not 150

**Original setting:** chunk_overlap=150 (≈19% of chunk_size=800)

**Problem observed:** Near-duplicate chunks from high-overlap chunking were
appearing in retrieval results. Dense pages of the Transformer paper (e.g. p.4
covering attention mechanisms) produced multiple chunks with substantially
overlapping content.

**Why this hurts:** DeepEval's `ContextualRelevancyMetric` counts each retrieved
chunk as a separate retrieval unit. Score = relevant_chunks / total_chunks.
If 8 chunks are retrieved but 3 are near-duplicates of each other, the evaluator
correctly marks the duplicates as redundant, reducing the score. The fix is
upstream — reduce overlap so fewer near-duplicates are created.

**Quantified impact:** Approximately 0.05–0.10 improvement in contextual
relevancy after reducing overlap from 150 to 100.

---

### Decision: Strip inline reference numbers `[1]`, `[23]`

```python
text = re.sub(r"\[\d+\]", "", text)
```

**Why:** Inline reference numbers add noise to chunk embeddings without adding
semantic meaning. A chunk containing *"the Adam optimizer [20] with β1 = 0.9"*
should embed the same as one containing *"the Adam optimizer with β1 = 0.9"*.

**Side effect resolved:** Early pipeline versions showed the LLM producing
answers with copied reference numbers (*"The Adam optimizer [20]..."*). Stripping
them at ingestion time eliminated this at source.

---

### Decision: Skip pages containing "Input-Input"

```python
if "Input-Input" in raw:
    continue
```

**Why:** The Transformer paper contains attention visualisation figures. PDF
extraction of these pages produces repetitive token strings
(*"Input-Input-Input-the-the-the..."*) with zero semantic content but high
token count. These pages produce chunks that attract spurious similarity search
results because they contain high-frequency tokens. Filtering them at extraction
is cleaner than trying to filter at retrieval time.

---

### Decision: Section detection via regex, not structural parsing

```python
SECTION_PATTERNS = re.compile(
    r'^(abstract|introduction|related work|...|references)',
    re.IGNORECASE | re.MULTILINE
)
```

**Purpose:** Chunk metadata carries a `section` field used for citation display
and debugging. When a chunk retrieves incorrectly, the section field immediately
shows whether the problem is in the wrong part of the paper.

**Known limitation:** Most chunks are labelled "Body" because the regex only
fires when a section heading appears at the start of the chunk. Section-aware
splitting (splitting at heading boundaries rather than character count) would
give more meaningful labels but was not implemented — it adds complexity without
improving retrieval quality in the evaluation runs conducted.

---

## Embedding Model

### Decision: `sentence-transformers/all-MiniLM-L6-v2`

**Why chosen:** Fast (22M parameters), runs on CPU alongside the GPU-hosted
inference model, produces 384-dimensional embeddings, and is sufficient for
single-domain technical corpora where queries and documents share vocabulary.

**Known limitation documented:** This model was trained on general text.
It has limited understanding of mathematical notation and technical jargon.
Queries phrased with different notation from the paper (e.g. *"why multiply by
1/√dk"* vs *"why scale dot products"*) may not retrieve the same chunk despite
referring to the same concept. This was observed in contextual relevancy scores
on the scaling query, which never exceeded 0.43–0.67 across evaluation rounds
— suggesting the retrieval space is imprecise for technical content.

**What would improve it:** `BAAI/bge-base-en-v1.5` or `intfloat/e5-base-v2` —
both trained on more technical corpora and consistently outperform MiniLM on
technical retrieval benchmarks. Implemeted `BAAI/bge-base-en-v1.5` because it 
had knowledge of technical papers and jargons. 

**Interview framing:** *"The embedding model shapes the retrieval space — what
'similar' means is entirely defined by this model. I documented the MiniLM
limitation as a known ceiling rather than pretending the contextual relevancy
plateau was a prompt problem."*

---

## Retrieval Strategy

### Decision: Hybrid retrieval (dense + BM25)

**Why hybrid:** Dense retrieval captures semantic meaning but can miss exact
technical terms. BM25 captures exact keyword matches but misses paraphrased
queries. Combining them improves recall — particularly for queries containing
specific notation (β1, dk, BLEU scores) or model names that may not embed well
in the general-purpose MiniLM space.

Deduplication is by `page_content` — if both retrievers return the same chunk,
only one copy is kept (dense result takes priority via dict ordering).

---

### Decision: MMR was tested and reverted

**What was tried:**
```python
docs = self.vectorstore.max_marginal_relevance_search(
    query=query, k=top_k, fetch_k=top_k * 4, lambda_mult=0.6
)
```

**Result:** Contextual relevancy got worse after implementing MMR.

**Why:** MMR (Maximum Marginal Relevance) penalises retrieving chunks that are
similar to already-selected chunks. In a multi-document corpus this enforces
diversity across sources. In a single dense technical paper, all the relevant
chunks are about the same topic and are therefore similar to each other by
design. MMR's diversity penalty actively avoids the most relevant chunks.

**When MMR would be appropriate:** Multi-document corpora where you want
coverage across different sources. For single-topic paper corpora, pure
similarity search + reranker is the better combination.

**Empirical result:** Reverting to `similarity_search_with_score` improved
contextual relevancy by approximately 0.2-0.6 across affected queries.

---

### Decision: dense_top_k=10, sparse_top_k=10, reranked_topk=3

**How these were arrived at:**

Initial settings were dense_top_k=20, reranked_topk=8. The citation output for
the main innovation query showed 8 chunks retrieved eg. from pages 1, 10, 1, 1, 2,
3, 8, 2 — page 1 appearing three times, page 8 and 10 appearing for a question
whose answer is in the abstract/introduction. The context was crowded and noisy.

Reducing to dense_top_k=10 and reranked_topk=3 produced cleaner, more focused
context. With only 3 highly-scored chunks the model had less noise to filter
through and faithfulness improved. Also introducing BM25 search and passing the
parameter through sparse_top_k retrieved the documents through the keyword search.

**Trade-off:** Lower reranked_topk risks missing information split across more
than 3 chunks. This was acceptable because the reranker concentrates highly
relevant content into its top scores — when the answer is present, the top 1–2
chunks typically contain it.

---

## Reranking & OOD Detection

### Decision: CrossEncoder as reranker and OOD gate

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` with sigmoid activation

**Why sigmoid activation:**
Raw cross-encoder scores are unbounded logits. Sigmoid maps them to [0,1],
making `reranker_ood_threshold` interpretable. Without sigmoid, the threshold
value is arbitrary and not meaningful across different queries.

**Dual role:**
The reranker serves two purposes simultaneously:
1. Reorder retrieved chunks by query-relevance (primary)
2. Filter chunks below OOD threshold — if none survive, query is OOD

This is the most efficient design — one model pass does both jobs.

---

### Decision: reranker_ood_threshold = 0.35

**Evolution:**

| Threshold | Observed problem |
|---|---|
| 0.10 (original) | DeepSeek-R1 query passed through, LLM hallucinated Transformer architecture |
| 0.35 (final) | DeepSeek-R1 correctly blocked; valid queries unaffected |

**Why 0.35:** With sigmoid activation, ms-marco MiniLM scores cluster as:
- `> 0.5` — chunk directly answers the query
- `0.2–0.5` — topically adjacent but not directly answering
- `< 0.2` — not relevant

0.35 sits between "topically adjacent" and "directly relevant" — the right
boundary for OOD detection. Queries whose best chunk scores in the 0.2–0.35
range are being retrieved for topical adjacency, not because the corpus
contains a direct answer.

---

### Decision: Single OOD filtering point inside `reranked_context`

**Problem with the original design:**
```python
# Original — two separate checks, potential inconsistency
filtered_docs = [doc for score, doc in ranked if score >= threshold]  # in reranked_context
ood = best_score < threshold  # in get_output, uses pre-filter best_score
```

`best_score` was computed from the ranked list before per-chunk filtering,
then checked again in `get_output`. Edge case: if the top chunk scores exactly
at threshold and one lower chunk scores below, `ood` is False while the
post-filter set is reduced — inconsistent behaviour.

**Fix:** Filter only inside `reranked_context`. `get_output` checks `len(docs) == 0`
after reranking — if no chunks survived, the query is OOD. One filtering point,
no redundancy.

---

### Decision: No LLM answerability check (`_is_answerable` removed)

**What was implemented:**
```python
def _is_answerable(self, query, formatted_context) -> bool:
    # YES/NO: does the context contain a direct answer?
    raw = self._inference(messages).strip().upper()
    return raw.startswith("YES")
```

**Why it was removed:**
It caused false negatives on valid in-domain queries. *"Why does the Transformer
remove recurrence?"* was being rejected as OOD because the 3B model would start
answering the question instead of returning YES, or would inconsistently return
NO even when the context clearly contained the answer.

Binary classification is not reliable from a generative model at 3B scale.
The reranker score is a more stable signal — it is a discriminative model
trained specifically for relevance scoring, not a generative model attempting
a classification task as a side job.

**The problem it was trying to solve (P100 GPU query):**
The paper mentions *"eight P100 GPUs"* in passing as a training duration
reference. A query asking *"What GPU model was used?"* passes the reranker gate
(P100 is mentioned) but the answer is imprecise. This is handled instead by
prompt rules that prevent the model from constructing specific claims from
incidental mentions.

---

### Decision: Named external system check via prompt (Rule 7), not retrieval

**The problem:**
The BERT paper mentions OpenAI GPT as a comparison baseline. A query about
*"GPT-4 architecture"* causes the reranker to retrieve those BERT chunks
(GPT is mentioned) and score them above 0.35. The query passes the reranker
OOD gate even though GPT-4 is not in the corpus.

**First attempt — regex entity matching:**
```python
query_entities = set(re.findall(r'\b[A-Z][a-z]+(?:-[A-Za-z0-9]+)+\b|...', query))
# Check if entities appear in retrieved chunks
```
This was corpus-specific and fragile — it caught hyphenated model names
(deepseek-r1) but failed on non-hyphenated entities and would break when
the corpus changed.

**Final solution — prompt rule 7:**
```
"If the question asks about a specific named external system or model
(e.g., GPT-4, DeepSeek) that is NOT mentioned by that exact name in the
sources, respond with exactly: 'Answer not found in the provided documents.'
Do NOT apply this rule to concepts or components from the indexed papers."
```

**Critical nuance in rule wording:** The rule must say "external system or model"
not "named entity." An earlier version said "named entity" and caused the model
to refuse answers about "transformer encoder layer" — treating "transformer"
as a named entity requiring corpus verification. The distinction between
*external system* (GPT-4, DeepSeek) and *concept from indexed papers*
(transformer encoder, self-attention mechanism) is the boundary the rule enforces.

---

## Generation & Prompting

### Decision: Clean prose answers, citations as separate list

**Evolution across development:**

| Version | Approach | Problem observed |
|---|---|---|
| v1 | Inline [N] citations in answer | Model produced "as stated in [1]" narration |
| v2 | Silent [N] after clause | Inconsistent — sometimes narrated, sometimes correct |
| Final | No [N] in answer; citations as separate list | Clean prose, no LLM involvement in citation display |

**Why small models fail at inline citation:**
A 3B model has insufficient instruction-following capacity to simultaneously
generate a faithful, complete answer AND implement a nuanced citation format.
Asking it to do both degrades both. The cleaner architecture separates concerns:
- LLM generates clean prose (one task)
- Pipeline code displays citations from metadata (zero LLM involvement)

---

### Decision: Few-shot example in system prompt

```python
"Example:
  Q: What optimizer was used?
  A: The model was trained with the Adam optimizer with a warmup learning
     rate schedule that increased linearly for the first 4000 steps..."
```

**Why:** Small models pattern-match examples more reliably than they follow
abstract rules. Showing a concrete desired output — complete sentence, no
citations, no hedging phrases — gives the model a template to follow.

Negative rules (*"do NOT write 'as stated in [N]'"*) were less effective because
the model would fixate on the forbidden pattern. The example implicitly teaches
the correct format without mentioning the wrong one.

**Example choice:** The optimizer example was deliberately chosen because it
has a two-part answer (optimizer name + learning rate schedule), which implicitly
demonstrates multi-source synthesis in the correct format.

---

### Decision: Complete sentence rule (Rule 3)

```
"Always answer in a complete sentence. Never respond with a standalone
number, symbol, or single word."
```

**The problem it fixes:**
The dropout rate query was answered with bare `"0.1"`. DeepEval's faithfulness
metric scored this 0.00 — it couldn't match the standalone number to the chunk
containing `"Pdrop = 0.1"` (a different string representation).

After adding Rule 3, the answer became *"The dropout rate used in the transformer
model is 0.1."* — faithfulness jumped from 0.00 to 1.00 on the same correct answer.

---

### Decision: Rules 6, 10, 11 — parametric knowledge containment

```
"6. Do NOT use your training knowledge to fill gaps."
"10. Do NOT introduce explanations not explicitly present in the context."
"11. Answer using exact statements from the context. Do NOT rephrase technical reasoning."
```

**The problem they fix:**
The BERT main idea query produced a consistent hallucination across 4 evaluation
rounds: *"design decisions intentionally made to closely resemble GPT for direct
comparison."* The paper actually says BERT and GPT were designed to be "minimally
compared" — the model distorted this using its parametric knowledge about the
BERT-GPT relationship.

These three rules together tell the model: reproduce, don't synthesise.
Combined with fixing the chunk boundary (increasing BERT chunk size so the
GPT comparison sentence had more surrounding context), the hallucination was
eliminated.

**Important limitation:** These rules reduce hallucination frequency but cannot
eliminate it without also retrieving the right context. Prompt rules cannot
substitute for chunking quality. This was validated empirically — the rules
reduced the hallucination but did not eliminate it until the chunk size was fixed.

---

### Decision: 13-rule prompt (verbose, not minimal)

The system prompt grew from 5 rules to 13 across development. Each rule was
added to fix a specific observed failure, not speculatively. The rules are
non-overlapping — each covers distinct ground:

| Rule | Problem it fixes |
|---|---|
| 1 | Citation narration ("as stated in [N]") |
| 2 | External knowledge leakage |
| 3 | Bare-number answers (faithfulness 0.00) |
| 4 | Paraphrasing that subtly changes meaning |
| 5 | Incomplete multi-part answers |
| 6 | Parametric knowledge filling gaps |
| 7 | Named external system OOD (GPT-4, DeepSeek) |
| 8 | Extra qualifiers not in source ("significantly", "solely") |
| 9 | Hedging preambles ("Based on the context...") |
| 10 | Inferred explanations not explicitly stated |
| 11 | Paraphrased technical reasoning (scaling hallucination) |
| 12 | External knowledge (backup to rule 6) |
| 13 | Explicit fallback for unanswerable queries |

---

## Evaluation Framework

### Decision: API judge (Mistral-small), not local 3B model

**Why not local:**

A local `HuggingFaceLLM` class was implemented in first iteration.
It failed because DeepEval passes Pydantic schemas to the judge and expects
structured JSON back. The 3B model failed to produce valid JSON reliably.
`lm-format-enforcer` was added to constrain decoding to valid JSON — but Colab and 
even Kaggle failed to provide an environment for installation and implementation
of `lm-format-enforcer` with correct formatting, and with multiple local models 
running already for embedding, generation, a model for evaluation is already 
infeasible for local development, though I took a local first, complete offline 
approach but local constraints are real. Using external service api still serves 
the purpose:

**Cost justification:** Evaluation runs on ~25 samples, not on every user query.
API cost per eval run is negligible. The evaluation signal is only as good as
the judge — using a weak judge to optimise the pipeline means optimising for
the judge's errors, not actual quality.

---

### Decision: Three separate LLMTestCase objects per metric

```python
faithfulness_case         = LLMTestCase(input=q, actual_output=a, retrieval_context=ctx)
answer_relevancy_case     = LLMTestCase(input=q, actual_output=a)
contextual_relevancy_case = LLMTestCase(input=q, actual_output=a, retrieval_context=ctx)
```

**Why:** `FaithfulnessMetric` requires `retrieval_context`. `AnswerRelevancyMetric`
does not — passing `retrieval_context` to it is harmless but makes the code
intent unclear. Separate test cases make each metric's required fields explicit
and self-documenting.

---

### Decision: Exclude OOD samples from aggregate score averages

**Why:** A correct refusal (*"Answer not found in the provided documents."*)
scores 0.00 answer_relevancy — the judge sees it as failing to address the
question, which is technically accurate from the evaluator's perspective.
Including OOD samples in averages penalises correct pipeline behaviour.

OOD gate correctness is assessed separately by verifying the `ood` flag is
True for known OOD queries (deepseek-r1, GPT-4) — it is not captured by
the three numeric metrics.

---

## What Was Tried and Reverted

| Approach | Observed metric impact | Reason reverted |
|---|---|---|
| chunk_overlap=150 | Contextual relevancy ~0.35 avg | Near-duplicate chunks diluted retrieval |
| Unified chunk_size=500 | BERT improved, Transformer faithfulness 0.60 | Per-paper sizing validated empirically |
| MMR retrieval | Contextual relevancy worse | Single-paper corpus: diversity penalty harmful |
| Reranker threshold=0.10 | OOD queries answered, faithfulness 1.00 on hallucinations | Raised to 0.35 |
| LLM answerability check (YES/NO) | False negatives on "why transformer remove recurrence" | Removed; reranker gate sufficient |
| Inline [N] citations in answer | "as stated in [1]" narration in output | Clean prose + separate citation list |
| Bare-number generation | Faithfulness 0.00 on correct answers | Complete sentence rule (Rule 3) added |
| dense_top_k=20, reranked_topk=8 | Context crowded, 8 chunks with duplicates | Reduced to 10 dense, 3 reranked |
| Named entity regex for OOD | Failed on non-hyphenated entities; corpus-specific | Replaced with prompt Rule 7 |
| `_sufficient_answer_inference` | Experimental; not stable enough | Kept in code as commented-out method |

---

## Metric Interpretation Guide

Understanding which metric diagnoses which pipeline component was the most
practically useful insight from this project.

| Metric | What it measures | Low score diagnosis | Component to fix |
|---|---|---|---|
| **Faithfulness** | Is the answer supported by retrieved chunks? | Answer fabricates or contradicts context | Generation / Prompt |
| **Answer Relevancy** | Does the answer address the question? | Answer is off-topic, incomplete, or hedging | Generation / Prompt |
| **Contextual Relevancy** | Are the retrieved chunks relevant to the query? | Retrieval returning noisy, duplicate, or tangential chunks | Retrieval / Chunking |

**Pattern observed repeatedly:**
High faithfulness + high answer relevancy + low contextual relevancy =
*"The LLM is using its context correctly, but retrieval is returning the wrong context."*

This pattern appeared in the early Transformer evaluations where contextual
relevancy was ~0.35 while faithfulness was 1.00. Prompt changes had zero effect
on contextual relevancy across multiple rounds — because contextual relevancy
is a retrieval metric, not a generation metric. The fix (chunk size) was upstream.

**Implication for debugging:**
Always check which metric is low before deciding what to change. Changing the
prompt when contextual relevancy is the problem wastes iteration cycles.
Changing chunk parameters when faithfulness is the problem does the same.

---

## Key Insight Summary

The single most important insight from building this pipeline:

> **In RAG, everything eventually depends on chunking strategy.**
> The embedding model, chunk size, and chunk overlap together define the
> *unit of retrieval* — the smallest piece of information the pipeline can
> find and return. If a chunk cuts an explanation in half, no reranker,
> no prompt, and no generation technique can recover the missing half.
> Chunking mistakes are the hardest to fix because they require full re-ingestion.

The practical hierarchy for a new RAG project:
1. **Chunking strategy first** — document structure, chunk size, overlap
2. **Embedding model second** — shapes the retrieval space permanently
3. **Chunk size tuning third** — empirical, requires evaluation loop
4. **Everything else** — reranker, prompt, OOD logic — all tunable at query
   time without re-ingestion, lower risk to iterate on

This hierarchy was validated by the fact that contextual relevancy did not
improve across multiple rounds of prompt and threshold tuning, but changed
immediately when chunk boundaries were adjusted. The metrics told us clearly
which component needed fixing — we just had to listen to them.
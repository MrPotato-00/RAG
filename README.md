# RAG Pipeline Project

A Retrieval-Augmented Generation (RAG) system for answering questions based on research papers, with built-in evaluation capabilities.

## Overview

This project implements a complete RAG pipeline that processes PDF documents, retrieves relevant information, and generates accurate answers using a large language model. It includes out-of-domain detection, reranking for improved relevance, and comprehensive evaluation using established metrics.  
Note: Go through the DECISION.md file to understand the various decisions and failures faced while undertaking this project. This is not a production grade project but helped me understand that building a RAG workflow is not only about retrieval and generation. A whole lot of engineering decision goes behind to be able to generate grounded answers to a query. Hope you enjoy the read.

## Features

- **Document Processing**: Automatic PDF ingestion with intelligent chunking and metadata extraction
- **Vector Search**: Efficient similarity search using Chroma vectorstore and sentence embeddings
- **Reranking**: Cross-encoder-based reranking for improved retrieval quality
- **BM25**: Applied lexical search along with semantic search to get more relevant context retrieval
- **Out-of-Domain Detection**: Automatic detection of queries outside the document scope
- **Citation Support**: Numbered citations in generated answers for traceability
- **Evaluation Framework**: Comprehensive evaluation using DeepEval metrics (Faithfulness, Answer Relevancy, Contextual Relevancy)
- **Configurable**: JSON-based configuration for easy customization

## Workflow

### 1. Document Ingestion
- Extract text from PDF documents using PyPDF
- Normalize and clean text content
- Split documents into semantic chunks with overlap
- Generate embeddings using Sentence Transformers
- Store chunks in Chroma vectorstore with rich metadata (source, page, section)

### 2. Retrieval and Reranking
- Perform hybrid search (vector search + BM25) to retrieve top-k relevant chunks
- Apply cross-encoder reranking to improve result quality
- Detect out-of-domain queries using reranker confidence scores
- Truncate context to fit model token limits

### 3. Answer Generation
- Use quantized LLM (Qwen2.5-3B) for answer generation
- Strict prompting to ensure answers are grounded in provided context
- Include numbered citations for source verification
- Handle out-of-domain queries gracefully

### 4. Evaluation
- Load evaluation dataset from configuration
- Run queries through the RAG pipeline
- Measure answer quality using DeepEval metrics:
  - **Faithfulness**: Consistency between answer and retrieved context
  - **Answer Relevancy**: Relevance of answer to the query
  - **Contextual Relevancy**: Relevance of retrieved context to the query
- Generate detailed reports with metrics and citations

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RAG
```

2. Install dependencies:
```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export EVAL_API_KEY=your_eval_api_key
```

## Usage

### Basic RAG Query
```python
from rag import rag_pipeline

result = rag_pipeline.get_output("What is the main innovation in transformers?")
print(result['message'])
print(result['citations'])
```

### Running Evaluation with setting eval=True in config.json
```python
python3 main.py
```

This will load the evaluation dataset from `config.json` and run comprehensive evaluation.

## Configuration

The system is configured via `config.json`:

- It contains a list of evaluation samples with queries, ground truth answers, and domains

Example configuration:
```json
{
  [
    {
      "query": "What is self-attention?",
      "grounded_answer": "Self-attention is a mechanism...",
      "domain": "in_domain_technical"
    }
  ]
}

The system is configured to run on arxiv based research papers. You can run you evaluation on preferred datasource (pdf) with custom set of evaluation query-answer pairs. 
```

## Dependencies

- PyTorch
- Transformers/Unsloth
- LangChain
- ChromaDB
- Sentence Transformers
- DeepEval
- LiteLLM
- PyPDF

See `requirements.txt` for complete list.

## Models Used

- SentenceTransformers (all-MiniLM-L6-v2): for embedding generation for the vectordb
- cross-encoder/ms-marco-MiniLM-L-6-v2: for reranking task
- Qwen/Qwen2.5-3B-Instruct: for generation of the output from the context
- Mistral API (free) key for Evaluation purpose using the LiteLLM wrapper

## API Keys supported by LiteLLM

OpenAI Models
- openai/gpt-3.5-turbo
- openai/gpt-4
- openai/gpt-4-turbo-preview

Anthropic Models
- anthropic/claude-3-opus
- anthropic/claude-3-sonnet
- anthropic/claude-3-haiku

Google Models
- google/gemini-pro
- google/gemini-ultra

Mistral Models
- mistral/mistral-small
- mistral/mistral-medium
- mistral/mistral-large

LM Studio Models
- lm-studio/Meta-Llama-3.1-8B-Instruct-GGUF
- lm-studio/Mistral-7B-Instruct-v0.2-GGUF
- lm-studio/Phi-2-GGUF

Ollama Models
- ollama/llama2
- ollama/mistral
- ollama/codellama
- ollama/neural-chat
- ollama/starling-lm

## Below are some evaluation executed using the RAG pipeline. ContextualRelevancy score needs to be improved but note that answer generation is working well. 
```json
[{'query': 'Why is scaling applied in the dot-product attention mechanism?',
  'domain': 'deep_technical',
  'ground_truth': 'Scaling prevents the dot product values from becoming too large, which would push the softmax function into regions with very small gradients and make training unstable.',
  'generated_answer': 'Scaling applied in the dot-product attention mechanism is to make the scores comparable and to normalize the values, which is given as 1/√dk in the context.',
  'metrics': {'faithfulness': 1.0,
   'faithfulness_reason': 'The score is 1.00 because the actual output perfectly aligns with the retrieval context, showing no contradictions and demonstrating complete faithfulness to the provided information.',
   'answer_relevancy': 1.0,
   'answer_relevancy_reason': 'The score is 1.00 because the answer fully and precisely addresses the input question about the purpose of scaling in dot-product attention without any irrelevant statements or omissions.',
   'contextual_relevancy': 0.5,
   'contextual_relevancy_reason': "The score is 0.50 because the retrieval context partially addresses the input by mentioning the scaling factor in dot-product attention ('Dot-product attention is identical to our algorithm, except for the scaling factor of 1√dk') but fails to explain why scaling is applied, as highlighted in the reasons for irrelevancy."},
  'ood': False},
 {'query': 'What are the main components of a transformer encoder layer?',
  'domain': 'conceptual',
  'ground_truth': 'A transformer encoder layer consists of a multi-head self-attention mechanism followed by a position-wise feed-forward network, with residual connections and layer normalization applied.',
  'generated_answer': 'A transformer encoder layer consists of a multi-head self-attention mechanism and a positionwise fully connected feed-forward network.',
  'metrics': {'faithfulness': 1.0,
   'faithfulness_reason': 'The score is 1.00 because the actual output perfectly aligns with the retrieval context, showing no contradictions or discrepancies.',
   'answer_relevancy': 1.0,
   'answer_relevancy_reason': 'The score is 1.00 because the answer perfectly addresses the input by clearly and concisely listing the main components of a transformer encoder layer without any irrelevant statements.',
   'contextual_relevancy': 0.5,
   'contextual_relevancy_reason': "The score is 0.50 because the retrieval context includes some relevant statements about the transformer encoder layer (e.g., 'The encoder is composed of a stack of N = 6 identical layers.' and 'Each layer has two sub-layers. The first is a multi-head self-attention mechanism...'), but it is diluted by irrelevant details about attention applications and encoder-decoder interactions (e.g., 'queries come from the previous decoder layer' and 'memory keys and values come from the output of the encoder')."},
  'ood': False},
 {'query': 'What is the dropout rate used in the transformer model?',
  'domain': 'factual_detail',
  'ground_truth': 'The dropout rate used in the transformer is 0.1.',
  'generated_answer': 'The Transformer (big) model used a dropout rate of 0.1.',
  'metrics': {'faithfulness': 1.0,
   'faithfulness_reason': 'The score is 1.00 because the actual output perfectly aligns with the retrieval context with no contradictions—great job!',
   'answer_relevancy': 1.0,
   'answer_relevancy_reason': 'The score is 1.00 because the output perfectly addresses the input by directly answering the dropout rate used in the transformer model without any irrelevant statements.',
   'contextual_relevancy': 0.25,
   'contextual_relevancy_reason': "The score is 0.25 because the retrieval context only contains one marginally relevant statement ('The Transformer (big) model trained for English-to-French used dropout rate Pdrop = 0.1, instead of 0.3.') while the rest of the context is entirely irrelevant, discussing checkpoint averaging and lacking any direct mention of the dropout rate used in the transformer model."},
  'ood': False},
 {'query': 'What GPU model was used to train the transformer in the original paper?',
  'domain': 'out-of-domain',
  'ground_truth': 'The paper mentions training on multiple GPUs but does not specify the exact GPU model, so this cannot be precisely answered from the source.',
  'generated_answer': 'Answer not found in the provided documents.',
  'citations': [],
  'metrics': {'faithfulness': None,
   'answer_relevancy': None,
   'contextual_relevancy': None},
  'ood': True},
 {'query': 'What is the architecture used by the deepseek-r1 model?',
  'domain': 'out-of-domain',
  'ground_truth': 'Answer not provided in the documents.',
  'generated_answer': 'Answer not found in the provided documents.',
  'citations': [],
  'metrics': {'faithfulness': None,
   'answer_relevancy': None,
   'contextual_relevancy': None},
  'ood': True}]

```

## Project Structure

```
├── config.json              # Configuration file
├── data/                    # Evaluation datasource and query-answer pair
├── evaluation.py            # Evaluation script
├── evaluation_framework.py  # Evaluation logic
├── main.py                  # Project entry point / placeholder script
├── process_document.py      # Document processing
├── rag.py                   # Main RAG pipeline
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

`main.py` is the project entry point and currently provides a simple startup placeholder. You can extend it to wire together ingestion, query handling, and evaluation flows for a command-line application.

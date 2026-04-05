# RAG Pipeline Project

A Retrieval-Augmented Generation (RAG) system for answering questions based on research papers, with built-in evaluation capabilities.

## Overview

This project implements a complete RAG pipeline that processes PDF documents, retrieves relevant information, and generates accurate answers using a large language model. It includes out-of-domain detection, reranking for improved relevance, and comprehensive evaluation using established metrics.

## Features

- **Document Processing**: Automatic PDF ingestion with intelligent chunking and metadata extraction
- **Vector Search**: Efficient similarity search using Chroma vectorstore and sentence embeddings
- **Reranking**: Cross-encoder-based reranking for improved retrieval quality
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
- Perform similarity search to retrieve top-k relevant chunks
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

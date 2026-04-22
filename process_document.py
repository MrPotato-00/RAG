from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
import torch
import re
import os
import unicodedata
import json


## loading the embedding model from huggingface
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")


SECTION_PATTERNS = re.compile(
    r'^(abstract|introduction|related work|background|methodology|method|methods|'
    r'model|architecture|approach|experiment|experiments|results|evaluation|'
    r'discussion|conclusion|conclusions|references|appendix|acknowledgements?)',
    re.IGNORECASE | re.MULTILINE
)

def normalize_text(text: str) -> str:
    """Clean raw PDF text."""
    text = unicodedata.normalize("NFKC", text) 
    text = re.sub(r"\S+@\S+", "", text)           
    text = re.sub(r'arXiv:\S+', '', text)          
    text = re.sub(r'Conference on.*?\n', '', text)
    text = re.sub(r'[†‡∗]', '', text)              
    text = re.sub(r'-\n', '', text)               
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  
    text = re.sub(r'[ \t]+', ' ', text)           
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r"\[\d+\]", "", text)
    return text.strip()


def detect_section(text: str) -> str:
    """Return the first section heading found in a chunk, else 'body'."""
    match = SECTION_PATTERNS.search(text)
    return match.group(0).strip().title() if match else "Body"



def extract_pdf_with_metadata(pdf_path: str) -> list[dict]:
    """
    Returns a list of dicts, one per page:
        {"text": ..., "page": int, "source": str, "authors": str}

    This step all good, tried and tested !!
    """
    reader = PdfReader(pdf_path)
    paper_title = os.path.splitext(os.path.basename(pdf_path))[0]

    first_page_text = reader.pages[0].extract_text() or ""

    pages = []
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        if "Input-Input" in raw:
          continue
        pages.append({
            "text": raw,
            "page": i + 1,
            "source": paper_title,
           
        })

    return pages


def chunk_paper(pdf_path: str,
                chunk_size: int = 800,
                chunk_overlap: int = 150) -> list[Document]:
    """
    Chunk a single PDF and attach rich metadata to every chunk.
    Returns a list of LangChain Documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )

    pages = extract_pdf_with_metadata(pdf_path)
    documents = []

    for page_info in pages:
        clean_text = normalize_text(page_info["text"])
        if not clean_text.strip():
            continue

        chunks = splitter.split_text(clean_text)
        for j, chunk in enumerate(chunks):
            chunk = chunk.strip()

            if chunk.startswith(". "):
                chunk = chunk[2:]
            if len(chunk) < 50:      
                continue

            section = detect_section(chunk)
            chunk_id = f"{page_info['source']}_p{page_info['page']}_c{j}"

            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source":   page_info["source"],
             
                    "page":     page_info["page"],
                    "section":  section,
                    "chunk_id": chunk_id,
                }
            ))

    print(f"  [{page_info['source']}] -> {len(documents)} chunks from {len(pages)} pages")
    return documents


def ingest_papers(pdf_paths_config: list[dict],
                  db_name: str = "my_chroma_db") -> tuple[Chroma, BM25Okapi, list[Document]]:
    """
    Ingest one or more PDFs into a fresh Chroma vectorstore and build a BM25 index.
    Supports multi-paper RAG out of the box.
    """
    # Clear existing collection
    if os.path.exists(db_name):
        Chroma(persist_directory=db_name,
               embedding_function=embed_model).delete_collection()

    all_docs = []
    for doc_config in pdf_paths_config:
        all_docs.extend(chunk_paper(doc_config['path'], chunk_size= doc_config['chunk_size'], chunk_overlap=doc_config['chunk_overlap']))
      

    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embed_model,
        persist_directory=db_name
    )

    tokenized_corpus = [doc.page_content.split(" ") for doc in all_docs]
    bm25_retriever = BM25Okapi(tokenized_corpus)
    print(f"\nVectorstore ready: {vectorstore._collection.count()} total chunks, BM25 index built.")
    return vectorstore, bm25_retriever, all_docs


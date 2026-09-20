from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
import os
import unicodedata
from rank_bm25 import BM25Okapi

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


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


def create_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Load the embedding model only when an index operation is requested."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model_name)


def _build_bm25(documents: list[Document]) -> BM25Okapi:
    if not documents:
        raise ValueError("The index contains no document chunks.")
    return BM25Okapi([doc.page_content.split() for doc in documents])


def build_index(pdf_paths_config: list[dict],
                db_name: str,
                embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
                rebuild: bool = False) -> tuple[Chroma, BM25Okapi, list[Document]]:
    """
    Build a Chroma and BM25 index explicitly.

    Existing indexes are protected unless ``rebuild`` is explicitly requested.
    """
    if os.path.exists(db_name):
        if not rebuild:
            raise FileExistsError(
                f"Index already exists at {db_name!r}. Use --rebuild to replace it."
            )
        embedding_model = create_embedding_model(embedding_model_name)
        Chroma(persist_directory=db_name,
               embedding_function=embedding_model).delete_collection()
    else:
        embedding_model = create_embedding_model(embedding_model_name)

    all_docs = []
    for doc_config in pdf_paths_config:
        all_docs.extend(chunk_paper(doc_config['pdf_path'], chunk_size= doc_config['chunk_size'], chunk_overlap=doc_config['chunk_overlap']))

    if not all_docs:
        raise ValueError("No document chunks were produced; check the configured PDFs.")

    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embedding_model,
        persist_directory=db_name
    )

    bm25_retriever = _build_bm25(all_docs)
    print(f"\nVectorstore ready: {vectorstore._collection.count()} total chunks, BM25 index built.")
    return vectorstore, bm25_retriever, all_docs


def load_index(db_name: str,
               embedding_model_name: str = DEFAULT_EMBEDDING_MODEL
               ) -> tuple[Chroma, BM25Okapi, list[Document]]:
    """Load an existing Chroma index and reconstruct its in-memory BM25 index."""
    if not os.path.exists(db_name):
        raise FileNotFoundError(
            f"No index found at {db_name!r}. Run the ingest command first."
        )

    vectorstore = Chroma(
        persist_directory=db_name,
        embedding_function=create_embedding_model(embedding_model_name),
    )
    stored = vectorstore.get(include=["documents", "metadatas"])
    documents = [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(stored["documents"], stored["metadatas"])
    ]
    return vectorstore, _build_bm25(documents), documents


def ingest_papers(pdf_paths_config: list[dict],
                  db_name: str = "my_chroma_db") -> tuple[Chroma, BM25Okapi, list[Document]]:
    """Backward-compatible explicit index build; never overwrites an index."""
    return build_index(pdf_paths_config, db_name)

import google.generativeai as genai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os
from typing import List

load_dotenv()

# Initialize Google Gemini client (FREE!)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Embedding configuration - using Google's free embedding model
EMBED_MODEL = "models/embedding-001"
EMBED_DIM = 768  # Dimension for Gemini embedding-001

# Text splitter configuration
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Load a PDF and chunk it into smaller text segments.
    
    Args:
        path: Path to the PDF file
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Load PDF documents
    docs = PDFReader().load_data(file=path)
    
    # Extract text from documents
    texts = [d.text for d in docs if getattr(d, "text", None)]
    
    # Chunk the text
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed texts using Google Gemini's FREE embedding model.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors (768 dimensions each)
    """
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        embeddings.append(result['embedding'])
    return embeddings


def embed_query(text: str) -> List[float]:
    """
    Embed a single query text for search.
    
    Args:
        text: Query text to embed
        
    Returns:
        Embedding vector (768 dimensions)
    """
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_query"
    )
    return result['embedding']


def get_embedding_dimension() -> int:
    """Return the embedding dimension for the current model."""
    return EMBED_DIM
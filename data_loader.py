from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os
from typing import List

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Embedding configuration
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072  # Dimension for text-embedding-3-large

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
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed texts using OpenAI's embedding model.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors (3072 dimensions each)
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def get_embedding_dimension() -> int:
    """Return the embedding dimension for the current model."""
    return EMBED_DIM
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from openai import OpenAI
import os
from typing import List


def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
    """
    Load a PDF and chunk it into smaller text segments.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Load PDF
    reader = PDFReader()
    documents = reader.load_data(file=pdf_path)
    
    # Chunk the documents
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    
    for doc in documents:
        nodes = splitter.get_nodes_from_documents([doc])
        chunks.extend([node.text for node in nodes])
    
    return chunks


def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
    """
    Embed texts using OpenAI's embedding model.
    
    Args:
        texts: List of texts to embed
        model: OpenAI embedding model to use
        
    Returns:
        List of embedding vectors
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    
    return [item.embedding for item in response.data]

import pydantic
from typing import Optional


class RAGChunkAndSrc(pydantic.BaseModel):
    """Represents chunks from a PDF with their source ID."""
    chunks: list[str]
    source_id: Optional[str] = None


class RAGUpsertResult(pydantic.BaseModel):
    """Result from upserting vectors into the database."""
    ingested: int


class RAGSearchResult(pydantic.BaseModel):
    """Result from searching the vector database."""
    contexts: list[str]
    sources: list[str]


class RAQQueryResult(pydantic.BaseModel):
    """Result from an AI query with answer and sources."""
    answer: str
    sources: list[str]
    num_contexts: int
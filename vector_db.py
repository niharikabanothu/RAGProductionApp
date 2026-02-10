from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from typing import List, Dict
import os


class QdrantStorage:
    """
    Wrapper for Qdrant vector database operations.
    """
    
    def __init__(self, url: str = "http://localhost:6333", collection: str = "docs", dim: int = 3072):
        """
        Initialize Qdrant client and ensure collection exists.
        
        Args:
            url: Qdrant server URL
            collection: Name of the collection to use
            dim: Vector dimension (3072 for text-embedding-3-large)
        """
        qdrant_url = os.getenv("QDRANT_URL", url)
        self.client = QdrantClient(url=qdrant_url, timeout=30)
        self.collection = collection
        self.dim = dim
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
    
    def upsert(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict]):
        """
        Upsert vectors into the collection.
        """
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(self.collection, points=points)
    
    def search(self, query_vector: List[float], top_k: int = 5) -> Dict[str, List]:
        """
        Search for similar vectors.
        """
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            with_payload=True,
            limit=top_k
        )
        
        contexts = []
        sources = set()
        
        for r in results:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)
        
        return {"contexts": contexts, "sources": list(sources)}
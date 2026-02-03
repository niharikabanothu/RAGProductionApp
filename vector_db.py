from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict
import os


class QdrantStorage:
    """
    Wrapper for Qdrant vector database operations.
    """
    
    def __init__(self, collection_name: str = "rag_collection"):
        """
        Initialize Qdrant client and ensure collection exists.
        
        Args:
            collection_name: Name of the collection to use
        """
        self.collection_name = collection_name
        
        # Initialize client (use in-memory or URL from env)
        qdrant_url = os.getenv("QDRANT_URL")
        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url)
        else:
            # Use in-memory storage for development
            self.client = QdrantClient(":memory:")
        
        # Create collection if it doesn't exist
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
    
    def upsert(self, ids: List[str], vectors: List[List[float]], payloads: List[Dict]):
        """
        Upsert vectors into the collection.
        
        Args:
            ids: List of unique IDs for the vectors
            vectors: List of embedding vectors
            payloads: List of metadata dictionaries
        """
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(self, query_vector: List[float], top_k: int = 5) -> Dict[str, List]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            Dictionary with 'contexts' and 'sources' lists
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        
        contexts = [hit.payload["text"] for hit in results]
        sources = [hit.payload["source"] for hit in results]
        
        return {"contexts": contexts, "sources": sources}

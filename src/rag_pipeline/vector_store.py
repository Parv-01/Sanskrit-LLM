"""Vector database interface for embedding storage and retrieval."""

from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path
import json


class VectorDatabase:
    """Vector database for storing and retrieving text embeddings.
    
    Provides interface for similarity search using FAISS or
    in-memory storage for development.
    
    Example:
        >>> db = VectorDatabase()
        >>> db.add_embeddings(embeddings, metadata)
        >>> results = db.search(query_embedding, top_k=5)
    """
    
    def __init__(
        self,
        dimension: int = 384,
        use_faiss: bool = False,
        index_path: Optional[str] = None,
    ):
        """Initialize the vector database.
        
        Args:
            dimension: Embedding dimension.
            use_faiss: Whether to use FAISS for indexing.
            index_path: Path to save/load index.
        """
        self.dimension = dimension
        self.use_faiss = use_faiss
        self.index_path = index_path
        self.index = None
        self.metadata = []
        self.texts = []
        
        if use_faiss:
            self._init_faiss()
    
    def _init_faiss(self) -> None:
        """Initialize FAISS index."""
        try:
            import faiss
            self.index = faiss.IndexFlatL2(self.dimension)
            self.faiss = faiss
        except ImportError:
            print("Warning: faiss-cpu not installed, using in-memory storage")
            self.use_faiss = False
            self.index = None
    
    def add_embeddings(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        metadata: Optional[List[Dict]] = None,
    ) -> None:
        """Add embeddings to the database.
        
        Args:
            embeddings: Array of embeddings.
            texts: Corresponding text strings.
            metadata: Optional metadata for each text.
        """
        self.texts.extend(texts)
        
        if metadata is None:
            metadata = [{}] * len(texts)
        self.metadata.extend(metadata)
        
        if self.use_faiss and self.index is not None:
            self.index.add(embeddings)
        else:
            if self.index is None:
                self.index = embeddings
            else:
                self.index = np.vstack([self.index, embeddings])
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.
            
        Returns:
            List of result dictionaries.
        """
        if self.index is None or len(self.texts) == 0:
            return []
        
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        top_k = min(top_k, len(self.texts))
        
        if self.use_faiss and self.index is not None:
            distances, indices = self.index.search(query_embedding, top_k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                results.append({
                    'text': self.texts[idx],
                    'metadata': self.metadata[idx],
                    'distance': float(dist),
                })
        else:
            if isinstance(self.index, np.ndarray):
                similarities = np.dot(self.index, query_embedding.T).flatten()
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                results = []
                for idx in top_indices:
                    results.append({
                        'text': self.texts[idx],
                        'metadata': self.metadata[idx],
                        'similarity': float(similarities[idx]),
                    })
        
        return results
    
    def save(self, path: str) -> None:
        """Save the database to disk.
        
        Args:
            path: Path to save database.
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.use_faiss and self.index is not None:
            faiss.write_index(self.index, str(save_path / "index.faiss"))
        
        np.save(save_path / "texts.npy", np.array(self.texts, dtype=object))
        
        with open(save_path / "metadata.json", 'w') as f:
            json.dump(self.metadata, f)
    
    def load(self, path: str) -> None:
        """Load the database from disk.
        
        Args:
            path: Path to load database from.
        """
        load_path = Path(path)
        
        if self.use_faiss:
            try:
                import faiss
                self.index = faiss.read_index(str(load_path / "index.faiss"))
            except:
                pass
        
        self.texts = np.load(load_path / "texts.npy", allow_pickle=True).tolist()
        
        with open(load_path / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
    
    def get_all_texts(self) -> List[str]:
        """Get all stored texts.
        
        Returns:
            List of all texts.
        """
        return self.texts
    
    def clear(self) -> None:
        """Clear all stored data."""
        self.texts = []
        self.metadata = []
        if self.use_faiss and self.index is not None:
            self.index.reset()
        else:
            self.index = None


def demo():
    """Demonstration function for vector database."""
    import numpy as np
    
    db = VectorDatabase(dimension=10)
    
    embeddings = np.random.randn(5, 10)
    texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
    
    db.add_embeddings(embeddings, texts)
    
    query = np.random.randn(10)
    results = db.search(query, top_k=3)
    
    print("Search results:")
    for r in results:
        print(f"  {r['text']}: {r.get('similarity', r.get('distance'))}")


if __name__ == "__main__":
    demo()

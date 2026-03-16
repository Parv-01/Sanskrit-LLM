"""Embedding generation for Sanskrit text."""

from typing import List, Optional, Dict, Any
import numpy as np


class EmbeddingGenerator:
    """Generates embeddings for Sanskrit text using transformer models.
    
    Uses pre-trained sentence transformers to create dense vector
    representations of Sanskrit Ayurvedic texts.
    
    Example:
        >>> generator = EmbeddingGenerator()
        >>> embeddings = generator.generate(["ज्वरस्य निदानं"])
        >>> print(embeddings.shape)
    """
    
    DEFAULT_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: str = "cpu",
    ):
        """Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model.
            device: Device to run model on ('cpu' or 'cuda').
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device
        self.model = None
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except ImportError:
            print("Warning: sentence-transformers not installed")
            self.model = None
    
    def generate(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts.
        
        Args:
            texts: List of input texts.
            
        Returns:
            Array of embeddings.
        """
        if self.model is None:
            self._load_model()
        
        if self.model is None:
            return self._dummy_embeddings(len(texts))
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def generate_with_metadata(
        self,
        texts: List[str],
        metadata: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate embeddings with metadata.
        
        Args:
            texts: List of input texts.
            metadata: Metadata for each text.
            
        Returns:
            List of dicts with embedding and metadata.
        """
        embeddings = self.generate(texts)
        
        results = []
        for i, (text, meta) in enumerate(zip(texts, metadata)):
            results.append({
                'text': text,
                'embedding': embeddings[i].tolist(),
                'metadata': meta,
            })
        
        return results
    
    def _dummy_embeddings(self, count: int, dim: int = 384) -> np.ndarray:
        """Generate dummy embeddings when model unavailable.
        
        Args:
            count: Number of embeddings.
            dim: Embedding dimension.
            
        Returns:
            Random embeddings.
        """
        np.random.seed(42)
        return np.random.randn(count, dim)
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between embeddings.
        
        Args:
            embedding1: First embedding.
            embedding2: Second embedding.
            
        Returns:
            Cosine similarity score.
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        return float(dot_product / (norm1 * norm2))
    
    def batch_generate(
        self,
        texts: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Generate embeddings in batches.
        
        Args:
            texts: List of input texts.
            batch_size: Size of each batch.
            
        Returns:
            Array of embeddings.
        """
        if self.model is None:
            self._load_model()
        
        if self.model is None:
            return self._dummy_embeddings(len(texts))
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        
        return embeddings


def demo():
    """Demonstration function for embedding generation."""
    generator = EmbeddingGenerator()
    
    texts = [
        "आयुर्वेदः सर्वदा रक्षति",
        "ज्वरस्य निदानं तिक्ताम्ललवणं परिहरेत्",
    ]
    
    print("Generating embeddings...")
    embeddings = generator.generate(texts)
    print(f"Embeddings shape: {embeddings.shape}")
    
    similarity = generator.compute_similarity(embeddings[0], embeddings[1])
    print(f"Similarity: {similarity:.4f}")


if __name__ == "__main__":
    demo()

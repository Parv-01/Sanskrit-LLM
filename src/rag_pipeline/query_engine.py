"""RAG query engine for question answering."""

from typing import List, Dict, Any, Optional
import numpy as np


class RAGQueryEngine:
    """Retrieval-Augmented Generation query engine.
    
    Combines retrieval from vector database with LLM generation
    to answer questions about Ayurvedic texts.
    
    Example:
        >>> engine = RAGQueryEngine()
        >>> engine.index_knowledge_base(texts, embeddings)
        >>> answer = engine.query("What treatments for fever?")
    """
    
    DEFAULT_CONTEXT_TEMPLATE: str = """
Context information:
{context}

Based on the above context, answer the following question:
Question: {question}

Answer (in English):
"""
    
    def __init__(
        self,
        llm_model: Optional[str] = None,
        embedding_generator: Optional[Any] = None,
        vector_store: Optional[Any] = None,
    ):
        """Initialize the RAG query engine.
        
        Args:
            llm_model: LLM model identifier.
            embedding_generator: Embedding generator instance.
            vector_store: Vector store instance.
        """
        self.llm_model = llm_model
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.llm_client = None
        self.context_template = self.DEFAULT_CONTEXT_TEMPLATE
    
    def _init_llm(self) -> None:
        """Initialize the LLM client."""
        try:
            from langchain_openai import ChatOpenAI
            self.llm_client = ChatOpenAI(
                model=self.llm_model or "gpt-3.5-turbo",
                temperature=0.3,
            )
        except ImportError:
            print("Warning: LLM client not available")
    
    def index_knowledge_base(
        self,
        texts: List[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """Index the knowledge base.
        
        Args:
            texts: List of text documents.
            embeddings: Pre-computed embeddings (optional).
        """
        if self.embedding_generator is None:
            from .embeddings import EmbeddingGenerator
            self.embedding_generator = EmbeddingGenerator()
        
        if embeddings is None:
            embeddings = self.embedding_generator.generate(texts)
        
        if self.vector_store is None:
            from .vector_store import VectorDatabase
            self.vector_store = VectorDatabase(dimension=embeddings.shape[1])
        
        self.vector_store.add_embeddings(embeddings, texts)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: Input query string.
            top_k: Number of documents to retrieve.
            
        Returns:
            List of retrieved documents with metadata.
        """
        if self.vector_store is None or self.embedding_generator is None:
            return []
        
        query_embedding = self.embedding_generator.generate([query])
        results = self.vector_store.search(query_embedding[0], top_k=top_k)
        
        return results
    
    def assemble_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Assemble context from retrieved documents.
        
        Args:
            retrieved_docs: List of retrieved documents.
            
        Returns:
            Assembled context string.
        """
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[{i}] {doc['text']}")
        
        return "\n\n".join(context_parts)
    
    def generate(
        self,
        query: str,
        context: str,
    ) -> str:
        """Generate answer using LLM.
        
        Args:
            query: Input query.
            context: Retrieved context.
            
        Returns:
            Generated answer.
        """
        if self.llm_client is None:
            self._init_llm()
        
        if self.llm_client is None:
            return "LLM not available. Retrieved context:\n" + context
        
        prompt = self.context_template.format(
            context=context,
            question=query,
        )
        
        try:
            response = self.llm_client.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error generating response: {e}"
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Complete RAG query pipeline.
        
        Args:
            question: Input question.
            top_k: Number of documents to retrieve.
            
        Returns:
            Dictionary with answer and metadata.
        """
        retrieved_docs = self.retrieve(question, top_k=top_k)
        
        if not retrieved_docs:
            return {
                'question': question,
                'answer': 'No relevant information found.',
                'sources': [],
            }
        
        context = self.assemble_context(retrieved_docs)
        answer = self.generate(question, context)
        
        sources = [
            {
                'text': doc['text'],
                'score': doc.get('similarity', doc.get('distance', 0)),
            }
            for doc in retrieved_docs
        ]
        
        return {
            'question': question,
            'answer': answer,
            'sources': sources,
            'num_sources': len(retrieved_docs),
        }
    
    def query_batch(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Query multiple questions.
        
        Args:
            questions: List of questions.
            
        Returns:
            List of answers.
        """
        return [self.query(q) for q in questions]


def demo():
    """Demonstration function for RAG query engine."""
    import numpy as np
    
    engine = RAGQueryEngine()
    
    texts = [
        "ज्वरस्य निदानं तिक्ताम्ललवणं परिहरेत्।",
        "पित्तप्रकोपे शीतलं जलं पिबेत्।",
        "कफरोगे गुडूचीरसायनं सेवयेत्।",
    ]
    
    print("Indexing knowledge base...")
    engine.index_knowledge_base(texts)
    
    print("\nQuery: What treatments are recommended?")
    result = engine.query("What treatments are recommended for fever?")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['num_sources']}")


if __name__ == "__main__":
    demo()

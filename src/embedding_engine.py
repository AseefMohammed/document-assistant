"""
Embedding Engine Module
Handles text embedding generation and vector storage using FAISS
for the fintech document assistant RAG system.
"""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    from langchain.schema import Document
except ImportError:
    SentenceTransformer = None
    faiss = None
    Document = None

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    Generates embeddings for text chunks and manages vector storage using FAISS.
    Optimized for fintech document processing and compliance requirements.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        vector_store_path: str = "./data/vector_store",
        index_name: str = "document_index"
    ):
        """
        Initialize the embedding engine.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to run the model on ('cpu' or 'cuda')
            vector_store_path: Path to store the FAISS index
            index_name: Name of the FAISS index file
        """
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")
        if faiss is None:
            raise ImportError("faiss-cpu not installed. Install with: pip install faiss-cpu")
        
        self.model_name = model_name
        self.device = device
        self.vector_store_path = Path(vector_store_path)
        self.index_name = index_name
        
        # Create vector store directory if it doesn't exist
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize the embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = None
        self.documents = []
        self.document_metadata = []
        
        # Load existing index if available
        self._load_index()
    
    def _get_index_path(self) -> Path:
        """Get the path to the FAISS index file."""
        return self.vector_store_path / f"{self.index_name}.faiss"
    
    def _get_metadata_path(self) -> Path:
        """Get the path to the metadata file."""
        return self.vector_store_path / f"{self.index_name}_metadata.pkl"
    
    def _get_documents_path(self) -> Path:
        """Get the path to the documents file."""
        return self.vector_store_path / f"{self.index_name}_documents.pkl"
    
    def _create_index(self) -> faiss.IndexFlatIP:
        """Create a new FAISS index for similarity search."""
        # Using IndexFlatIP for cosine similarity (after L2 normalization)
        index = faiss.IndexFlatIP(self.embedding_dimension)
        logger.info(f"Created new FAISS index with dimension {self.embedding_dimension}")
        return index
    
    def _load_index(self) -> bool:
        """
        Load existing FAISS index and metadata.
        
        Returns:
            True if index was loaded successfully, False otherwise
        """
        index_path = self._get_index_path()
        metadata_path = self._get_metadata_path()
        documents_path = self._get_documents_path()
        
        if index_path.exists() and metadata_path.exists() and documents_path.exists():
            try:
                # Load FAISS index
                self.index = faiss.read_index(str(index_path))
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    self.document_metadata = pickle.load(f)
                
                # Load documents
                with open(documents_path, 'rb') as f:
                    self.documents = pickle.load(f)
                
                logger.info(f"Loaded existing index with {self.index.ntotal} vectors")
                return True
                
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                logger.info("Creating new index")
        
        # Create new index if loading failed
        self.index = self._create_index()
        self.documents = []
        self.document_metadata = []
        return False
    
    def _save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        try:
            index_path = self._get_index_path()
            metadata_path = self._get_metadata_path()
            documents_path = self._get_documents_path()
            
            # Save FAISS index
            faiss.write_index(self.index, str(index_path))
            
            # Save metadata
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.document_metadata, f)
            
            # Save documents
            with open(documents_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            logger.info(f"Saved index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings with shape (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10,
                batch_size=32
            )
            
            # Normalize embeddings for cosine similarity
            embeddings = embeddings.astype('float32')
            faiss.normalize_L2(embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
        """
        if not documents:
            logger.warning("No documents provided to add")
            return
        
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Extract texts for embedding
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        if embeddings.size == 0:
            logger.warning("No embeddings generated")
            return
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store documents and metadata
        self.documents.extend(documents)
        for i, doc in enumerate(documents):
            metadata = doc.metadata.copy()
            metadata['vector_id'] = len(self.document_metadata) + i
            self.document_metadata.append(metadata)
        
        # Save updated index
        self._save_index()
        
        logger.info(f"Successfully added {len(documents)} documents. Total vectors: {self.index.ntotal}")
    
    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents using the query.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of tuples containing (Document, similarity_score)
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        if self.index.ntotal == 0:
            logger.warning("No documents in the index")
            return []
        
        logger.info(f"Searching for: '{query[:50]}...' (top_k={top_k}, threshold={threshold})")
        
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])
            
            if query_embedding.size == 0:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search in FAISS index
            similarities, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
            
            # Filter results by threshold and prepare response
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    break
                
                if similarity >= threshold:
                    document = self.documents[idx]
                    results.append((document, float(similarity)))
                else:
                    logger.debug(f"Result {i} below threshold: {similarity} < {threshold}")
            
            logger.info(f"Found {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """
        Retrieve a document by its chunk ID.
        
        Args:
            doc_id: The chunk ID to search for
            
        Returns:
            Document object if found, None otherwise
        """
        for doc in self.documents:
            if doc.metadata.get('chunk_id') == doc_id:
                return doc
        return None
    
    def get_all_documents(self) -> List[Document]:
        """
        Get all documents in the vector store.
        
        Returns:
            List of all Document objects
        """
        return self.documents.copy()
    
    def delete_documents_by_source(self, source_path: str) -> int:
        """
        Delete all documents from a specific source file.
        
        Args:
            source_path: Path of the source file
            
        Returns:
            Number of documents deleted
        """
        source_path = os.path.abspath(source_path)
        indices_to_remove = []
        
        for i, metadata in enumerate(self.document_metadata):
            if metadata.get('source') == source_path:
                indices_to_remove.append(i)
        
        if not indices_to_remove:
            logger.info(f"No documents found for source: {source_path}")
            return 0
        
        # Remove from documents and metadata (in reverse order to maintain indices)
        for i in reversed(indices_to_remove):
            del self.documents[i]
            del self.document_metadata[i]
        
        # Rebuild the FAISS index (FAISS doesn't support efficient deletion)
        logger.info("Rebuilding FAISS index after deletion")
        self.index = self._create_index()
        
        if self.documents:
            texts = [doc.page_content for doc in self.documents]
            embeddings = self.generate_embeddings(texts)
            self.index.add(embeddings)
        
        # Update vector IDs in metadata
        for i, metadata in enumerate(self.document_metadata):
            metadata['vector_id'] = i
        
        self._save_index()
        
        logger.info(f"Deleted {len(indices_to_remove)} documents from {source_path}")
        return len(indices_to_remove)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index."""
        stats = {
            "total_vectors": self.index.ntotal if self.index else 0,
            "embedding_dimension": self.embedding_dimension,
            "model_name": self.model_name,
            "device": self.device,
            "index_size_mb": 0
        }
        
        # Calculate index file size
        index_path = self._get_index_path()
        if index_path.exists():
            stats["index_size_mb"] = index_path.stat().st_size / (1024 * 1024)
        
        # Document statistics
        if self.documents:
            sources = set(doc.metadata.get('source', '') for doc in self.documents)
            file_types = {}
            for doc in self.documents:
                file_type = doc.metadata.get('file_type', 'unknown')
                file_types[file_type] = file_types.get(file_type, 0) + 1
            
            stats.update({
                "unique_sources": len(sources),
                "file_types": file_types
            })
        
        return stats
    
    def clear_index(self) -> None:
        """Clear all documents and reset the index."""
        logger.warning("Clearing all documents from the index")
        
        self.index = self._create_index()
        self.documents = []
        self.document_metadata = []
        
        # Remove saved files
        for path in [self._get_index_path(), self._get_metadata_path(), self._get_documents_path()]:
            if path.exists():
                path.unlink()
        
        logger.info("Index cleared successfully")

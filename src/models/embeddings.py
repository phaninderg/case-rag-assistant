from typing import List, Dict, Any, Optional, Union
import os
import logging
from pathlib import Path

from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config.settings import settings
from .factory import ModelFactory

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the embedding model to use. If None, uses the default from settings.
        """
        self.model_name = model_name or settings.default_embedding
        self.embedding_config = settings.get_embedding_config(self.model_name)
        
        # Initialize the embedding model
        self.embeddings = ModelFactory.create_embeddings(self.embedding_config)
        
        # Initialize vector store
        self.vector_store = Chroma(
            collection_name="case_embeddings",
            embedding_function=self.embeddings,
            persist_directory=str(settings.embeddings_dir)
        )
        
        # Configure text splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
    
    def _prepare_documents(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """Split text into chunks and create document objects."""
        # Split text into chunks
        texts = self.text_splitter.split_text(text)
        
        # Create document objects with metadata
        documents = []
        for i, text_chunk in enumerate(texts):
            doc_metadata = metadata.copy()
            doc_meta = {
                "chunk": i,
                "chunk_size": len(text_chunk),
                **doc_metadata
            }
            documents.append(Document(
                page_content=text_chunk,
                metadata=doc_meta
            ))
        
        return documents
    
    def add_document(self, case_id: str, text: str, metadata: Dict[str, Any]):
        """
        Add a document to the vector store.
        
        Args:
            case_id: Unique identifier for the case
            text: Text content to embed
            metadata: Additional metadata for the document
        """
        # Prepare documents with proper chunking
        docs = self._prepare_documents(text, {
            "case_id": case_id,
            **metadata
        })
        
        # Add to vector store
        self.vector_store.add_documents(docs)
        self.vector_store.persist()
    
    def search_similar(self, query: str, k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: The search query
            k: Number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of similar documents with scores
        """
        # Perform similarity search
        docs_and_scores = self.vector_store.similarity_search_with_score(
            query,
            k=min(k, 20),  # Limit max results
            **kwargs
        )
        
        # Format results
        results = []
        for doc, score in docs_and_scores:
            result = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            }
            results.append(result)
        
        return results
    
    def get_all_cases(self) -> List[Dict[str, Any]]:
        """
        Get all cases from the vector store with their metadata.
        
        Returns:
            List of cases with their chunks and metadata
        """
        # Get all documents from the collection
        collection = self.vector_store._collection
        if collection is None:
            return []
            
        results = collection.get()
        
        # Group chunks by case_id
        cases = {}
        for i, (doc_id, doc_text, metadata) in enumerate(zip(
            results["ids"],
            results["documents"],
            results["metadatas"]
        )):
            case_id = metadata.get("case_id")
            if not case_id:
                continue
                
            if case_id not in cases:
                cases[case_id] = {
                    "case_id": case_id,
                    "title": metadata.get("title", "Untitled"),
                    "created_at": metadata.get("created_at"),
                    "chunks": [],
                    "metadata": {k: v for k, v in metadata.items() 
                                 if k not in ["chunk", "chunk_size"]}
                }
            
            cases[case_id]["chunks"].append({
                "chunk_id": metadata.get("chunk", i),
                "text": doc_text,
                "length": len(doc_text)
            })
        
        # Sort chunks by chunk_id
        for case in cases.values():
            case["chunks"].sort(key=lambda x: x["chunk_id"])
        
        return list(cases.values())

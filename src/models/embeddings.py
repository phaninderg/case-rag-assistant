from typing import List, Dict, Any, Optional, Union
import os
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

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
        self.vector_store = self._initialize_vector_store()
        
        # Configure text splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
        )
    
    def _initialize_vector_store(self):
        """Initialize the vector store with the current embedding model"""
        try:
            # Ensure the directory exists with full permissions
            persist_dir = Path(settings.embeddings_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Set very permissive permissions
            persist_dir.chmod(0o777)
            
            # Log directory permissions for debugging
            import os
            import stat
            
            def get_permissions(path):
                st = os.stat(str(path))
                return {
                    'readable': os.access(str(path), os.R_OK),
                    'writable': os.access(str(path), os.W_OK),
                    'executable': os.access(str(path), os.X_OK),
                    'mode': oct(stat.S_IMODE(st.st_mode)),
                    'owner': st.st_uid,
                    'group': st.st_gid
                }
            
            logger.info(f"Vector store directory info: {persist_dir}")
            logger.info(f"Directory permissions: {get_permissions(persist_dir)}")
            
            # Create the vector store with explicit permissions
            vector_store = Chroma(
                collection_name="case_embeddings",
                embedding_function=self.embeddings,
                persist_directory=str(persist_dir)
            )
            
            # Verify we can write to the directory
            test_file = persist_dir / ".test_write"
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                test_file.unlink()
                logger.info("Successfully wrote test file to vector store directory")
            except Exception as e:
                logger.error(f"Failed to write test file to {persist_dir}: {str(e)}")
                raise
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}", exc_info=True)
            raise
    
    def update_embedding_model(self, model_name: str, model_path: Optional[str] = None):
        """
        Update the embedding model being used
        
        Args:
            model_name: Name of the model to use
            model_path: Optional path to a local model
        """
        try:
            self.model_name = model_name
            if model_path:
                # Load model from local path
                self.embeddings = ModelFactory.create_embeddings({
                    'name': model_name,
                    'path': model_path
                })
            else:
                # Load model from name
                self.embedding_config = settings.get_embedding_config(model_name)
                self.embeddings = ModelFactory.create_embeddings(self.embedding_config)
            
            # Reinitialize the vector store with the new embeddings
            self.vector_store = self._initialize_vector_store()
            logger.info(f"Updated embedding model to {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to update embedding model: {str(e)}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        try:
            # Handle empty or None text
            if not text or not isinstance(text, str) or not text.strip():
                logger.warning("Empty or invalid text provided for embedding")
                return [0.0] * self.embeddings.dimensions
                
            # Generate embedding
            embedding = self.embeddings.embed_query(text)
            
            # Ensure we have a valid embedding
            if not embedding or not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
                logger.error(f"Invalid embedding generated for text: {text[:100]}...")
                return [0.0] * self.embeddings.dimensions
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return zero vector of appropriate dimension
            return [0.0] * (self.embeddings.dimensions if hasattr(self.embeddings, 'dimensions') else 768)
    
    def similarity_search(
        self, 
        query_embedding: List[float], 
        k: int = 5, 
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Find similar cases using vector similarity search
        
        Args:
            query_embedding: Embedding vector to compare against
            k: Maximum number of results to return
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of similar cases with scores
        """
        try:
            # Convert query embedding to the expected format
            if not isinstance(query_embedding, (list, np.ndarray)) or not all(isinstance(x, (int, float, np.number)) for x in query_embedding):
                logger.error("Invalid query embedding format")
                return []
            
            # Convert to numpy array if needed, then to list
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)
            
            # Convert to Python list for ChromaDB compatibility
            query_embedding_list = query_embedding.tolist()
            
            # Get documents and scores in one go using the vector
            results = self.vector_store.similarity_search_by_vector(
                embedding=query_embedding_list,
                k=k
            )
            
            # Get the scores using the collection's query method directly
            collection = self.vector_store._collection
            query_results = collection.query(
                query_embeddings=[query_embedding_list],
                n_results=k
            )
            
            # Process and filter results
            processed_results = []
            for i, doc in enumerate(results):
                if i < len(query_results['distances'][0]):
                    # Get the score (distance) for this document
                    distance = query_results['distances'][0][i]
                    # Convert distance to similarity score (1 - distance for cosine similarity)
                    similarity = 1.0 - float(distance)
                    
                    if similarity >= min_score:
                        metadata = doc.metadata
                        processed_results.append({
                            'case_number': metadata.get('case_number', 'N/A'),
                            'subject': metadata.get('subject', 'No subject'),
                            'description': doc.page_content,
                            'similarity_score': similarity,
                            'metadata': metadata
                        })
            
            # Sort by similarity score (highest first)
            processed_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}", exc_info=True)
            return []
    
    def add_case(self, case_data: Dict[str, Any]) -> bool:
        """
        Add a case to the vector store

        Args:
            case_data: Dictionary containing case data

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Extract relevant fields
            case_task_number = case_data.get('case_task_number')
            if not case_task_number:
                logger.error("Case task number is required")
                return False

            # Prepare document text with all relevant information
            text = (
                f"Case Task: {case_task_number}\n"
                f"Parent Case: {case_data.get('parent_case', 'N/A')}\n"
                f"Tags: {case_data.get('case_task_tags', 'N/A')}\n"
                f"Issue: {case_data.get('issue', '')}\n"
                f"Root Cause: {case_data.get('root_cause', '')}\n"
                f"Resolution: {case_data.get('resolution', '')}\n"
                f"Support Steps: {case_data.get('steps_support', '')}"
            )

            # Include all fields in metadata
            metadata = {
                'case_task_number': case_task_number,
                'parent_case': case_data.get('parent_case', ''),
                'case_task_tags': case_data.get('case_task_tags', ''),
                'issue': case_data.get('issue', ''),
                'root_cause': case_data.get('root_cause', ''),
                'resolution': case_data.get('resolution', ''),
                'steps_support': case_data.get('steps_support', ''),
                'created_at': case_data.get('created_at', datetime.utcnow().isoformat()),
                'updated_at': case_data.get('updated_at', datetime.utcnow().isoformat())
            }

            # Add to vector store
            self.vector_store.add_documents([Document(page_content=text, metadata=metadata)])

            # Persist changes
            self.vector_store.persist()

            return True

        except Exception as e:
            logger.error(f"Error adding case to vector store: {str(e)}")
            return False
    
    def get_all_cases(self) -> List[Dict[str, Any]]:
        """
        Get all cases from the vector store
        
        Returns:
            List of case documents
        """
        try:
            # Get all document IDs
            collection = self.vector_store._collection
            if not collection:
                return []
                
            # Get all documents
            docs = collection.get(include=['metadatas', 'documents'])
            if not docs or 'metadatas' not in docs or 'documents' not in docs:
                return []
                
            # Format results
            results = []
            for i, (metadata, content) in enumerate(zip(docs['metadatas'], docs['documents'])):
                results.append({
                    'id': i,
                    'content': content,
                    'metadata': metadata or {}
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Error getting all cases: {str(e)}")
            return []
    
    def add_document(self, case_number: str, text: str, metadata: Dict[str, Any]):
        """
        Add a document to the vector store.
        
        Args:
            case_number: The case number (used as the primary identifier)
            text: Text content to embed
            metadata: Additional metadata for the document
        """
        if not case_number:
            raise ValueError("Case number is required")
            
        # Prepare documents with proper chunking
        docs = self._prepare_documents(
            text, 
            {
                "case_number": case_number,
                **metadata
            }
        )
        
        # Add to vector store
        self.vector_store.add_documents(docs)
        self.vector_store.persist()
    
    def get_vector_store_info(self) -> Dict[str, Any]:
        """
        Get information about the vector store contents.
        
        Returns:
            Dictionary with vector store information
        """
        try:
            collection = self.vector_store._collection
            return {
                "num_documents": collection.count(),
                "embedding_dimension": len(collection.peek(1)['embeddings'][0]) if collection.count() > 0 else 0,
                "collection_name": collection.name,
                "persist_directory": str(self.vector_store._persist_directory)
            }
        except Exception as e:
            logger.error(f"Error getting vector store info: {str(e)}")
            return {"error": str(e)}

    def search_similar(self, query: str, k: int = 5, min_score: float = 0.5, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: The search query
            k: Number of results to return
            min_score: Minimum similarity score (0-1)
            **kwargs: Additional search parameters
            
        Returns:
            List of similar documents with scores
        """
        try:
            logger.info(f"Searching for query: '{query}' with k={k}, min_score={min_score}")
            
            # Get query embedding
            try:
                if hasattr(self.embeddings, 'encode'):
                    query_embedding = self.embeddings.encode(query, convert_to_tensor=False)
                    if hasattr(query_embedding, 'tolist'):
                        query_embedding = query_embedding.tolist()
                    logger.debug(f"Generated query embedding of length: {len(query_embedding)}")
                else:
                    logger.error("Embedding model does not support 'encode' method")
                    return []
            except Exception as e:
                logger.error(f"Error generating query embedding: {str(e)}")
                return []
            
            # Perform similarity search
            try:
                # First try with the query embedding
                docs_and_scores = self.vector_store.similarity_search_with_score(
                    query_embedding,
                    k=min(k * 2, 20),  # Get more results to filter by score
                    **kwargs
                )
                
                # Format and filter results
                results = []
                for doc, score in docs_and_scores:
                    similarity = 1.0 - score  # Convert distance to similarity
                    if similarity >= min_score:
                        results.append({
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "score": similarity
                        })
                
                # Sort by score (highest first) and limit to k results
                results.sort(key=lambda x: x['score'], reverse=True)
                results = results[:k]
                
                logger.info(f"Found {len(results)} results with scores >= {min_score}")
                if results:
                    logger.debug(f"Top result score: {results[0]['score']:.4f}, Content: {results[0]['content'][:100]}...")
                
                return results
                
            except Exception as e:
                logger.error(f"Error in similarity search: {str(e)}", exc_info=True)
                return []
                
        except Exception as e:
            logger.error(f"Unexpected error in search_similar: {str(e)}", exc_info=True)
            return []
    
    def _format_search_results(self, docs_and_scores) -> List[Dict[str, Any]]:
        """Format search results into a consistent format."""
        results = []
        for doc, score in docs_and_scores:
            try:
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                    metadata = getattr(doc, 'metadata', {})
                elif hasattr(doc, 'content'):
                    content = doc.content
                    metadata = {k: v for k, v in doc.__dict__.items() if k != 'content'}
                else:
                    content = str(doc)
                    metadata = {}
                
                result = {
                    "content": content,
                    "metadata": metadata,
                    "score": float(1.0 - score) if not isinstance(score, dict) else score.get('score', 0.0)
                }
                results.append(result)
            except Exception as e:
                logger.warning(f"Error formatting search result: {str(e)}")
                continue
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def _prepare_documents(self, text: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Split text into chunks and create document objects.
        
        Args:
            text: The text to split into chunks
            metadata: Metadata to include with each chunk
            
        Returns:
            List of Document objects
        """
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

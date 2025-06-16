import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
import re
import uuid

from langchain.docstore.document import Document

from src.models.embeddings import EmbeddingService
from src.services.llm_service import LLMService
from src.config.settings import settings
from src.utils.helpers import generate_id, format_timestamp

logger = logging.getLogger(__name__)

# In src/services/case_service.py
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import logging
from langchain.docstore.document import Document
from src.models.embeddings import EmbeddingService
from src.services.llm_service import LLMService

logger = logging.getLogger(__name__)

class CaseService:
    # Maximum length for description in characters
    MAX_DESCRIPTION_LENGTH = 100000

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service
        self.embedding_service = EmbeddingService()

    def _prepare_case_document(self, case_data: Dict[str, Any]) -> Document:
        """
        Prepare a case document for storage in the vector database.
        
        This method creates a searchable document with all relevant case information
        and ensures all fields are properly indexed for retrieval.
        
        Args:
            case_data: Dictionary containing case data
            
        Returns:
            Document: A document ready for storage in the vector database
        """
        # Extract and clean all fields
        case_task_number = case_data.get('case_task_number', '')
        parent_case = case_data.get('parent_case', '')
        issue = case_data.get('issue', '')
        root_cause = case_data.get('root_cause', '')
        resolution = case_data.get('resolution', '')
        steps_support = case_data.get('steps_support', '')
        created_at = case_data.get('created_at', datetime.utcnow().isoformat())
        updated_at = case_data.get('updated_at', created_at)
        
        # Clean text fields (remove extra whitespace, newlines, etc.)
        clean_text = lambda x: ' '.join(str(x).split()) if x else ''
        
        # Create metadata with all fields for filtering and retrieval
        metadata = {
            "case_task_number": clean_text(case_task_number),
            "parent_case": clean_text(parent_case) if parent_case else None,
            "created_at": created_at,
            "updated_at": updated_at,
            "source": "case_management_system"
        }
        
        # Create searchable text with all relevant information
        search_text = (
            f"CASE_TASK_NUMBER: {clean_text(case_task_number)}\n"
            f"PARENT_CASE: {clean_text(parent_case)}\n"
            f"ISSUE: {clean_text(issue)}\n"
            f"ROOT_CAUSE: {clean_text(root_cause)}\n"
            f"RESOLUTION: {clean_text(resolution)}\n"
            f"STEPS_SUPPORT: {clean_text(steps_support)}"
        )
        
        # Add additional fields to metadata for filtering
        metadata.update({
            "has_parent": bool(parent_case),
            "text_length": len(search_text),
            "has_resolution": bool(resolution.strip()),
            "has_steps": bool(steps_support.strip())
        })
        
        # Add any additional metadata from the case_data
        if 'metadata' in case_data and isinstance(case_data['metadata'], dict):
            for k, v in case_data['metadata'].items():
                if v is not None and k not in metadata:
                    metadata[k] = str(v)
        
        # Filter out None values from metadata
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        return Document(page_content=search_text, metadata=metadata)

    def create_case(
        self,
        case_task_number: str,
        issue: str,
        root_cause: str,
        resolution: str,
        steps_support: str,
        parent_case: Optional[str] = None,
        **metadata
    ) -> Dict[str, Any]:
        """
        Create a new case and store it in the vector database.

        Args:
            case_task_number: Unique identifier for the case task
            issue: Description of the issue
            root_cause: Analysis of the root cause
            resolution: Steps to resolve the issue
            steps_support: Detailed support steps taken
            parent_case: Optional reference to a parent case
            **metadata: Additional metadata fields

        Returns:
            Dictionary containing the created case data
        """
        # Validate required fields
        if not case_task_number or not isinstance(case_task_number, str) or not case_task_number.strip():
            raise ValueError("A valid case task number is required")
            
        if not issue or not isinstance(issue, str) or not issue.strip():
            raise ValueError("A valid issue description is required")
            
        if not root_cause or not isinstance(root_cause, str) or not root_cause.strip():
            raise ValueError("Root cause analysis is required")
            
        if not resolution or not isinstance(resolution, str) or not resolution.strip():
            raise ValueError("Resolution steps are required")
            
        if not steps_support or not isinstance(steps_support, str) or not steps_support.strip():
            raise ValueError("Support steps are required")
        
        # Check if case with this number already exists
        existing_case = self.get_case_by_task_number(case_task_number)
        if existing_case:
            raise ValueError(f"Case with task number {case_task_number} already exists")
        
        created_at = datetime.utcnow().isoformat()
        
        # Prepare the case data
        case_data = {
            "case_task_number": case_task_number.strip(),
            "parent_case": parent_case.strip() if parent_case and isinstance(parent_case, str) else None,
            "issue": issue.strip(),
            "root_cause": root_cause.strip(),
            "resolution": resolution.strip(),
            "steps_support": steps_support.strip(),
            "created_at": created_at,
            "updated_at": created_at,
            **{k: v for k, v in metadata.items() if v is not None}
        }
        
        # Create and store document in vector database
        doc = self._prepare_case_document(case_data)
        self.embedding_service.add_document(case_task_number, doc.page_content, doc.metadata)
        
        return case_data

    def get_case_by_task_number(self, case_task_number: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a case by its task number.
        
        Args:
            case_task_number: The case task number to look up
            
        Returns:
            The case data if found, None otherwise
        """
        if not case_task_number:
            return None
            
        try:
            results = self.embedding_service.vector_store.similarity_search(
                "",
                filter={"case_task_number": case_task_number},
                k=1
            )
            
            if not results:
                return None
                
            doc = results[0]
            metadata = doc.metadata
            
            return {
                "case_task_number": metadata.get("case_task_number", ""),
                "parent_case": metadata.get("parent_case"),
                "issue": metadata.get("issue", ""),
                "root_cause": metadata.get("root_cause", ""),
                "resolution": metadata.get("resolution", ""),
                "steps_support": metadata.get("steps_support", ""),
                "created_at": metadata.get("created_at", ""),
                "updated_at": metadata.get("updated_at", "")
            }
            
        except Exception as e:
            logger.error(f"Error retrieving case by task number {case_task_number}: {str(e)}", exc_info=True)
            return None

    def get_case(self, case_task_number: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a case by its case task number from the vector database.

        Args:
            case_task_number: The case task number to retrieve
            
        Returns:
            The case data if found, None otherwise
        """
        if not case_task_number:
            return None
            
        try:
            # Search for the case by case_task_number in the metadata
            results = self.embedding_service.vector_store.similarity_search(
                "",
                filter={"case_task_number": case_task_number.strip()},
                k=1
            )
            
            if not results:
                return None
                
            doc = results[0]
            metadata = doc.metadata
            
            # Parse the page content to extract the fields
            content = doc.page_content
            fields = {}
            for line in content.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    fields[key.strip().lower()] = value.strip()
            
            # Merge metadata with parsed fields, giving priority to metadata
            result = {
                "case_task_number": metadata.get("case_task_number", fields.get("case_task_number", "")),
                "parent_case": metadata.get("parent_case", fields.get("parent_case")),
                "issue": metadata.get("issue", fields.get("issue", "")),
                "root_cause": metadata.get("root_cause", fields.get("root_cause", "")),
                "resolution": metadata.get("resolution", fields.get("resolution", "")),
                "steps_support": metadata.get("steps_support", fields.get("steps_support", "")),
                "created_at": metadata.get("created_at", ""),
                "updated_at": metadata.get("updated_at", ""),
            }
            
            # Add any additional metadata fields
            for key, value in metadata.items():
                if key not in result:
                    result[key] = value
                    
            return result
                
        except Exception as e:
            logger.error(f"Error retrieving case {case_task_number}: {str(e)}", exc_info=True)
            return None

    async def generate_case_summary(
        self,
        case_task_number: str,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None
    ) -> str:
        """
        Generate a summary of a case using the LLM.
        
        Args:
            case_task_number: The case task number to summarize
            model_name: Optional model name to use
            model_path: Optional path to a local model
            
        Returns:
            str: A 4-5 sentence summary of the case
        """
        if not self.llm_service:
            return "LLM service not available for summarization."
            
        try:
            # Get the case data
            case = self.get_case_by_task_number(case_task_number)
            if not case:
                return f"Case {case_task_number} not found."
                
            # Extract the relevant fields
            issue = case.get('issue', '')
            if not issue:
                return f"No issue description available for case {case_task_number}."
                
            root_cause = case.get('root_cause', '')
            resolution = case.get('resolution', '')
            
            # Generate the summary using the LLM service
            summary = await self.llm_service.summarize_case(
                issue=issue,
                root_cause=root_cause,
                resolution=resolution
            )
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error generating case summary: {str(e)}", exc_info=True)
            return f"Error generating summary: {str(e)}"

    async def get_related_cases(self, case_task_number: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Find cases related to the given case using LLM.
        
        Args:
            case_task_number: The reference case task number
            k: Maximum number of related cases to return (1-10)
            
        Returns:
            List of dictionaries containing case_task_number, issue, and relevance_score
            
        Raises:
            ValueError: If the case is not found or LLM service is not available
        """
        # Get the reference case
        case = self.get_case(case_task_number)
        if not case:
            raise ValueError(f"Case {case_task_number} not found")
            
        # Create a search query based on the case details
        search_query = f"""
        Find cases related to:
        Issue: {case['issue']}
        Root Cause: {case['root_cause']}
        Resolution: {case['resolution'][:500]}...
        """
        
        # Use the search_cases method with the generated query
        related_cases = await self.search_cases(search_query, k=k+1, include_details=False)  # +1 to account for self
        
        # Filter out the reference case itself and limit to k results
        return [
            c for c in related_cases 
            if c['case_task_number'] != case_task_number
        ][:k]

    async def find_similar_cases(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.5,
        timeout: float = 10.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Find similar cases using vector similarity search.
        
        Args:
            query: The search query
            k: Number of results to return
            min_score: Minimum similarity score (0-1)
            timeout: Maximum time in seconds to wait for search to complete
            **kwargs: Additional search parameters
            
        Returns:
            List of Document objects with metadata and scores
        """
        try:
            import asyncio
            logger.info(f"Searching for cases similar to: '{query}' with k={k}, min_score={min_score}")
            
            # Run the search with timeout
            try:
                results = await asyncio.wait_for(
                    self._search_with_embedding(query, k, min_score, **kwargs),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"Vector search timed out after {timeout} seconds")
                return []
            
            # If no results, try with a simpler query
            if not results and len(query.split()) > 2:
                logger.info("No results found, trying with simplified query...")
                simple_query = " ".join(query.split()[:2])  # Use first two words
                try:
                    results = await asyncio.wait_for(
                        self._search_with_embedding(simple_query, k, min_score, **kwargs),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Simplified query search timed out after {timeout} seconds")
                    return []
            
            logger.info(f"Found {len(results)} similar cases")
            return results
            
        except Exception as e:
            logger.error(f"Error in find_similar_cases: {str(e)}", exc_info=True)
            return []

    async def _search_with_embedding(self, query: str, k: int, min_score: float, **kwargs):
        """Helper method to perform the actual search with embedding"""
        try:
            logger.info(f"Starting search for query: '{query}' with k={k}, min_score={min_score}")
            
            # Get the query embedding with error handling
            try:
                if hasattr(self.embedding_service.embeddings, 'embed_query'):
                    logger.debug("Using embed_query for embedding generation")
                    query_embedding = await asyncio.get_event_loop().run_in_executor(
                        None,  # Uses default ThreadPoolExecutor
                        self.embedding_service.embeddings.embed_query,
                        query
                    )
                elif hasattr(self.embedding_service.embeddings, 'encode'):
                    logger.debug("Using encode for embedding generation")
                    query_embedding = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.embedding_service.embeddings.encode(
                            query,
                            convert_to_tensor=False
                        )
                    )
                else:
                    error_msg = "Embedding model doesn't support embed_query or encode methods"
                    logger.error(error_msg)
                    return []
                
                if hasattr(query_embedding, 'tolist'):
                    query_embedding = query_embedding.tolist()
                
                logger.debug(f"Generated query embedding of length: {len(query_embedding) if query_embedding else 0}")
                
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
                return []
            
            # Get the collection and perform search with error handling
            try:
                collection = self.embedding_service.vector_store._collection
                if not collection:
                    logger.error("Failed to get collection from vector store")
                    return []
                
                logger.debug(f"Querying collection with {k} results...")
                
                # Perform the search with a timeout
                query_results = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: collection.query(
                        query_embeddings=[query_embedding],
                        n_results=min(k * 2, 20),  # Get more results to filter by score
                        include=["metadatas", "documents", "distances"],
                        **kwargs
                    )
                )
                
                if not query_results or 'ids' not in query_results or not query_results['ids'][0]:
                    logger.debug("No results found in query")
                    return []
                
                # Process results
                formatted_results = []
                for i in range(len(query_results['ids'][0])):
                    try:
                        doc_id = query_results['ids'][0][i]
                        doc_metadata = query_results['metadatas'][0][i] if query_results.get('metadatas') and query_results['metadatas'][0] else {}
                        doc_content = query_results['documents'][0][i] if query_results.get('documents') and query_results['documents'][0] else ""
                        distance = query_results['distances'][0][i] if query_results.get('distances') and query_results['distances'][0] else 1.0
                        
                        # Convert distance to similarity score (assuming cosine similarity)
                        similarity = (1.0 + (1.0 - distance)) / 2.0
                        
                        logger.debug(f"Result {i+1} - ID: {doc_id}, Distance: {distance:.4f}, Similarity: {similarity:.4f}")
                        
                        if similarity >= min_score:
                            result = {
                                "content": doc_content,
                                "metadata": doc_metadata,
                                "score": similarity,
                                "id": doc_id
                            }
                            formatted_results.append(result)
                            logger.debug(f"Added result with score: {similarity:.4f}")
                        else:
                            logger.debug(f"Skipped result with score below threshold: {similarity:.4f} < {min_score}")
                            
                    except Exception as e:
                        logger.error(f"Error processing result {i}: {str(e)}", exc_info=True)
                        continue
                
                # Sort by score and limit to k results
                formatted_results.sort(key=lambda x: x['score'], reverse=True)
                logger.info(f"Found {len(formatted_results)} results after filtering (min_score={min_score})")
                
                return formatted_results[:k]
                
            except Exception as e:
                logger.error(f"Error querying collection: {str(e)}", exc_info=True)
                return []
            
        except Exception as e:
            logger.error(f"Unexpected error in _search_with_embedding: {str(e)}", exc_info=True)
            return []

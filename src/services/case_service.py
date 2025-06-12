import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
import re

from langchain.docstore.document import Document

from src.models.embeddings import EmbeddingService
from src.services.llm_service import LLMService
from src.config.settings import settings
from src.utils.helpers import generate_id, format_timestamp

logger = logging.getLogger(__name__)

class CaseService:
    # Maximum length for description in characters (adjust as needed)
    MAX_DESCRIPTION_LENGTH = 100000  # Increased from default
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service
        self.embedding_service = EmbeddingService()
    
    def _prepare_case_document(self, case_data: Dict[str, Any]) -> Document:
        """Prepare a case document for storage in the vector database."""
        # Get and clean the description
        description = case_data.get('description', '')
        # Replace all newlines with spaces and clean up whitespace
        description = description.replace('\n', ' ').replace('\r', ' ')
        description = ' '.join(description.split())
        
        # Truncate if it exceeds max length
        if len(description) > self.MAX_DESCRIPTION_LENGTH:
            description = description[:self.MAX_DESCRIPTION_LENGTH]
            logger.warning(f"Description truncated to {self.MAX_DESCRIPTION_LENGTH} characters")
        
        # Clean close_notes as well
        close_notes = None
        if 'close_notes' in case_data and case_data['close_notes']:
            close_notes = case_data['close_notes'].replace('\n', ' ').replace('\r', ' ').strip()
        
        # Store the full description in metadata for reliable retrieval
        metadata = {
            "case_number": case_data["case_number"],
            "subject": case_data.get("subject", ""),
            "description": description,  # Store full description in metadata
            "created_at": case_data.get("created_at") or datetime.utcnow().isoformat(),
            "updated_at": case_data.get("updated_at") or datetime.utcnow().isoformat(),
            "tags": json.dumps(case_data.get("tags", [])),
            "description_length": len(description),
        }
        
        # Add close_notes to metadata if it exists
        if close_notes:
            metadata["close_notes"] = close_notes
        
        # Only add parent_case if it exists and is a non-empty string
        parent_case = case_data.get("parent_case")
        if parent_case and isinstance(parent_case, str) and parent_case.strip():
            metadata["parent_case"] = parent_case.strip()
        
        # Create searchable text from all relevant fields
        search_text = (
            f"SUBJECT: {case_data.get('subject', '')}\n"
            f"DESCRIPTION: {description}\n"
            f"CASE_NUMBER: {case_data.get('case_number', '')}\n"
            f"TAGS: {', '.join(case_data.get('tags', []))}"
        )
        
        # Add close notes if they exist
        if close_notes:
            search_text += f"\nCLOSE_NOTES: {close_notes}"
        
        # Filter out None values from metadata
        metadata = {k: v for k, v in metadata.items() if v is not None}
        
        return Document(page_content=search_text, metadata=metadata)
    
    def create_case(
        self, 
        subject: str,
        description: str,
        case_number: str,
        parent_case: Optional[str] = None,
        close_notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
        **metadata
    ) -> Dict[str, Any]:
        """
        Create a new case and store it in the vector database.
        
        Args:
            subject: Case subject/short description
            description: Detailed description of the case
            case_number: Case number (required, used as primary key)
            parent_case: Optional parent case reference
            close_notes: Optional close notes
            tags: Optional list of tags
            **metadata: Additional metadata fields
            
        Returns:
            Dictionary containing the created case data
            
        Raises:
            ValueError: If required fields are missing or case already exists
        """
        # Validate required fields
        if not case_number or not isinstance(case_number, str) or not case_number.strip():
            raise ValueError("A valid case_number is required")
            
        if not subject or not isinstance(subject, str) or not subject.strip():
            raise ValueError("A valid subject is required")
            
        if not description or not isinstance(description, str) or not description.strip():
            raise ValueError("A valid description is required")
        
        created_at = datetime.utcnow().isoformat()
        
        # Check if case with this number already exists
        existing_case = self.get_case(case_number)
        if existing_case:
            raise ValueError(f"Case with number {case_number} already exists")
        
        # Prepare the case data
        case_data = {
            "case_number": case_number.strip(),
            "subject": subject.strip(),
            "description": description.strip(),
            "parent_case": parent_case.strip() if parent_case and isinstance(parent_case, str) else None,
            "close_notes": close_notes.strip() if close_notes and isinstance(close_notes, str) else None,
            "tags": [tag.strip() for tag in tags if isinstance(tag, str) and tag.strip()] if tags else [],
            "created_at": created_at,
            "updated_at": created_at,
            **{k: v for k, v in metadata.items() if v is not None}
        }
        
        # Create and store document in vector database
        doc = self._prepare_case_document(case_data)
        self.embedding_service.add_document(case_number, doc.page_content, doc.metadata)
        
        return case_data
    
    def get_case(self, case_number: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a case by its case number from the vector database.
        
        Args:
            case_number: The case number to retrieve
            
        Returns:
            The case data if found, None otherwise
        """
        if not case_number:
            return None
            
        try:
            # Search for the case by case_number in the metadata
            results = self.embedding_service.vector_store.similarity_search(
                "",
                filter={"case_number": case_number},
                k=1
            )
            
            if not results:
                return None
                
            doc = results[0]
            metadata = doc.metadata
            
            # Get description directly from metadata where it's stored in full
            description = metadata.get("description", "")
            
            return {
                "case_number": metadata.get("case_number", ""),
                "subject": metadata.get("subject", ""),
                "description": description,
                "parent_case": metadata.get("parent_case"),
                "close_notes": metadata.get("close_notes"),
                "created_at": metadata.get("created_at", ""),
                "updated_at": metadata.get("updated_at", ""),
                "tags": json.loads(metadata.get("tags", "[]")),
            }
            
        except Exception as e:
            logger.error(f"Error retrieving case {case_number}: {str(e)}", exc_info=True)
            return None

    async def search_cases(
        self, 
        query: str, 
        k: int = 5, 
        include_details: bool = False,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None
    ) -> Union[List[Dict[str, Any]], str]:
        """
        Search for cases similar to the query.
        
        Args:
            query: Search query string
            k: Number of results to return
            include_details: Whether to include detailed LLM analysis
            model_name: Optional model name to use for search
            model_path: Optional path to a local model
            
        Returns:
            List of matching cases or formatted string with analysis
        """
        try:
            # Load the specified model if different from current
            if model_name or model_path:
                self.llm_service.load_model(
                    model_name=model_name or "default",
                    model_path=model_path
                )
                
            # Generate query embedding
            query_embedding = self.llm_service.get_embedding(query)
            
            # Find similar cases
            results = self.embedding_service.similarity_search(
                query_embedding=query_embedding,
                k=k
            )
            
            if not include_details:
                return results
                
            # Generate detailed analysis using LLM
            context = "\n".join(
                f"Case {i+1}:\n"
                f"Subject: {res['subject']}\n"
                f"Description: {res['description']}\n"
                f"Relevance Score: {res['score']:.4f}\n"
                for i, res in enumerate(results)
            )
            
            prompt = (
                f"Analyze these search results for the query: '{query}'\n\n"
                f"{context}\n\n"
                "Provide a concise analysis of the most relevant cases "
                "and their potential solutions."
            )
            
            analysis = await self.llm_service.generate_response(prompt)
            return analysis
            
        except Exception as e:
            logger.error(f"Error in case search: {str(e)}", exc_info=True)
            raise

    async def generate_case_summary(
        self,
        case_number: str,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None
    ) -> str:
        """
        Generate a concise summary of a case's description.
        
        Args:
            case_number: The case number to summarize
            model_name: Optional model name to use
            model_path: Optional path to a local model
            
        Returns:
            str: A 4-5 sentence summary of the core problem
        """
        try:
            # Get the case details
            case = self.get_case(case_number)
            if not case:
                raise ValueError(f"Case with number {case_number} not found")
                
            subject = case.get('subject', '').strip()
            description = case.get('description', '').strip()
            
            if not description:
                return f"No description available for case {case_number}."
            
            # Extract the main issue from the description
            issue_section = ""
            issue_markers = [
                "Issue Definition:",
                "Problem Description:",
                "Issue:",
                "Problem:",
                "Description:"
            ]
            
            for marker in issue_markers:
                if marker in description:
                    # Get text after the marker
                    issue_text = description.split(marker, 1)[1]
                    # Take text until the next section or end
                    issue_text = re.split(r'\n[A-Z][a-z]+:', issue_text)[0]
                    issue_section = issue_text.strip()
                    break
            
            # If no specific issue section found, use the beginning of the description
            if not issue_section:
                issue_section = ' '.join(description.split('\n')[:10])
            
            # Clean up the text
            issue_section = re.sub(r'<<[^>]*>>', '', issue_section)  # Remove template markers
            issue_section = re.sub(r'https?://\S+', '', issue_section)  # Remove URLs
            issue_section = ' '.join(issue_section.split())  # Normalize whitespace
            
            if not issue_section:
                return f"Unable to extract issue details from case {case_number}."
            
            # Prepare a focused prompt with clear instructions
            prompt = (
                "Provide a 4-5 sentence summary of the following issue. "
                "Focus on the core problem and its impact. Be specific and technical.\n\n"
                f"Subject: {subject}\n\n"
                f"Details: {issue_section[:1500]}"
            )
            
            # Generate the summary
            response = await self.llm_service.generate_response(
                prompt=prompt,
                max_length=500,
                temperature=0.3,
                top_p=0.9,
                do_sample=False,
                max_new_tokens=300
            )
            
            # Clean up the response
            summary = response.strip()
            
            # Remove any instance of the prompt in the response
            summary = summary.replace(prompt, '').strip()
            
            # Remove any remaining quotes and extra whitespace
            summary = re.sub(r'^["\']|["\']$', '', summary)
            summary = re.sub(r'\s+', ' ', summary).strip()
            
            # If the summary still looks like it contains the prompt, try to extract just the summary part
            if 'Subject:' in summary and 'Details:' in summary:
                summary = summary.split('Details:')[-1].strip()
            
            # Ensure we have a valid summary
            if not summary or len(summary.split()) < 10:
                # Fallback: return a simple summary based on the subject and first part of the issue
                return (
                    f"Issue with {subject}. "
                    f"{issue_section[:300]}"
                )
                
            return summary
            
        except Exception as e:
            logger.error(f"Error generating case summary: {str(e)}", exc_info=True)
            return f"Error generating summary: {str(e)}"

    async def get_related_cases(self, case_number: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Find cases related to the given case using LLM.
        
        Args:
            case_number: The reference case number
            k: Maximum number of related cases to return (1-10)
            
        Returns:
            List of dictionaries containing case_number, subject, and relevance_score
            
        Raises:
            ValueError: If the case is not found or LLM service is not available
        """
        # Get the reference case
        case = self.get_case(case_number)
        if not case:
            raise ValueError(f"Case {case_number} not found")
            
        # Create a search query based on the case details
        search_query = f"""
        Find cases related to:
        Subject: {case['subject']}
        Description: {case['description'][:500]}...
        """
        
        # Use the search_cases method with the generated query
        related_cases = await self.search_cases(search_query, k=k+1, include_details=False)  # +1 to account for self
        
        # Filter out the reference case itself and limit to k results
        return [
            c for c in related_cases 
            if c['case_number'] != case_number
        ][:k]

    async def find_similar_cases(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.5,  # Lowered default min_score
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Find similar cases using vector similarity search.
        
        Args:
            query: The search query
            k: Number of results to return
            min_score: Minimum similarity score (0-1)
            **kwargs: Additional search parameters
            
        Returns:
            List of Document objects with metadata and scores
        """
        try:
            logger.info(f"Searching for cases similar to: '{query}'")
            
            # First try with the original query
            results = await self._search_with_embedding(query, k, min_score, **kwargs)
            
            # If no results, try with a simpler query
            if not results and len(query.split()) > 2:
                logger.info("No results found, trying with simplified query...")
                simple_query = " ".join(query.split()[:2])  # Use first two words
                results = await self._search_with_embedding(simple_query, k, min_score, **kwargs)
            
            logger.info(f"Found {len(results)} similar cases")
            return results
            
        except Exception as e:
            logger.error(f"Error in find_similar_cases: {str(e)}", exc_info=True)
            return []

    async def _search_with_embedding(self, query: str, k: int, min_score: float, **kwargs):
        """Helper method to perform the actual search with embedding"""
        try:
            logger.info(f"Starting search for query: '{query}' with k={k}, min_score={min_score}")
            
            # Get the query embedding
            try:
                if hasattr(self.embedding_service.embeddings, 'embed_query'):
                    logger.debug("Using embed_query for embedding generation")
                    query_embedding = self.embedding_service.embeddings.embed_query(query)
                elif hasattr(self.embedding_service.embeddings, 'encode'):
                    logger.debug("Using encode for embedding generation")
                    query_embedding = self.embedding_service.embeddings.encode(
                        query,
                        convert_to_tensor=False
                    )
                else:
                    error_msg = "Embedding model doesn't support embed_query or encode methods"
                    logger.error(error_msg)
                    return []
                
                if hasattr(query_embedding, 'tolist'):
                    query_embedding = query_embedding.tolist()
                
                logger.debug(f"Generated query embedding of length: {len(query_embedding) if query_embedding else 0}")
                if query_embedding and len(query_embedding) > 0:
                    logger.debug(f"First few values: {query_embedding[:3]}...")
                
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}", exc_info=True)
                return []
            
            # Get the collection directly
            try:
                collection = self.embedding_service.vector_store._collection
                logger.debug(f"Collection info - Name: {collection.name}, Count: {collection.count()}")
                
                # Query the collection with the pre-computed embedding
                logger.debug("Querying collection...")
                query_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(k * 2, 20),  # Get more results to filter by score
                    include=["metadatas", "documents", "distances"],
                    **kwargs
                )
                
                logger.debug(f"Query results keys: {query_results.keys() if query_results else 'None'}")
                if query_results and 'ids' in query_results:
                    logger.debug(f"Found {len(query_results['ids'][0])} results before filtering")
                
                # Process results
                formatted_results = []
                if query_results and 'ids' in query_results and query_results['ids'][0]:
                    for i in range(len(query_results['ids'][0])):
                        try:
                            doc_id = query_results['ids'][0][i]
                            doc_metadata = query_results['metadatas'][0][i] if query_results.get('metadatas') and query_results['metadatas'][0] else {}
                            doc_content = query_results['documents'][0][i] if query_results.get('documents') and query_results['documents'][0] else ""
                            distance = query_results['distances'][0][i] if query_results.get('distances') and query_results['distances'][0] else 1.0
                            
                            # Convert distance to similarity score (assuming cosine similarity)
                            # For cosine similarity, the range is [-1, 1], so we normalize to [0, 1]
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

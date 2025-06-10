import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, AsyncGenerator

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
        min_score: float = 0.7
    ) -> Union[List[Dict[str, Any]], str]:
        """
        Search for cases similar to the query using vector similarity search.
        
        Args:
            query: The search query
            k: Maximum number of results to return (1-20)
            include_details: If True, returns a natural language response with case references
            min_score: Minimum similarity score (0-1) for results to be included
            
        Returns:
            If include_details is False: List of case results with scores
            If include_details is True: Natural language response with case references
            
        Raises:
            ValueError: If LLM service is not available for include_details=True
        """
        try:
            # First, try vector similarity search
            vector_results = self.embedding_service.vector_store.similarity_search_with_score(
                query,
                k=min(k * 2, 20)  # Get more results to filter by score
            )
            
            # Filter results by score and format
            results = []
            seen_cases = set()
            
            for doc, score in vector_results:
                if score > min_score:
                    continue
                    
                metadata = doc.metadata
                case_number = metadata.get('case_number')
                
                # Skip if we've already seen this case (to avoid duplicates)
                if case_number in seen_cases:
                    continue
                    
                seen_cases.add(case_number)
                
                # Get full case details
                case = self.get_case(case_number)
                if case:
                    results.append({
                        **case,
                        'relevance_score': 1.0 - score,  # Convert distance to similarity score
                        'match_reason': f"Vector similarity: {1.0 - score:.2f}"
                    })
                
                if len(results) >= k:
                    break
            
            # If no results from vector search, try keyword search as fallback
            if not results:
                logger.warning(f"No vector search results for query: {query}")
                
                # Get all cases and do simple keyword matching
                all_cases = self.embedding_service.get_all_cases()
                for case in all_cases:
                    metadata = case.get('metadata', {})
                    case_number = metadata.get('case_number')
                    
                    if not case_number or case_number in seen_cases:
                        continue
                        
                    seen_cases.add(case_number)
                    case_details = self.get_case(case_number)
                    
                    if not case_details:
                        continue
                    
                    # Simple keyword matching
                    text_to_search = f"{case_details.get('subject', '')} {case_details.get('description', '')}".lower()
                    query_terms = query.lower().split()
                    
                    # Count matching terms
                    matches = sum(1 for term in query_terms if term in text_to_search)
                    if matches > 0:
                        results.append({
                            **case_details,
                            'relevance_score': min(0.7, matches * 0.1),  # Cap at 0.7 for keyword matches
                            'match_reason': f"Keyword match: {matches} terms"
                        })
                    
                    if len(results) >= k:
                        break
            
            # Sort results by relevance score (highest first)
            results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
            
            # If include_details is True, generate a natural language response
            if include_details and self.llm_service:
                return await self._generate_search_response(query, results[:k])
                
            return results[:k]
            
        except Exception as e:
            logger.error(f"Error in search_cases: {str(e)}", exc_info=True)
            if include_details:
                return f"An error occurred while searching: {str(e)}"
            return []
    
    async def _generate_search_response(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generate a concise natural language response for search results."""
        if not results:
            return "I couldn't find any relevant cases matching your query."
            
        # Get the most relevant case
        most_relevant = results[0]
        
        prompt = f"""
        You are a helpful support assistant. Provide a concise response (max 5 lines) 
        to the user's query based on the following case information.
        
        User Query: "{query}"
        
        Most Relevant Case:
        - Subject: {most_relevant.get('subject', 'No subject')}
        - Description: {most_relevant.get('description', 'No description available')[:500]}
        
        Provide a brief, helpful response that addresses the user's query using the case information.
        Focus on the key points and keep it concise (max 5 lines).
        """
        
        try:
            response = await self.llm_service.generate(prompt)
            # Ensure the response is exactly 5 lines
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            return '\n'.join(lines[:5])
        except Exception as e:
            logger.error(f"Error generating search response: {str(e)}")
            # Fall back to a simple response if LLM fails
            return f"Based on case {most_relevant.get('case_number', 'N/A')}: {most_relevant.get('subject', 'No subject')}"
    
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
    
    async def generate_case_summary(self, case_number: str) -> str:
        """
        Generate a summary of a case using LLM.
        
        Args:
            case_number: The case number to summarize
            
        Returns:
            Generated summary text
            
        Raises:
            ValueError: If LLM service is not available or case not found
        """
        if not self.llm_service:
            raise ValueError("LLM service is required for summarization")
            
        case = self.get_case(case_number)
        if not case:
            raise ValueError(f"Case {case_number} not found")
            
        prompt = f"""
        Please provide a concise summary of the following case:
        
        Case Number: {case['case_number']}
        Subject: {case['subject']}
        Description: {case['description']}
        
        Summary:
        """
        
        return await self.llm_service.generate(prompt)

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
        
        # Create metadata with only non-None values
        metadata = {
            "case_number": case_data["case_number"],
            "subject": case_data.get("subject", ""),
            "created_at": case_data.get("created_at") or datetime.utcnow().isoformat(),
            "updated_at": case_data.get("updated_at") or datetime.utcnow().isoformat(),
            "tags": json.dumps(case_data.get("tags", [])),
            "description_length": len(description),
        }
        
        # Add optional fields if they exist and are not None
        if close_notes:
            metadata["close_notes"] = close_notes
        
        # Only add parent_case if it exists and is a non-empty string
        parent_case = case_data.get("parent_case")
        if parent_case and isinstance(parent_case, str) and parent_case.strip():
            metadata["parent_case"] = parent_case.strip()
        
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
            
            # Get all available data from metadata first
            metadata = doc.metadata
            
            # Try to get description from metadata if available
            description = metadata.get("description", "")
            
            # If not in metadata, try to extract from page_content
            if not description and doc.page_content:
                if "DESCRIPTION:" in doc.page_content:
                    # Extract content after DESCRIPTION: until the next section or end
                    desc_parts = doc.page_content.split("DESCRIPTION:", 1)
                    if len(desc_parts) > 1:
                        remaining = desc_parts[1]
                        # Look for the next section header (all caps followed by colon)
                        import re
                        match = re.search(r'\n[A-Z_]+:', remaining)
                        if match:
                            description = remaining[:match.start()].strip()
                        else:
                            description = remaining.strip()
                else:
                    # Fallback to the full page_content if we can't find the DESCRIPTION marker
                    description = doc.page_content.strip()
            
            # Clean up the description
            if description:
                # Replace any remaining newlines with spaces
                description = ' '.join(description.split())
            
            # Get subject from metadata first, fallback to extraction from page_content
            subject = metadata.get("subject", "")
            if not subject and "SUBJECT:" in doc.page_content:
                subject_parts = doc.page_content.split("SUBJECT:", 1)
                if len(subject_parts) > 1:
                    subject = subject_parts[1].split("\n")[0].strip()
            
            return {
                "case_number": metadata.get("case_number", ""),
                "subject": subject,
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

    async def search_cases(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for cases similar to the query using LLM.
        
        Args:
            query: The search query
            k: Maximum number of results to return (1-10)
            
        Returns:
            List of dictionaries containing case_number, subject, and relevance_score
            
        Raises:
            ValueError: If LLM service is not available
        """
        if not self.llm_service:
            raise ValueError("LLM service is required for semantic search")
        
        # Get all cases first
        all_cases = [self.get_case(case_number) for case_number in self.embedding_service.vector_store.get_all_ids()]
        all_cases = [case for case in all_cases if case]
        if not all_cases:
            return []
            
        # Format cases for the prompt
        cases_info = []
        for case in all_cases:
            cases_info.append(f"Case Number: {case['case_number']} - Subject: {case['subject']}")
        
        cases_text = "\n".join(cases_info)
        
        # Create prompt for the LLM
        prompt = f"""
        You are a helpful assistant that finds relevant cases based on a search query.
        
        Here are the available cases:
        {cases_text}
        
        Search query: "{query}"
        
        Please return the top {k} most relevant case numbers and their subjects.
        For each case, provide:
        1. A relevance score between 0.0 and 1.0 (1.0 being most relevant)
        2. The case number
        3. A very brief reason for the match (5-10 words)
        
        Format your response as a JSON array of objects with these fields:
        - case_number: The case number
        - subject: The case subject
        - relevance_score: A float between 0.0 and 1.0
        - match_reason: A very brief explanation of why this case matches
        
        Example:
        [
            {{
                "case_number": "CASE-123",
                "subject": "Server Outage",
                "relevance_score": 0.95,
                "match_reason": "Matches server downtime issue"
            }}
        ]
        
        Now, provide the top {k} most relevant cases for the query "{query}":
        """
        
        try:
            # Get response from LLM
            response = await self.llm_service.generate(prompt)
            
            # Parse the JSON response
            try:
                results = json.loads(response)
                if not isinstance(results, list):
                    results = [results]
                    
                # Ensure all required fields are present
                valid_results = []
                for result in results:
                    if all(field in result for field in ["case_number", "subject", "relevance_score", "match_reason"]):
                        valid_results.append({
                            "case_number": str(result["case_number"]).strip(),
                            "subject": str(result["subject"]).strip(),
                            "relevance_score": float(result["relevance_score"]),
                            "match_reason": str(result["match_reason"]).strip()
                        })
                
                # Sort by relevance score (highest first)
                valid_results.sort(key=lambda x: x["relevance_score"], reverse=True)
                
                return valid_results[:k]
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"Error parsing LLM response: {str(e)}\nResponse: {response}")
                raise ValueError("Failed to process search results. Please try again.")
                
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            raise ValueError(f"Search failed: {str(e)}")
    
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
        related_cases = await self.search_cases(search_query, k=k+1)  # +1 to account for self
        
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

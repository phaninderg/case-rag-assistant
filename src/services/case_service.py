import os
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, AsyncGenerator

from src.models.embeddings import EmbeddingService
from src.services.llm_service import LLMService
from src.config.settings import settings
from src.utils.helpers import generate_id, format_timestamp, save_json, load_json

logger = logging.getLogger(__name__)

class CaseService:
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service
        self.embedding_service = EmbeddingService()
        self.case_data_dir = Path(settings.case_data_path)
        self.case_data_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_case_path(self, case_id: str) -> Path:
        """Get the file path for a case by ID."""
        return self.case_data_dir / f"{case_id}.json"
    
    def create_case(self, title: str, description: str, tags: List[str] = None, **metadata) -> Dict[str, Any]:
        """Create a new case and store it in the vector database."""
        case_id = generate_id()
        created_at = format_timestamp(datetime.utcnow())
        
        case_data = {
            "case_id": case_id,
            "title": title,
            "description": description,
            "tags": tags or [],
            "created_at": created_at,
            "updated_at": created_at,
            **metadata
        }
        
        # Save case data to JSON file
        save_json(self._get_case_path(case_id), case_data)
        
        # Add to vector store
        self.embedding_service.add_document(
            case_id=case_id,
            text=description,
            metadata={
                "title": title,
                "tags": ",".join(tags) if tags else "",
                "created_at": created_at
            }
        )
        
        return case_data
    
    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a case by its ID."""
        case_path = self._get_case_path(case_id)
        if not case_path.exists():
            return None
            
        return load_json(case_path)
    
    def search_cases(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for cases similar to the query."""
        results = self.embedding_service.search_similar(query, k=limit)
        
        cases = []
        for result in results:
            case_id = result["metadata"].get("case_id")
            if case_id:
                case = self.get_case(case_id)
                if case:
                    cases.append({
                        **case,
                        "similarity_score": result.get("score", 0.0)
                    })
        
        return cases
    
    async def search_similar_cases(
        self,
        query: str,
        k: int = 5,
        threshold: float = 0.7,
        include_analysis: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for cases similar to the query using semantic search.
        
        Args:
            query: The search query
            k: Maximum number of results to return
            threshold: Minimum similarity score (0.0 to 1.0)
            include_analysis: Whether to include AI analysis of results
            
        Returns:
            List of matching cases with similarity scores and optional analysis
        """
        # First, get similar cases using the embedding service
        similar_docs = self.embedding_service.search_similar(query, k=k)
        
        # Filter by threshold and format results
        results = []
        for doc in similar_docs:
            if doc.get('score', 0) < threshold:
                continue
                
            case_id = doc['metadata'].get('case_id')
            if not case_id:
                continue
                
            case = self.get_case(case_id)
            if not case:
                continue
                
            result = {
                **case,
                'similarity_score': doc.get('score', 0),
                'metadata': doc.get('metadata', {})
            }
            
            # Add AI analysis if requested
            if include_analysis and self.llm_service:
                try:
                    analysis = await self._analyze_search_result(query, case)
                    result['analysis'] = analysis
                except Exception as e:
                    logger.error(f"Error analyzing search result: {str(e)}")
            
            results.append(result)
        
        return results
    
    async def _analyze_search_result(self, query: str, case: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI analysis of why a case is relevant to the query."""
        if not self.llm_service:
            return {"error": "LLM service not available for analysis"}
            
        prompt = f"""Analyze why this case is relevant to the query "{query}".
        
        Case Title: {case.get('title', 'N/A')}
        Case Description: {case.get('description', 'N/A')}
        
        Provide a brief analysis of the relevance in 1-2 sentences."""
        
        try:
            response = await self.llm_service.generate(prompt)
            return {"relevance_analysis": response.strip()}
        except Exception as e:
            logger.error(f"Error generating analysis: {str(e)}")
            return {"error": "Failed to generate analysis"}
    
    def list_cases(
        self,
        limit: int = 10,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ) -> Dict[str, Any]:
        """
        List all cases with pagination and sorting.
        
        Args:
            limit: Maximum number of cases to return
            offset: Number of cases to skip
            sort_by: Field to sort by
            sort_order: Sort order ('asc' or 'desc')
            
        Returns:
            Dictionary containing the list of cases and pagination metadata
        """
        # Get all case files
        case_files = list(self.case_data_dir.glob("*.json"))
        
        # Load and sort cases
        cases = []
        for file_path in case_files:
            case_data = load_json(file_path)
            cases.append({
                "case_id": case_data["case_id"],
                "title": case_data["title"],
                "description": case_data.get("description", ""),
                "created_at": case_data["created_at"],
                "updated_at": case_data.get("updated_at", case_data["created_at"]),
                "tags": case_data.get("tags", [])
            })
        
        # Sort cases
        reverse_sort = (sort_order.lower() == "desc")
        try:
            cases.sort(key=lambda x: x.get(sort_by, ""), reverse=reverse_sort)
        except (KeyError, TypeError):
            # Fallback to default sorting if the sort_by field is invalid
            cases.sort(key=lambda x: x["created_at"], reverse=reverse_sort)
        
        # Apply pagination
        total_cases = len(cases)
        paginated_cases = cases[offset:offset + limit]
        
        return {
            "items": paginated_cases,
            "total": total_cases,
            "offset": offset,
            "limit": limit
        }

from fastapi import FastAPI, HTTPException, status, Query, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any, Union
import logging
import json
import uuid
from datetime import datetime

# Update imports to use absolute paths
from src.config.settings import settings
from src.services.case_service import CaseService
from src.services.llm_service import LLMService
from src.models.embeddings import EmbeddingService

# Configure logging
logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Case RAG Assistant API",
    description="API for managing and searching case tasks using RAG with multiple LLM providers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
llm_service = LLMService()
case_service = CaseService(llm_service=llm_service)

# Request/Response Models
class CaseCreate(BaseModel):
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=10)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CaseResponse(CaseCreate):
    case_id: str
    created_at: str
    updated_at: str
    
    class Config:
        from_attributes = True

class CaseSearchRequest(BaseModel):
    query: str
    k: int = Field(5, ge=1, le=20)
    threshold: float = Field(0.7, ge=0.0, le=1.0)
    include_analysis: bool = True

class CaseSearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class CaseSummaryRequest(BaseModel):
    case_id: str
    include_similar: bool = True
    similar_k: int = Field(3, ge=1, le=5)

class LLMConfigUpdate(BaseModel):
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    streaming: Optional[bool] = None

# API Endpoints
@app.post("/api/cases", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
async def create_case(case_data: CaseCreate):
    """Create a new case."""
    try:
        case = case_service.create_case(
            title=case_data.title,
            description=case_data.description,
            tags=case_data.tags,
            **case_data.metadata
        )
        return case
    except Exception as e:
        logger.error(f"Error creating case: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create case: {str(e)}"
        )

@app.get("/api/cases/{case_id}", response_model=Dict[str, Any])
async def get_case(case_id: str):
    """Get a case by ID with additional metadata."""
    case = case_service.get_case(case_id)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case with ID {case_id} not found"
        )
    return case

@app.get("/api/cases", response_model=Dict[str, Any])
async def list_cases(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("created_at", regex="^[a-zA-Z0-9_]+$"),
    sort_order: str = Query("desc", regex="^(asc|desc)$")
):
    """List all cases with pagination and sorting."""
    return case_service.list_cases(
        limit=limit,
        offset=offset,
        sort_by=sort_by,
        sort_order=sort_order
    )

@app.post("/api/cases/search", response_model=CaseSearchResponse)
async def search_cases(search_request: CaseSearchRequest):
    """Search for cases similar to the query."""
    try:
        results = await case_service.search_similar_cases(
            query=search_request.query,
            k=search_request.k,
            threshold=search_request.threshold,
            include_analysis=search_request.include_analysis
        )
        
        return {
            "results": results,
            "metadata": {
                "query": search_request.query,
                "result_count": len(results),
                "threshold": search_request.threshold
            }
        }
    except Exception as e:
        logger.error(f"Error searching cases: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@app.post("/api/cases/{case_id}/summarize")
async def summarize_case(
    case_id: str,
    request: CaseSummaryRequest
):
    """Generate a summary of a case using the LLM."""
    try:
        summary = await case_service.generate_case_summary(
            case_id=case_id,
            include_similar=request.include_similar,
            similar_k=request.similar_k
        )
        return summary
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate summary: {str(e)}"
        )

@app.post("/api/llm/config")
async def update_llm_config(config: LLMConfigUpdate):
    """Update the LLM configuration."""
    try:
        global llm_service, case_service
        
        # Create new LLM service with updated config
        new_llm_service = LLMService(
            model_name=config.model_name or llm_service.model_name,
            temperature=config.temperature if config.temperature is not None else llm_service.llm.temperature,
            max_tokens=config.max_tokens if config.max_tokens is not None else llm_service.llm.max_tokens,
            streaming=config.streaming if config.streaming is not None else llm_service.streaming
        )
        
        # Update services
        llm_service = new_llm_service
        case_service.llm_service = new_llm_service
        
        return {
            "status": "success",
            "model": llm_service.model_name,
            "config": {
                "temperature": llm_service.llm.temperature,
                "max_tokens": llm_service.llm.max_tokens,
                "streaming": llm_service.streaming
            }
        }
    except Exception as e:
        logger.error(f"Error updating LLM config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid configuration: {str(e)}"
        )

@app.get("/api/llm/models")
async def list_available_models():
    """List all available LLM and embedding models."""
    from src.config.models import DEFAULT_LLM_MODELS, DEFAULT_EMBEDDING_MODELS
    
    return {
        "llm_models": list(DEFAULT_LLM_MODELS.keys()),
        "embedding_models": list(DEFAULT_EMBEDDING_MODELS.keys())
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "models": {
            "llm": llm_service.model_name,
            "embedding": case_service.embedding_service.model_name
        },
        "timestamp": datetime.utcnow().isoformat()
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.web_concurrency
    )

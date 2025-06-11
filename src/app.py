from fastapi import FastAPI, HTTPException, status, Query, Body, Request, UploadFile, File
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
from src.services.training_service import TrainingService

# Configure logging
logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Case RAG Assistant API",
    description="API for managing and searching cases using RAG with vector database",
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
training_service = TrainingService()

# Request/Response Models
class CaseCreate(BaseModel):
    subject: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=10)
    case_number: str = Field(..., min_length=1, max_length=50)
    parent_case: Optional[str] = None
    close_notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CaseResponse(CaseCreate):
    created_at: str
    updated_at: str
    
    class Config:
        from_attributes = True

class CaseSearchRequest(BaseModel):
    query: str
    k: int = Field(5, ge=1, le=20)
    include_details: bool = False

class CaseSearchResponse(BaseModel):
    results: Union[List[Dict[str, Any]], str]
    metadata: Dict[str, Any]

class CaseSummaryRequest(BaseModel):
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
            subject=case_data.subject,
            description=case_data.description,
            case_number=case_data.case_number,
            parent_case=case_data.parent_case,
            close_notes=case_data.close_notes,
            tags=case_data.tags,
            **case_data.metadata
        )
        return case
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating case: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create case: {str(e)}"
        )

@app.get("/api/cases/{case_number}", response_model=Dict[str, Any])
async def get_case(case_number: str):
    """Get a case by case number."""
    case = case_service.get_case(case_number)
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Case with number {case_number} not found"
        )
    return case

@app.post("/api/cases/search", response_model=CaseSearchResponse)
async def search_cases(search_request: CaseSearchRequest):
    """Search for cases similar to the query."""
    try:
        response = await case_service.search_cases(
            query=search_request.query,
            k=search_request.k,
            include_details=search_request.include_details
        )
        
        # If include_details is True, the response is already a string from the LLM
        if search_request.include_details:
            return {
                "results": response,
                "metadata": {
                    "query": search_request.query,
                    "result_count": 1 if response != "I couldn't find any relevant cases matching your query." else 0
                }
            }
        
        # Otherwise, it's a list of results
        return {
            "results": response,
            "metadata": {
                "query": search_request.query,
                "result_count": len(response)
            }
        }
    except Exception as e:
        logger.error(f"Error searching cases: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )

@app.post("/api/cases/{case_number}/summarize", response_model=str)
async def summarize_case(
    case_number: str,
    request: Optional[CaseSummaryRequest] = None
):
    """Generate a summary of a case using the LLM.
    
    Returns:
        str: The generated summary as plain text
    """
    try:
        if request is None:
            request = CaseSummaryRequest()
            
        summary = await case_service.generate_case_summary(case_number)
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
            "message": "LLM configuration updated",
            "config": {
                "model_name": llm_service.model_name,
                "temperature": llm_service.llm.temperature,
                "max_tokens": llm_service.llm.max_tokens,
                "streaming": llm_service.streaming
            }
        }
    except Exception as e:
        logger.error(f"Error updating LLM config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update LLM config: {str(e)}"
        )

@app.get("/api/models")
async def list_available_models():
    """List all available LLM and embedding models."""
    try:
        return {
            "llm_models": ["gpt-3.5-turbo", "gpt-4", "claude-2"],
            "embedding_models": ["text-embedding-ada-002", "all-mpnet-base-v2"]
        }
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list available models"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        return {
            "status": "healthy",
            "version": "1.0.0",
            "services": {
                "llm": "operational" if llm_service else "unavailable",
                "vector_db": "operational" if hasattr(case_service, 'embedding_service') else "unavailable"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service unavailable"
        )

@app.post("/train", tags=["training"])
async def train_model(
    cases: Optional[List[Dict[str, Any]]] = Body(None),
    cases_file: UploadFile = File(None),
    output_dir: str = "./trained_models/case_model"
):
    """
    Train a model on case data.
    
    Args:
        cases: List of case dictionaries containing training data
        cases_file: JSON file containing cases (alternative to cases parameter)
        output_dir: Directory to save the trained model
    """
    try:
        training_data = cases
        
        # If no cases provided directly, check for file upload
        if not training_data and cases_file:
            try:
                content = await cases_file.read()
                data = json.loads(content)
                # Handle both direct cases array and metadata wrapper format
                training_data = data.get('cases', data) if isinstance(data, dict) else data
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid JSON file"
                )
        
        if not training_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No training data provided. Either 'cases' or 'cases_file' must be provided."
            )
            
        training_service.train_model(training_data, output_dir)
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Model training completed. Saved to {output_dir}",
                "training_samples": len(training_data)
            },
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )

@app.post("/load-model", tags=["training"])
async def load_trained_model(
    model_path: str
):
    """
    Load a previously trained model.
    
    Args:
        model_path: Path to the trained model directory
    """
    try:
        training_service.load_trained_model(model_path)
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Model loaded from {model_path}"
            },
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )

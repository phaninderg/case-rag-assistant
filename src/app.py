from fastapi import FastAPI, HTTPException, status, Query, Body, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any, Union
import logging
import json
import uuid
from datetime import datetime
from pathlib import Path
import pandas as pd
import io

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
case_service = CaseService()
llm_service = LLMService(case_service=case_service)
case_service.llm_service = llm_service  # Set up circular dependency

# Initialize training service with lazy loading
_training_service = None
def get_training_service():
    global _training_service
    if _training_service is None:
        from src.config.models import DEFAULT_LLM
        _training_service = TrainingService(base_model=DEFAULT_LLM, llm_service=llm_service)
        logger.info(f"Initialized training service with model: {DEFAULT_LLM}")
    return _training_service

# Request/Response Models
# In src/app.py
class CaseCreate(BaseModel):
    """
    Model for creating a new case with the following fields from case_task.csv:
    - case_task_number: Unique identifier for the case task
    - parent_case: Reference to a parent case (optional)
    - issue: Description of the issue
    - root_cause: Analysis of the root cause
    - resolution: Steps to resolve the issue
    - steps_support: Detailed support steps taken
    """
    case_task_number: str = Field(..., min_length=1, max_length=50, description="Unique identifier for the case task")
    parent_case: Optional[str] = Field(None, description="Reference to a parent case (optional)")
    issue: str = Field(..., min_length=3, description="Description of the issue")
    root_cause: str = Field(..., description="Analysis of the root cause")
    resolution: str = Field(..., description="Steps to resolve the issue")
    steps_support: str = Field(..., description="Detailed support steps taken")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class CaseResponse(CaseCreate):
    created_at: str
    updated_at: str
    
    class Config:
        from_attributes = True

class CaseSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    k: int = Field(5, ge=1, le=20, description="Maximum number of results to return")
    min_score: float = Field(0.6, ge=0.0, le=1.0, description="Minimum similarity score (0-1)")
    include_solutions: bool = Field(True, description="Whether to include AI-generated solutions")
    model_name: Optional[str] = Field(None, description="Optional model name to use for search")
    model_path: Optional[str] = Field(None, description="Optional path to a local model")

class SearchResult(BaseModel):
    solution: Optional[str] = Field(None, description="The generated solution text or case content")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score between query and result (0-1)")
    case_number: str = Field(..., description="The case number for reference")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional case metadata")

class SearchResponse(BaseModel):
    """Response model for case search results."""
    results: Optional[List[SearchResult]] = Field(
        None, 
        description="List of search results (when include_solutions is False)"
    )
    ai_summary: Optional[str] = Field(
        None, 
        description="AI-generated summary (when include_solutions is True)"
    )
    query: str = Field(..., description="The original search query")
    total_results: int = Field(0, description="Total number of results found")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional metadata about the search"
    )

class CaseSummaryRequest(BaseModel):
    model_name: Optional[str] = None
    model_path: Optional[str] = None

class TrainModelRequest(BaseModel):
    """
    Request model for training a new model.
    
    The CSV file should contain the following columns:
    - issue: Description of the issue
    - root_cause: Analysis of the root cause
    - resolution: Steps to resolve the issue
    - steps_support: Detailed support steps taken
    """
    cases_file: UploadFile = File(..., description="CSV file containing case tasks with columns: issue, root_cause, resolution, steps_support")
    output_dir: str = "./trained_models/case_model"
    epochs: int = 3
    learning_rate: float = 2e-5

class TrainRequest(BaseModel):
    output_dir: str = "./trained_models/case_model"
    epochs: int = 3
    learning_rate: float = 2e-5

class ChatMessage(BaseModel):
    role: str  # 'system', 'user', or 'assistant'
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 1000
    stream: bool = False

class ChatResponse(BaseModel):
    message: ChatMessage
    usage: Dict[str, int] = {
        'prompt_tokens': 0,
        'completion_tokens': 0,
        'total_tokens': 0
    }

# API Endpoints
@app.post("/api/cases", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
async def create_case(case_data: CaseCreate):
    """Create a new case."""
    try:
        case = case_service.create_case(
            case_task_number=case_data.case_task_number,
            parent_case=case_data.parent_case,
            issue=case_data.issue,
            root_cause=case_data.root_cause,
            resolution=case_data.resolution,
            steps_support=case_data.steps_support,
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

@app.post("/api/cases/search", response_model=SearchResponse)
async def search_cases(search_request: CaseSearchRequest):
    """
    Search for similar cases and provide potential solutions.
    
    When include_solutions is True:
    - Returns a list of case task numbers in the 'case_task_numbers' field
    - Includes an AI-generated summary in the 'ai_summary' field
    
    When include_solutions is False:
    - Returns detailed case information in the 'results' field
    """
    try:
        # Perform the search
        search_results = await llm_service.search_similar_cases(
            query=search_request.query,
            k=search_request.k,
            min_score=search_request.min_score,
            include_solutions=search_request.include_solutions
        )
        
        # Prepare response based on include_solutions flag
        if search_request.include_solutions:
            # For AI solutions, we expect a dict with ai_summary
            return SearchResponse(
                ai_summary=search_results.get("ai_summary", "No summary available."),
                query=search_request.query,
                total_results=search_results.get("total_results", 0),
                metadata={
                    "model": llm_service.model_name,
                    "include_solutions": True,
                    "min_score": search_request.min_score
                }
            )
        else:
            # For regular search, we expect a list of results
            return SearchResponse(
                results=search_results,
                query=search_request.query,
                total_results=len(search_results),
                metadata={
                    "model": llm_service.model_name,
                    "include_solutions": False,
                    "min_score": search_request.min_score
                }
            )
            
    except Exception as e:
        logger.error(f"Search failed: {str(e)}", exc_info=True)
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
    
    Returns a concise 4-5 sentence summary focusing on the core problem.
    """
    try:
        if request is None:
            request = CaseSummaryRequest()
            
        summary = await case_service.generate_case_summary(
            case_task_number=case_number,  # Changed from case_number to case_task_number
            model_name=request.model_name, 
            model_path=request.model_path
        )
        return summary
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate summary: {str(e)}"
        )

@app.get("/api/vector-store/status")
async def get_vector_store_status():
    """
    Get information about the vector store status.
    
    Returns:
        Dict with vector store information
    """
    try:
        embedding_service = EmbeddingService()
        status_info = embedding_service.get_vector_store_info()
        
        # Add additional debug info
        status_info.update({
            "embedding_model": embedding_service.model_name,
            "embedding_config": str(embedding_service.embedding_config),
            "is_embedding_model_loaded": hasattr(embedding_service.embeddings, 'model') or hasattr(embedding_service.embeddings, 'client')
        })
        
        return status_info
        
    except Exception as e:
        logger.error(f"Error getting vector store status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting vector store status: {str(e)}"
        )

@app.get("/api/models")
async def list_available_models():
    """List all available LLM and embedding models."""
    try:
        from src.config.models import DEFAULT_LLM_MODELS, DEFAULT_EMBEDDING_MODELS
        
        return {
            "llm_models": list(DEFAULT_LLM_MODELS.keys()),
            "embedding_models": list(DEFAULT_EMBEDDING_MODELS.keys())
        }
    except Exception as e:
        logger.error(f"Error listing available models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list available models: {str(e)}"
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
    request: TrainModelRequest
):
    """
    Train a model on case task data with instruction fine-tuning.
    
    Args:
        cases_file: CSV file containing case tasks
        output_dir: Directory to save the trained model
        epochs: Number of training epochs
        learning_rate: Learning rate for training
    """
    try:
        # Read and parse CSV file
        contents = await request.cases_file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Convert DataFrame to list of dictionaries
        cases = df.to_dict('records')
        
        # Train the model
        training_service = get_training_service()
        output_path = training_service.train(
            cases=cases,
            output_dir=request.output_dir,
            num_train_epochs=request.epochs,
            learning_rate=request.learning_rate
        )
        
        return {
            "status": "success",
            "message": "Model trained successfully",
            "model_path": output_path,
            "training_samples": len(cases)
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )

@app.post("/load-model", tags=["training"])
async def load_trained_model(
    model_path: str = Body(..., embed=True, description="Path to the trained model directory")
):
    """
    Load a previously trained model.
    
    Args:
        model_path: Path to the trained model directory
    """
    try:
        # Verify the model path exists
        model_dir = Path(model_path)
        if not model_dir.exists() or not model_dir.is_dir():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model directory not found: {model_path}"
            )
            
        # Load the model using the LLMService
        training_service = get_training_service()
        training_service.load_trained_model(model_path)
        llm_service.load_model(model_path)
        
        return {
            "status": "success",
            "message": f"Model loaded from {model_path}",
            "model_path": model_path
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}"
        )

@app.post("/api/train", status_code=200)
async def train_model(
    cases_file: UploadFile = File(..., description="CSV file containing case tasks"),
    output_dir: str = Form("./trained_models/case_model"),
    epochs: int = Form(3),
    learning_rate: float = Form(2e-5)
):
    """
    Train a new model on the provided cases.
    
    Args:
        cases_file: CSV file containing case tasks
        output_dir: Directory to save the trained model
        epochs: Number of training epochs
        learning_rate: Learning rate for training
        
    Returns:
        dict: Status and path to the trained model
    """
    try:
        # Read and parse CSV file
        contents = await cases_file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Convert DataFrame to list of dictionaries
        cases = df.to_dict('records')
        
        # Train the model
        training_service = get_training_service()
        output_path = training_service.train(
            cases=cases,
            output_dir=output_dir,
            num_train_epochs=epochs,
            learning_rate=learning_rate
        )
        
        return {
            "status": "success",
            "message": "Model trained successfully",
            "model_path": output_path,
            "training_samples": len(cases)
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {str(e)}"
        )

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Handle chat completion requests with the LLM.
    """
    logger.info("Chat endpoint called")
    try:
        # Log the incoming request
        logger.info(f"Received chat request with {len(request.messages)} messages")
        for i, msg in enumerate(request.messages):
            logger.debug(f"Message {i + 1}: {msg.role} - {msg.content[:100]}{'...' if len(msg.content) > 100 else ''}")
        
        # Convert messages to the format expected by the LLM service
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        logger.info(f"Calling LLM service with temperature={request.temperature}, max_tokens={request.max_tokens}")
        
        # Generate response using the LLM service
        response = await llm_service.generate_response(
            messages=messages,
            temperature=request.temperature,
            max_length=request.max_tokens,
            stream=request.stream
        )
        
        logger.info("Successfully generated response from LLM")
        logger.debug(f"Response content: {response.get('content', '')[:200]}...")
        
        # Format the response
        result = {
            "message": {
                "role": "assistant",
                "content": response.get('content', '')
            },
            "usage": response.get('usage', {
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_tokens': 0
            })
        }
        
        logger.info(f"Returning response with {len(result['message']['content'])} characters")
        return result
        
    except Exception as e:
        error_msg = f"Chat error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating chat response: {str(e)}"
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

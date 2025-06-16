# In src/config/settings.py
from pydantic import BaseSettings
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv
from .models import ModelType

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # API Keys
    huggingface_api_key: Optional[str] = os.getenv("HUGGINGFACE_API_KEY")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # Model Defaults
    default_llm: str = os.getenv("DEFAULT_LLM", "gemma-2b-it")
    default_embedding: str = os.getenv("DEFAULT_EMBEDDING", "all-mpnet-base-v2")
    
    # Paths
    data_dir: str = os.getenv("DATA_DIR", "data")
    embeddings_dir: str = os.path.join(os.getenv("DATA_DIR", "data"), "embeddings")
    case_data_path: str = os.path.join(os.getenv("DATA_DIR", "data"), "cases")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # CORS
    cors_origins: list = ["*"]
    
    def get_llm_config(self, model_name: str) -> Dict[str, Any]:
        """Get LLM configuration by model name."""
        from .models import get_model_config
        return get_model_config(model_name, ModelType.CHAT)
    
    def get_embedding_config(self, model_name: str) -> Dict[str, Any]:
        """Get embedding model configuration by model name."""
        from .models import get_model_config
        return get_model_config(model_name, ModelType.EMBEDDING)
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True

# Create settings instance
settings = Settings()
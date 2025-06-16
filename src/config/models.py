from enum import Enum
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from pathlib import Path
import torch

class ModelProvider(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    LLAMA_CPP = "llama_cpp"
    GPTQ = "gptq"
    CT_TRANSFORMERS = "ctransformers"

class ModelType(str, Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"

class ModelConfig(BaseModel):
    """Base configuration for all models."""
    name: str
    provider: ModelProvider
    model_type: ModelType
    model_id: str  # Model ID or path
    context_length: int = 4096
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    device: str = "auto"  # auto, cuda, cpu, mps
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

class EmbeddingModelConfig(ModelConfig):
    """Configuration for embedding models."""
    model_type: ModelType = ModelType.EMBEDDING
    normalize_embeddings: bool = True
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

class LLMConfig(ModelConfig):
    """Configuration for LLM models."""
    model_type: ModelType = ModelType.CHAT
    stream: bool = False
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

# Default model configurations
DEFAULT_EMBEDDING_MODELS = {
    "openai": EmbeddingModelConfig(
        name="text-embedding-3-small",
        provider=ModelProvider.OPENAI,
        model_id="text-embedding-3-small",
        context_length=8191,
    ),
    "all-mpnet-base-v2": EmbeddingModelConfig(
        name="all-mpnet-base-v2",
        provider=ModelProvider.HUGGINGFACE,
        model_id="sentence-transformers/all-mpnet-base-v2",
        context_length=384,
    )
}

DEFAULT_LLM_MODELS = {
    "llama3-8b": LLMConfig(
        name="llama3-8b",
        provider=ModelProvider.HUGGINGFACE,
        model_id="meta-llama/Llama-3.1-8B",
        context_length=8192,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        model_kwargs={
            "device_map": "auto",
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
    ),
    "tinyllama": LLMConfig(
        name="tinyllama",
        provider=ModelProvider.HUGGINGFACE,
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        context_length=2048,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        model_kwargs={
            "device_map": "auto",
            "torch_dtype": torch.float32,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
    ),
    # Keep distilgpt2 as fallback for compatibility
    "distilgpt2": LLMConfig(
        name="distilgpt2",
        provider=ModelProvider.HUGGINGFACE,
        model_id="distilgpt2",
        context_length=1024,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.1,
        model_kwargs={
            "device_map": "auto",
            "torch_dtype": torch.float32,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }
    )
}

# Set default models
DEFAULT_LLM = "llama3-8b"
DEFAULT_EMBEDDING = "all-mpnet-base-v2"

def get_model_config(model_name: str, model_type: ModelType) -> ModelConfig:
    """Get model configuration by name and type."""
    if model_type == ModelType.EMBEDDING:
        return DEFAULT_EMBEDDING_MODELS.get(model_name, DEFAULT_EMBEDDING_MODELS["all-mpnet-base-v2"])
    return DEFAULT_LLM_MODELS.get(model_name, DEFAULT_LLM_MODELS["llama3-8b"])  # Default to llama3-8b
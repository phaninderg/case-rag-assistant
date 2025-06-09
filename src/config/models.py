from enum import Enum
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from pathlib import Path

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
    "tinyllama": LLMConfig(
        name="tinyllama",
        provider=ModelProvider.HUGGINGFACE,
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        context_length=2048,
        model_kwargs={
            "device_map": "auto",
            "load_in_4bit": True,  # Enable 4-bit quantization
            "torch_dtype": "auto",
            "trust_remote_code": True
        }
    ),
    "mistral-7b": LLMConfig(
        name="mistral-7b",
        provider=ModelProvider.HUGGINGFACE,
        model_id="mistralai/Mistral-7B-v0.1",
        context_length=8192,
        model_kwargs={
            "device_map": "auto",
            "load_in_4bit": True,  # Enable 4-bit quantization
            "torch_dtype": "auto",
            "trust_remote_code": True,
            "low_cpu_mem_usage": True
        }
    )
}

def get_model_config(model_name: str, model_type: ModelType) -> ModelConfig:
    """Get model configuration by name and type."""
    if model_type == ModelType.EMBEDDING:
        return DEFAULT_EMBEDDING_MODELS.get(model_name, DEFAULT_EMBEDDING_MODELS["all-mpnet-base-v2"])
    return DEFAULT_LLM_MODELS.get(model_name, DEFAULT_LLM_MODELS["tinyllama"])  # Default to tinyllama
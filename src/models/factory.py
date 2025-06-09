# In src/models/factory.py
from typing import Dict, Any, Optional, List, Union
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms.base import BaseLLM
from langchain.embeddings.base import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings

from src.config.models import ModelConfig, ModelProvider, LLMConfig, EmbeddingModelConfig
from src.config import settings

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory class for creating language and embedding models optimized for Apple Silicon."""
    
    @classmethod
    def create_llm(cls, config: LLMConfig, **kwargs) -> BaseLLM:
        """Create a language model based on the configuration."""
        try:
            if config.provider == ModelProvider.HUGGINGFACE:
                return cls._create_hf_llm(config, **kwargs)
            elif config.provider == ModelProvider.OPENAI:
                return cls._create_openai_llm(config, **kwargs)
            else:
                raise ValueError(f"Unsupported provider: {config.provider}")
        except Exception as e:
            logger.error(f"Error creating LLM: {str(e)}")
            raise

    @classmethod
    def create_embeddings(cls, config: EmbeddingModelConfig, **kwargs) -> Embeddings:
        """Create an embedding model based on the configuration."""
        try:
            if config.provider == ModelProvider.HUGGINGFACE:
                return cls._create_hf_embeddings(config, **kwargs)
            else:
                raise ValueError(f"Unsupported provider: {config.provider}")
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise

    @staticmethod
    def _create_hf_llm(config: LLMConfig, **kwargs) -> BaseLLM:
        """Create a HuggingFace LLM optimized for Apple Silicon."""
        from huggingface_hub import login
        from huggingface_hub.utils import HfHubHTTPError
        
        try:
            # Configure tokenizer with authentication
            tokenizer_kwargs = {
                "use_fast": True,
                "padding_side": "left",
                "truncation_side": "left",
                "trust_remote_code": True
            }
            
            # Add token if available
            if settings.huggingface_api_key:
                tokenizer_kwargs["use_auth_token"] = settings.huggingface_api_key
            
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_id,
                **tokenizer_kwargs
            )
            
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Configure model with 4-bit quantization
            model_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.float32,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                **config.model_kwargs
            }
            
            # Add authentication token if available
            if settings.huggingface_api_key:
                model_kwargs["use_auth_token"] = settings.huggingface_api_key
            
            # Configure 4-bit quantization if specified
            if model_kwargs.get("load_in_4bit", False):
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs.pop("load_in_4bit", None)
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                **model_kwargs
            )
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                device_map="auto",
            )
            
            from langchain.llms import HuggingFacePipeline
            return HuggingFacePipeline(pipeline=pipe)
            
        except HfHubHTTPError as e:
            if "gated" in str(e).lower() or "access" in str(e).lower():
                logger.error(f"Model access error. Please accept the terms at: https://huggingface.co/{config.model_id}")
            raise RuntimeError(f"Failed to load model {config.model_id}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading model {config.model_id}: {str(e)}")
            raise

    @staticmethod
    def _create_openai_llm(config: LLMConfig, **kwargs) -> BaseLLM:
        """Create an OpenAI LLM."""
        from langchain.chat_models import ChatOpenAI
        return ChatOpenAI(
            model_name=config.model_id,
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
            model_kwargs={
                "top_p": config.top_p,
                **config.model_kwargs
            },
            **kwargs
        )

    @staticmethod
    def _create_hf_embeddings(config: EmbeddingModelConfig, **kwargs) -> Embeddings:
        """Create HuggingFace embeddings."""
        try:
            model_kwargs = {
                'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
                **config.model_kwargs
            }
            
            return HuggingFaceEmbeddings(
                model_name=config.model_id,
                model_kwargs=model_kwargs,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error creating HuggingFace embeddings: {str(e)}")
            raise
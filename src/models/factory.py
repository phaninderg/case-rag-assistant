from typing import Dict, Any, Optional, List, Union
import logging
import torch
import sys
import platform
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain.llms.base import BaseLLM
from langchain.embeddings.base import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings

from src.config.models import ModelConfig, ModelProvider, LLMConfig, EmbeddingModelConfig
from src.config import settings

logger = logging.getLogger(__name__)

# Check if we're on macOS
IS_MACOS = sys.platform == 'darwin' or platform.system() == 'Darwin'

# Check if auto-gptq is available
AUTO_GPTQ_AVAILABLE = False
if not IS_MACOS:
    try:
        import auto_gptq
        AUTO_GPTQ_AVAILABLE = True
    except ImportError:
        logger.warning("auto-gptq is not available. Some quantization features will be disabled.")
        AUTO_GPTQ_AVAILABLE = False

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
            elif config.provider == ModelProvider.OPENAI:
                return cls._create_openai_embeddings(config, **kwargs)
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
        from langchain.llms import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
        
        try:
            # Configure tokenizer
            tokenizer_kwargs = {
                "use_fast": True,
                "padding_side": "left",
                "truncation_side": "left",
                "trust_remote_code": True
            }
            
            # Add token if available
            if settings.huggingface_api_key:
                login(token=settings.huggingface_api_key)
            
            # Get model kwargs and handle quantization
            model_kwargs = config.model_kwargs.copy()
            
            # Convert quantization config dict to BitsAndBytesConfig if needed
            if 'quantization_config' in model_kwargs and isinstance(model_kwargs['quantization_config'], dict):
                q_config = model_kwargs['quantization_config']
                model_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=q_config.get('load_in_4bit', False),
                    load_in_8bit=q_config.get('load_in_8bit', False),
                    bnb_4bit_quant_type=q_config.get('bnb_4bit_quant_type', 'nf4'),
                    bnb_4bit_compute_dtype={
                        'float16': torch.float16,
                        'bfloat16': torch.bfloat16,
                        'float32': torch.float32
                    }.get(q_config.get('bnb_4bit_compute_dtype', 'float16'), torch.float16),
                    bnb_4bit_use_double_quant=q_config.get('bnb_4bit_use_double_quant', True)
                )
            
            # Ensure device_map is set
            if 'device_map' not in model_kwargs:
                model_kwargs['device_map'] = 'auto'
            
            # Load model and tokenizer
            logger.info(f"Loading model: {config.model_id}")
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_id,
                **tokenizer_kwargs
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                **model_kwargs
            )
            
            # Create text generation pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                device_map=model_kwargs.get('device_map', 'auto'),
            )
            
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
        
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key is required but not set")
            
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
            
    @staticmethod
    def _create_openai_embeddings(config: EmbeddingModelConfig, **kwargs) -> Embeddings:
        """Create OpenAI embeddings."""
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key is required but not set")
            
        return OpenAIEmbeddings(
            model=config.model_id,
            **kwargs
        )
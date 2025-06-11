import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
import torch
from langchain.schema import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import login, HfApi
import os

from src.config.settings import settings
from src.models.factory import ModelFactory

logger = logging.getLogger(__name__)

class LLMService:
    """Service for handling LLM interactions with support for multiple providers."""
    
    def __init__(self, model_name: Optional[str] = None, streaming: bool = False):
        """
        Initialize the LLM service with the specified model and settings.
        
        Args:
            model_name: Name of the LLM to use. If None, uses the default from settings.
            streaming: Whether to enable streaming responses
        """
        self.model_name = model_name or settings.default_llm
        self.streaming = streaming
        self.llm = None
        
        # Setup Hugging Face authentication
        self._setup_huggingface_auth()
        
        # Initialize the model
        self.llm = self._initialize_llm()
    
    def _setup_huggingface_auth(self):
        """Set up Hugging Face authentication if API key is provided."""
        if settings.huggingface_api_key:
            try:
                login(token=settings.huggingface_api_key)
                logger.info("Successfully logged in to Hugging Face Hub")
            except Exception as e:
                logger.warning(f"Failed to login to Hugging Face Hub: {str(e)}")
    
    def _get_model_kwargs(self, llm_config):
        """Get model initialization kwargs with appropriate defaults."""
        model_kwargs = {}
        
        # Start with any model-specific kwargs from config
        if hasattr(llm_config, 'model_kwargs'):
            model_kwargs.update(llm_config.model_kwargs)
        
        # Default to CPU for stability
        device = "cpu"
        torch_dtype = torch.float32
        
        # Only use CUDA if explicitly available and requested
        if torch.cuda.is_available() and model_kwargs.get('device_map') != 'cpu':
            device = "cuda"
            torch_dtype = torch.float16
        
        # Clean up model kwargs
        model_kwargs = {
            'device_map': device,
            'torch_dtype': torch_dtype,
            'low_cpu_mem_usage': True,
            'trust_remote_code': model_kwargs.get('trust_remote_code', True),
        }
        
        return model_kwargs
    
    def _initialize_llm(self):
        """Initialize the language model with proper error handling."""
        llm_config = settings.get_llm_config(self.model_name)
        
        try:
            logger.info(f"Initializing model: {self.model_name}")
            
            # Get model kwargs with appropriate defaults
            model_kwargs = self._get_model_kwargs(llm_config)
            
            # Special handling for smaller models
            if 'tinyllama' in self.model_name.lower() or 'distilgpt2' in self.model_name.lower():
                model_kwargs['max_memory'] = {0: '2GB'}  # Limit memory usage
                model_kwargs['device_map'] = 'auto'  # Let accelerate handle device placement
            
            # Initialize the model
            return ModelFactory.create_llm(
                llm_config,
                model_kwargs=model_kwargs,
                streaming=self.streaming
            )
            
        except Exception as e:
            error_msg = f"Error initializing {self.model_name}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(f"Failed to initialize model. {error_msg}")
    
    def _truncate_text(self, text: str, max_tokens: int = 1000) -> str:
        """Truncate text to a maximum number of tokens."""
        if not text:
            return text
            
        # Simple word-based truncation (can be replaced with tokenizer-based if needed)
        words = text.split()
        if len(words) > max_tokens:
            return ' '.join(words[:max_tokens]) + '... [truncated]'
        return text
    
    def _format_prompt(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_length: int = 1024  
    ) -> str:
        """
        Format the prompt with system message and chat history.
        
        Args:
            prompt: The user's input prompt
            system_message: Optional system message
            chat_history: List of previous messages
            max_length: Maximum length in characters (not tokens)
            
        Returns:
            Formatted prompt string
        """
        def truncate(text: str, max_chars: int) -> str:
            """Truncate text to a maximum number of characters."""
            if not text or len(text) <= max_chars:
                return text
            return text[:max_chars-3] + '...'
        
        parts = []
        current_length = 0
        
        # Add system message if provided (limited to 20% of max length)
        if system_message:
            sys_max = max_length // 5  # 20% of max length for system message
            sys_msg = f"System: {system_message.strip()}"
            if len(sys_msg) > sys_max:
                sys_msg = f"System: {truncate(system_message.strip(), sys_max - 9)} [truncated]"
            parts.append(sys_msg)
            current_length += len(sys_msg)
        
        # Add chat history if provided (limited to 50% of remaining length)
        remaining_length = max_length - current_length
        history_max = remaining_length // 2
        
        if chat_history and history_max > 100:  # Only include history if we have enough space
            history_parts = []
            history_length = 0
            
            # Process most recent messages first
            for msg in reversed(chat_history):
                role = str(msg.get("role", "user")).capitalize()
                content = str(msg.get("content", "")).strip()
                if not content:
                    continue
                    
                # Truncate content to fit remaining space
                max_content_length = history_max - history_length - len(role) - 5  # 5 for ": " and some padding
                if max_content_length < 20:  # Minimum length for any message
                    break
                    
                truncated_content = truncate(content, max_content_length)
                msg_str = f"{role}: {truncated_content}"
                
                history_parts.append(msg_str)
                history_length += len(msg_str)
                
                if history_length >= history_max:
                    break
            
            # Add history in chronological order
            if history_parts:
                parts.extend(reversed(history_parts))
                current_length += history_length
        
        # Add the current prompt (use remaining space)
        remaining_for_prompt = max(10, max_length - current_length - 50)  # Leave space for "User: " and "Assistant:"
        prompt_str = f"User: {truncate(prompt.strip(), remaining_for_prompt)}"
        parts.append(prompt_str)
        parts.append("Assistant:")
        
        # Join all parts with newlines and ensure we don't exceed max_length
        formatted_prompt = "\n".join(parts)
        if len(formatted_prompt) > max_length:
            formatted_prompt = formatted_prompt[:max_length-3] + '...'
        
        return formatted_prompt
    
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from the LLM with robust error handling for tensor shapes.
        
        Args:
            prompt: The user's input prompt
            system_message: Optional system message
            chat_history: List of previous messages
            **kwargs: Additional parameters for generation
            
        Returns:
            The generated response as a string
        """
        if not self.llm:
            raise RuntimeError("LLM not initialized. Please check the logs for initialization errors.")
        
        try:
            # Format the prompt with strict truncation
            formatted_prompt = self._format_prompt(prompt, system_message, chat_history)
            
            # Log the prompt length for debugging
            prompt_length = len(formatted_prompt.split())
            logger.debug(f"Generating response for prompt (tokens: ~{prompt_length})")
            
            # Set conservative generation parameters
            generation_kwargs = {
                'max_new_tokens': 512,  # Limit response length
                'temperature': 0.7,    # Balanced creativity
                'top_p': 0.9,          # Nucleus sampling
                'do_sample': True,     # Enable sampling
                'num_return_sequences': 1,
                'eos_token_id': self.llm.tokenizer.eos_token_id if hasattr(self.llm, 'tokenizer') else None,
                'pad_token_id': self.llm.tokenizer.pad_token_id if hasattr(self.llm, 'tokenizer') else None,
                **kwargs
            }
            
            # Try with a smaller batch size if needed
            if 'batch_size' not in generation_kwargs:
                generation_kwargs['batch_size'] = 1
            
            # Generate the response with error handling
            try:
                response = await self.llm.agenerate(
                    [formatted_prompt],
                    **generation_kwargs
                )
                
                # Extract and clean the generated text
                if hasattr(response, 'generations') and response.generations:
                    if response.generations and len(response.generations) > 0:
                        generation = response.generations[0][0]
                        if hasattr(generation, 'text'):
                            return str(generation.text).strip()
                
                return "No response generated"
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
                    logger.warning("CUDA out of memory, trying with reduced batch size")
                    generation_kwargs['batch_size'] = 1
                    response = await self.llm.agenerate(
                        [formatted_prompt],
                        **generation_kwargs
                    )
                    if hasattr(response, 'generations') and response.generations:
                        return str(response.generations[0][0].text).strip()
                raise
                
        except Exception as e:
            logger.error(f"Error in generate: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate response: {str(e)}")
    
    async def stream(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream responses from the LLM.
        
        Args:
            prompt: The user's input prompt
            system_message: Optional system message
            chat_history: List of previous messages
            **kwargs: Additional parameters for generation
            
        Yields:
            Chunks of the generated response
        """
        if not self.llm:
            yield "Error: LLM not initialized. Please check the logs for initialization errors."
            return
            
        if not self.streaming:
            self.llm.streaming = True
            
        try:
            # Format the prompt with truncation
            formatted_prompt = self._format_prompt(prompt, system_message, chat_history)
            
            # Set generation parameters
            generation_kwargs = {
                "max_length": 512,  # Limit response length
                "temperature": 0.7,
                "do_sample": True,
                **kwargs
            }
            
            # Log the prompt for debugging
            logger.debug(f"Streaming response for prompt (length: {len(formatted_prompt)}): {formatted_prompt[:200]}...")
            
            # Stream the response
            full_response = ""
            try:
                async for chunk in self.llm.astream(formatted_prompt, **generation_kwargs):
                    if hasattr(chunk, 'content'):
                        content = str(chunk.content)
                    elif hasattr(chunk, 'text'):
                        content = str(chunk.text)
                    elif hasattr(chunk, 'generated_text'):
                        content = str(chunk.generated_text)
                    else:
                        content = str(chunk)
                    
                    # Only yield new content
                    if content and len(content) > len(full_response):
                        new_content = content[len(full_response):]
                        full_response = content
                        if new_content:
                            yield new_content
            except Exception as e:
                logger.error(f"Error during streaming: {str(e)}", exc_info=True)
                yield f"\n\nError during streaming: {str(e)}"
            
            logger.debug(f"Full response: {full_response}")
            
        except Exception as e:
            error_msg = f"Error in streaming response: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield f"Error: {error_msg}"
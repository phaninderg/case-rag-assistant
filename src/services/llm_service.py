import logging
from typing import Dict, List, Optional, Any, AsyncGenerator

from langchain.schema import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from huggingface_hub import login, HfApi

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
        
        # Try to initialize the primary model
        self.llm = self._initialize_llm()
    
    def _setup_huggingface_auth(self):
        """Set up Hugging Face authentication if API key is provided."""
        if settings.huggingface_api_key:
            try:
                login(token=settings.huggingface_api_key)
                logger.info("Successfully logged in to Hugging Face Hub")
            except Exception as e:
                logger.warning(f"Failed to login to Hugging Face Hub: {str(e)}")
    
    def _initialize_llm(self):
        """Initialize the language model with proper error handling."""
        llm_config = settings.get_llm_config(self.model_name)
        
        try:
            logger.info(f"Initializing model: {self.model_name}")
            return ModelFactory.create_llm(
                llm_config,
                streaming=self.streaming,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]) if self.streaming else None,
            )
        except Exception as e:
            error_msg = f"Error initializing {self.model_name}: {str(e)}"
            logger.error(error_msg)
            
            # If the error is about model access, suggest visiting the model card
            if "gated" in str(e).lower() or "access" in str(e).lower():
                model_id = llm_config.model_id if hasattr(llm_config, 'model_id') else 'unknown'
                logger.error(f"You may need to accept the model's terms at: https://huggingface.co/{model_id}")
            
            raise RuntimeError(f"Failed to initialize model. {error_msg}")
    
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The user's input prompt
            system_message: Optional system message to set the behavior
            chat_history: List of previous messages in the conversation
            **kwargs: Additional parameters for generation
            
        Returns:
            The generated response as a string
        """
        if not self.llm:
            raise RuntimeError("LLM not initialized. Please check the logs for initialization errors.")
            
        messages = self._prepare_messages(prompt, system_message, chat_history)
        
        try:
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
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
            system_message: Optional system message to set the behavior
            chat_history: List of previous messages in the conversation
            **kwargs: Additional parameters for generation
            
        Yields:
            Chunks of the generated response
        """
        if not self.llm:
            yield "Error: LLM not initialized. Please check the logs for initialization errors."
            return
            
        if not self.streaming:
            self.llm.streaming = True
            
        messages = self._prepare_messages(prompt, system_message, chat_history)
        
        try:
            full_response = ""
            async for chunk in self.llm.astream(messages, **kwargs):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                full_response += content
                yield content
                
            logger.debug(f"Full response: {full_response}")
        except Exception as e:
            error_msg = f"Error in streaming response: {str(e)}"
            logger.error(error_msg)
            yield f"Error: {error_msg}"
    
    def _prepare_messages(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> List[BaseMessage]:
        """Prepare messages for the LLM with proper formatting."""
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append(SystemMessage(content=system_message))
        
        # Add chat history
        if chat_history:
            for msg in chat_history:
                role = msg.get("role", "").lower()
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
        
        # Add current prompt
        messages.append(HumanMessage(content=prompt))
        
        return messages
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
import torch
from langchain.schema import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login, HfApi
import os

from src.config.settings import settings

logger = logging.getLogger(__name__)

class LLMService:
    """Service for handling LLM interactions with support for multiple providers."""
    
    def __init__(self, model_path: str = None, model_name: str = "distilgpt2", case_service: 'CaseService' = None):
        """
        Initialize the LLM service.
        
        Args:
            model_path: Path to a fine-tuned model
            model_name: Name of the base model to use if no path is provided
            case_service: Instance of CaseService for vector search
        """
        self.model_path = model_path
        self.model_name = model_name
        self.case_service = case_service
        self.device = self._get_device()
        logger.info(f"Using device: {self.device}")
        self.model = None
        self.tokenizer = None
        self.load_model(model_path or model_name)
        
        # Setup Hugging Face authentication
        self._setup_huggingface_auth()
        
    def _setup_huggingface_auth(self):
        """Set up Hugging Face authentication if API key is provided."""
        if settings.huggingface_api_key:
            try:
                login(token=settings.huggingface_api_key)
                logger.info("Successfully logged in to Hugging Face Hub")
            except Exception as e:
                logger.warning(f"Failed to login to Hugging Face Hub: {str(e)}")
    
    def _get_device(self):
        """Force CPU usage for all operations."""
        logger.info("Using CPU for all operations")
        return "cpu"

    def load_model(self, model_path: str = None):
        """
        Load the model and tokenizer.
        
        Args:
            model_path: Path to the model to load
        """
        try:
            if model_path:
                self.model_path = model_path
                logger.info(f"Loading model from {model_path}")
                
                # Load tokenizer first
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                
                # Configure padding token if needed
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load model with appropriate settings
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32  # Use float32 for better compatibility
                ).to(self.device)
                
                logger.info(f"Successfully loaded model on {self.device}")
            else:
                logger.info(f"Using default model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32
                ).to(self.device)
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise

    async def generate_response(self, prompt: str, max_length: int = 200, **kwargs) -> str:
        """
        Generate a response using the loaded model.
        
        Args:
            prompt: Input prompt for generation
            max_length: Maximum length of the generated response
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated text response
        """
        try:
            if not self.model or not self.tokenizer:
                raise ValueError("Model or tokenizer not loaded. Call load_model() first.")
                
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    **kwargs
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the response part if it follows the instruction format
            if "### Response:" in response:
                response = response.split("### Response:")[1].strip()
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            raise

    async def summarize_case(self, subject: str, description: str) -> str:
        """
        Generate a summary for a support case.
        
        Args:
            subject: Case subject
            description: Case description
            
        Returns:
            str: Generated summary
        """
        prompt = (
            "### Instruction: Summarize this support case in 3-5 sentences "
            "focusing on the core issue.\n\n"
            f"### Input:\nSubject: {subject}\nDescription: {description}\n\n"
            "### Response:"
        )
        return await self.generate_response(prompt, max_length=300)

    async def search_similar_cases(
        self, 
        query: str, 
        k: int = 5, 
        min_score: float = 0.6,
        include_solutions: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Find similar cases based on the query using ChromaDB vector search.
        
        Args:
            query: Search query string
            k: Number of results to return
            min_score: Minimum similarity score (0-1)
            include_solutions: Whether to include AI-generated solutions
            **kwargs: Additional search parameters
            
        Returns:
            List of solutions with scores
        """
        try:
            if not self.model or not self.tokenizer:
                raise ValueError("Model or tokenizer not loaded. Call load_model() first.")
            
            if not self.case_service:
                raise ValueError("CaseService not provided. Cannot perform vector search.")
            
            # Get similar cases using case service's vector search
            similar_docs = await self.case_service.find_similar_cases(
                query=query,
                k=k,
                min_score=min_score,
                **kwargs
            )
            
            if not similar_docs:
                logger.info("No similar cases found in the vector database.")
                return []
                
            solutions = []
            for doc in similar_docs:
                try:
                    # Handle both document objects and dictionaries
                    if hasattr(doc, 'metadata'):
                        metadata = doc.metadata
                        content = getattr(doc, 'page_content', '')
                    elif isinstance(doc, dict):
                        metadata = doc.get('metadata', {})
                        content = doc.get('content', '')
                        if not content and 'page_content' in doc:
                            content = doc['page_content']
                    else:
                        logger.warning(f"Unexpected document type: {type(doc)}")
                        continue
                    
                    # Get the similarity score
                    score = doc.get('score', metadata.get('score', 0.0))
                    
                    # If no solution needed or possible, return basic info
                    if not include_solutions:
                        solution_text = content[:500]  # Return first 500 chars if no solution needed
                    else:
                        try:
                            # Extract key information from the case
                            subject = metadata.get('subject', 'No subject')
                            close_notes = metadata.get('close_notes', '').strip()
                            
                            # Check if close_notes exist
                            if not close_notes:
                                solution_text = "No close_notes found for this case, cannot propose a solution."
                            else:
                                # Use close_notes as the basis for the solution
                                prompt = (
                                    "### Support Case Summary ###\n"
                                    f"Subject: {subject}\n\n"
                                    "### Close Notes ###\n"
                                    f"{close_notes}\n\n"
                                    "### Your Task ###\n"
                                    "Based on the above case close notes, summarize the solution that was implemented.\n"
                                    "Focus on the key resolution steps and final outcome.\n\n"
                                    "### Solution Summary ###\n"
                                )
                                
                                solution_text = await self.generate_response(
                                    prompt=prompt,
                                    max_length=500,
                                    temperature=0.3,  # Lower temperature for more factual responses
                                    top_p=0.9,
                                    do_sample=True,
                                    max_new_tokens=300
                                )
                                
                                # Clean up the response
                                solution_text = solution_text.strip()
                                
                                # Remove any remaining prompt artifacts
                                for marker in ["### Solution Summary ###", "Solution:", "SOLUTION:"]:
                                    if marker in solution_text:
                                        solution_text = solution_text.split(marker, 1)[-1].strip()
                                
                                # Ensure we have a valid solution
                                if not solution_text or solution_text.isspace() or len(solution_text) < 10:
                                    solution_text = "No detailed solution could be generated from the close notes."
                        
                        except Exception as e:
                            logger.error(f"Error generating solution: {str(e)}", exc_info=True)
                            solution_text = "Error generating solution from close notes."
                    
                    # Add to results
                    solutions.append({
                        'solution': solution_text,
                        'similarity_score': score,
                        'case_number': metadata.get('case_number', 'N/A')
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing document: {str(e)}", exc_info=True)
                    continue
            
            # Sort by similarity score (highest first)
            solutions.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            
            return solutions
            
        except Exception as e:
            logger.error(f"Error in search_similar_cases: {str(e)}", exc_info=True)
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding for the given text.
        
        Args:
            text: Input text to generate embedding for
            
        Returns:
            List[float]: The generated embedding
        """
        try:
            if not text or not isinstance(text, str) or not text.strip():
                logger.warning("Empty or invalid text provided for embedding")
                return [0.0] * 768  # Default dimension
                
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use mean of last hidden state as the embedding
                last_hidden = outputs.hidden_states[-1]
                # Mean pooling
                input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = sum_embeddings / sum_mask
                
                return embedding[0].cpu().numpy().tolist()
                
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return a zero vector of appropriate dimension
            hidden_size = getattr(self.model.config, 'hidden_size', 768) if hasattr(self.model, 'config') else 768
            return [0.0] * hidden_size
    
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
        """Initialize the language model with CPU-only configuration."""
        try:
            # Force CPU usage
            os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA
            logger.info("Initializing model with CPU-only configuration")
            
            # Configure tokenizer
            tokenizer_kwargs = {
                'use_fast': True,
                'padding_side': 'left',
                'truncation_side': 'left',
                'trust_remote_code': True
            }
            
            logger.info(f"Loading tokenizer for model: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **tokenizer_kwargs
            )
            
            logger.info(f"Loading model: {self.model_name}")
            
            # Create text generation pipeline with CPU device
            pipe = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=tokenizer,
                device=-1,  # Force CPU
                framework="pt",
                model_kwargs={
                    'device_map': None,  # Disable device_map when using device
                    'low_cpu_mem_usage': True
                }
            )
            
            logger.info(f"Successfully loaded model: {self.model_name}")
            return pipe
            
        except Exception as e:
            error_msg = f"Error initializing {self.model_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)
    
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
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized. Please check the logs for initialization errors.")
        
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
                'eos_token_id': self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None,
                'pad_token_id': self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else None,
                **kwargs
            }
            
            # Try with a smaller batch size if needed
            if 'batch_size' not in generation_kwargs:
                generation_kwargs['batch_size'] = 1
            
            # Generate the response with error handling
            try:
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **generation_kwargs
                    )
                
                # Extract and clean the generated text
                if hasattr(outputs, 'shape'):
                    if outputs.shape and len(outputs.shape) > 0:
                        generation = outputs[0]
                        if hasattr(generation, 'cpu'):
                            generation = generation.cpu()
                        if hasattr(generation, 'numpy'):
                            generation = generation.numpy()
                        return str(generation).strip()
                
                return "No response generated"
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda out of memory" in str(e).lower():
                    logger.warning("CUDA out of memory, trying with reduced batch size")
                    generation_kwargs['batch_size'] = 1
                    inputs = self.tokenizer(
                        formatted_prompt,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding=True
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            **generation_kwargs
                        )
                    
                    if hasattr(outputs, 'shape'):
                        if outputs.shape and len(outputs.shape) > 0:
                            generation = outputs[0]
                            if hasattr(generation, 'cpu'):
                                generation = generation.cpu()
                            if hasattr(generation, 'numpy'):
                                generation = generation.numpy()
                            return str(generation).strip()
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
            str: The next chunk of the generated response
        """
        if not self.model or not self.tokenizer:
            yield "Error: Model not initialized. Please check the logs for initialization errors."
            return
            
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
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **generation_kwargs
                    )
                
                # Extract and clean the generated text
                if hasattr(outputs, 'shape'):
                    if outputs.shape and len(outputs.shape) > 0:
                        generation = outputs[0]
                        if hasattr(generation, 'cpu'):
                            generation = generation.cpu()
                        if hasattr(generation, 'numpy'):
                            generation = generation.numpy()
                        full_response = str(generation).strip()
                
                yield full_response
                
            except Exception as e:
                logger.error(f"Error during streaming: {str(e)}", exc_info=True)
                yield f"\n\nError during streaming: {str(e)}"
            
            logger.debug(f"Full response: {full_response}")
            
        except Exception as e:
            error_msg = f"Error in streaming response: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield f"Error: {error_msg}"
    
    async def generate_response(self, prompt: str, max_length: int = 256, **kwargs) -> str:
        """
        Generate a response for the given prompt using the current model.
        
        Args:
            prompt: The input prompt
            max_length: Maximum length of the generated response in tokens
            **kwargs: Additional generation parameters (will override defaults)
            
        Returns:
            Generated text response
        """
        try:
            if not self.model or not self.tokenizer:
                raise ValueError("Model or tokenizer not loaded. Call load_model() first.")
            
            # Ensure model is on the correct device
            self.model = self.model.to(self.device)
                
            # Set default generation parameters with more stable values
            default_params = {
                'max_new_tokens': min(max_length, 300),
                'temperature': 0.7,  # Default temperature
                'top_p': 0.9,
                'top_k': 50,
                'repetition_penalty': 1.1,  # Slightly lower to avoid extreme values
                'no_repeat_ngram_size': 3,
                'do_sample': True,
                'num_beams': 1,  # Start with greedy decoding
                'early_stopping': True,
                'length_penalty': 1.0,
                'pad_token_id': self.tokenizer.eos_token_id,
                'eos_token_id': self.tokenizer.eos_token_id
            }
            
            # Update defaults with any provided kwargs
            generation_params = {**default_params, **kwargs}
            
            # Encode the input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # First try with standard parameters
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **{k: v for k, v in generation_params.items() 
                           if k not in inputs and k != 'prompt'}
                    )
            except RuntimeError as e:
                if 'probability tensor' in str(e).lower():
                    # If we get probability errors, try with more stable settings
                    logger.warning("Probability error detected, retrying with more stable parameters")
                    stable_params = {
                        'temperature': 0.7,
                        'top_p': 0.9,
                        'top_k': 0,  # Disable top_k sampling for more stability
                        'do_sample': True,
                        'num_beams': 1,  # Use greedy decoding
                        'repetition_penalty': 1.0  # No repetition penalty
                    }
                    generation_params.update(stable_params)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            **generation_params
                        )
                else:
                    raise
            
            # Handle different output formats
            if hasattr(outputs, 'sequences'):
                output_sequences = outputs.sequences
            elif isinstance(outputs, torch.Tensor):
                output_sequences = outputs
            else:
                output_sequences = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            
            # Move to CPU for decoding if needed
            if output_sequences.device != torch.device('cpu'):
                output_sequences = output_sequences.cpu()
            
            # Decode the response
            response = self.tokenizer.decode(
                output_sequences[0], 
                skip_special_tokens=True
            )
            
            # Remove the input prompt from the response if it's included
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
                
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            # Return a simple error message that can be handled upstream
            return f"Error generating response: {str(e)}"
    
    async def generate_summary(self, text: str, max_length: int = 200, **kwargs) -> str:
        """
        Generate a summary of the input text using the instruction-finetuned model.
        
        Args:
            text: The text to summarize
            max_length: Maximum length of the summary in tokens
            **kwargs: Additional generation parameters
            
        Returns:
            Generated summary (3-5 sentences)
        """
        try:
            if not self.model or not self.tokenizer:
                raise ValueError("Model or tokenizer not loaded. Call load_model() first.")
            
            # Format the prompt for the instruction-finetuned model
            prompt = (
                f"### Instruction:\nSummarize this support case in 3-5 sentences, "
                f"focusing on the core issue and resolution.\n\n"
                f"### Input:\n{text}\n\n"
                f"### Response:\n"
            )
            
            # Set generation parameters
            generation_params = {
                'max_new_tokens': min(max_length, 300),
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 50,
                'repetition_penalty': 1.1,
                'do_sample': True,
                'num_return_sequences': 1,
                'pad_token_id': self.tokenizer.eos_token_id,
                'eos_token_id': self.tokenizer.eos_token_id,
                **kwargs
            }
            
            # Generate response
            response = await self.generate_response(
                prompt=prompt,
                **{k: v for k, v in generation_params.items() 
                   if k not in ['prompt']}
            )
            
            # Clean up the response
            response = response.strip()
            
            # Remove any remaining prompt parts
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()
                
            # Ensure proper sentence structure
            sentences = [s.strip() for s in response.split('.') if s.strip()]
            
            # Take first 3-5 sentences
            if len(sentences) > 5:
                response = '. '.join(sentences[:5]).strip() + '.'
            elif sentences:
                response = '. '.join(sentences).strip()
                if not response.endswith('.'):
                    response += '.'
            
            # Fallback if response is too short
            if not response or len(response.split()) < 3:
                return self._extractive_summarize(text)
                
            return response
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            # Fallback to extractive summarization
            return self._extractive_summarize(text)
    
    def _extractive_summarize(self, text: str, max_sentences: int = 3) -> str:
        """
        Fallback extractive summarization method.
        Returns the first few sentences of the text.
        """
        import re
        # Split into sentences (naive approach)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Take first few sentences
        summary = ' '.join(sentences[:max_sentences]).strip()
        # Ensure it ends with proper punctuation
        if summary and not any(summary.endswith(p) for p in ['.', '!', '?']):
            summary += '.'
        return summary if summary else "Unable to generate summary."
    
    async def stream_response(
        self,
        prompt: str,
        max_length: int = 1000,
        **generation_params
    ) -> AsyncGenerator[str, None]:
        """
        Stream the model's response token by token.
        
        Args:
            prompt: The input prompt
            max_length: Maximum length of the generated response
            **generation_params: Additional generation parameters
            
        Yields:
            str: The next token in the generated response
        """
        try:
            if not self.model or not self.tokenizer:
                raise ValueError("Model or tokenizer not loaded. Call load_model() first.")
                
            # Tokenize the input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Set default generation parameters if not provided
            if 'max_length' not in generation_params:
                generation_params['max_length'] = max_length
            if 'temperature' not in generation_params:
                generation_params['temperature'] = 0.7
            if 'top_p' not in generation_params:
                generation_params['top_p'] = 0.9
                
            # Generate tokens one by one
            with torch.no_grad():
                for output in self.model.generate(
                    **inputs,
                    max_new_tokens=generation_params.get('max_length', max_length),
                    temperature=generation_params.get('temperature'),
                    top_p=generation_params.get('top_p'),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **{k: v for k, v in generation_params.items() 
                       if k not in ['max_length', 'temperature', 'top_p', 'do_sample']}
                ):
                    # Decode the new tokens
                    new_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    # Skip the prompt in the first chunk
                    if len(new_text) <= len(prompt):
                        continue
                    # Yield only the new text
                    yield new_text[len(prompt):]
                    
        except Exception as e:
            logger.error(f"Error in stream_response: {str(e)}")
            yield f"Error generating response: {str(e)}"
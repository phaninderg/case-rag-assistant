import logging
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
import torch
from langchain.schema import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login, HfApi
import os
import psutil
import re

from src.config.settings import settings

logger = logging.getLogger(__name__)

class LLMService:
    """Service for handling LLM interactions with support for multiple providers."""
    
    def __init__(self, model_path: str = None, model_name: str = None, case_service: 'CaseService' = None):
        """
        Initialize the LLM service.
        
        Args:
            model_path: Path to a fine-tuned model
            model_name: Name of the base model to use if no path is provided (defaults to settings.default_llm)
            case_service: Instance of CaseService for vector search
        """
        self.model_path = model_path
        self.model_name = model_name or settings.default_llm
        self.case_service = case_service
        self.device = self._get_device()
        logger.info(f"Using device: {self.device}")
        self.model = None
        self.tokenizer = None
        
        # Configure environment for Apple Silicon
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        
        # Log the model being loaded for debugging
        model_to_load = self.model_path or self.model_name
        
        # Load model using the instance variable which has the correct default from settings
        self.load_model(model_to_load)
        
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
        """Get the appropriate device for model inference."""
        if torch.cuda.is_available():
            logger.info("Using CUDA device")
            return "cuda"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            logger.info("Using MPS device")
            return "mps"
        else:
            logger.info("Using CPU")
            return "cpu"

    def _log_memory_usage(self, prefix: str = ""):
        """Log current memory usage."""
        process = psutil.Process()
        mem_info = process.memory_info()
        logger.info(f"{prefix}Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
    
    def load_model(self, model_identifier: str = None):
        """
        Load the model with 4-bit quantization and memory optimizations.
        """
        if model_identifier is None:
            model_identifier = self.model_path or self.model_name
            
        logger.info(f"Loading model: {model_identifier}")
        self._log_memory_usage("Before loading model: ")
        
        try:
            # Get model configuration
            from src.config.models import get_model_config, ModelType
            model_config = get_model_config(model_identifier, ModelType.CHAT)
            
            # Use model_id from config which includes the full path
            model_id = model_config.model_id
            logger.info(f"Using model ID from config: {model_id}")
            
            # Check for Apple Silicon MPS
            mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            
            # Configure device settings
            device_map = "auto"
            torch_dtype = torch.bfloat16
            
            if mps_available:
                logger.info("Using Apple MPS (Metal) for acceleration")
                device = torch.device("mps")
                device_map = {"": device}
                torch_dtype = torch.float32  # MPS works better with float32
                load_in_4bit = False  # 4-bit not supported on MPS
                load_in_8bit = False
            elif torch.cuda.is_available():
                logger.info("Using CUDA for acceleration")
                device_map = "auto"
                torch_dtype = torch.bfloat16
                load_in_4bit = True  # Prefer 4-bit for CUDA
                load_in_8bit = False
            else:
                logger.warning("No GPU acceleration available. Using CPU with optimizations.")
                device_map = None
                torch_dtype = torch.float32
                load_in_4bit = False
                load_in_8bit = False
            
            # Configure tokenizer with proper padding
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,  # Use model_id from config
                padding_side="left",
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure model kwargs with optimizations
            model_kwargs = {
                "device_map": device_map,
                "torch_dtype": torch_dtype,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "offload_folder": "offload",
            }
            
            # Apply 4-bit or 8-bit quantization if supported
            if load_in_4bit:
                model_kwargs["load_in_4bit"] = True
                model_kwargs["bnb_4bit_compute_dtype"] = torch_dtype
                model_kwargs["bnb_4bit_quant_type"] = "nf4"
                model_kwargs["bnb_4bit_use_double_quant"] = True
                logger.info("Using 4-bit quantization")
            elif load_in_8bit:
                model_kwargs["load_in_8bit"] = True
                logger.info("Using 8-bit quantization")
            
            # Special handling for Gemma
            if 'gemma' in model_identifier.lower():
                model_kwargs["attn_implementation"] = "eager"
            
            # Load the model with progress tracking
            logger.info("Loading model, this may take a few minutes...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )
            
            # Set model to eval mode
            self.model.eval()
            
            # Move model to device if not using device_map
            if device_map is None:
                self.model = self.model.to("cpu")
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self._log_memory_usage("After loading model: ")
            logger.info(f"Model loaded successfully on device: {self.model.device}")
            
        except Exception as e:
            logger.error(f"Error loading model {model_identifier}: {str(e)}")
            self._log_memory_usage("Error loading model: ")
            raise

    async def generate_response(
        self,
        messages: Union[str, List[Dict[str, str]]],
        max_length: int = 1000,
        temperature: float = 0.7,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate a response for the given prompt or chat messages.
        
        Args:
            messages: Either a string prompt or a list of message dictionaries with 'role' and 'content' keys
            max_length: Maximum length of the generated response in tokens
            temperature: Sampling temperature (0.0 to 2.0)
            stream: Whether to stream the response (not yet implemented)
            **kwargs: Additional generation parameters
            
        Returns:
            If input is a string, returns a string response.
            If input is a list of messages, returns a dictionary with 'content' and 'usage' keys.
        """
        logger.info("Starting generate_response")
        try:
            if not self.model or not self.tokenizer:
                error_msg = "Model or tokenizer not loaded. Call load_model() first."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Handle both string prompts and chat messages
            if isinstance(messages, str):
                logger.debug("Processing string prompt")
                prompt = messages
                is_chat = False
            else:
                logger.debug(f"Processing {len(messages)} chat messages")
                # Format chat messages into a single prompt
                prompt = self._format_chat_prompt(messages)
                is_chat = True
            
            logger.debug(f"Generated prompt (first 200 chars): {prompt[:200]}...")
            
            # Tokenize the prompt
            logger.debug("Tokenizing prompt...")
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096 - max_length,  # Leave room for response
                return_token_type_ids=False
            ).to(self.device)
            
            logger.debug(f"Input tensors prepared on device: {self.device}")
            
            # Prepare generation parameters
            generation_params = {
                'max_new_tokens': max_length,
                'temperature': temperature,
                'do_sample': temperature > 0,
                'top_p': 0.9 if temperature > 0 else 1.0,
                'pad_token_id': self.tokenizer.eos_token_id,
            }
            
            # Filter out any None values and update with any additional kwargs
            generation_params = {k: v for k, v in generation_params.items() if v is not None}
            
            # Add any additional kwargs that don't conflict
            for k, v in kwargs.items():
                if k not in generation_params and v is not None:
                    generation_params[k] = v
            
            logger.debug(f"Generation parameters: {generation_params}")
            
            # Generate response
            logger.info("Generating response...")
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **generation_params
                    )
                logger.debug("Successfully generated response")
            except Exception as e:
                logger.error(f"Error during model.generate(): {str(e)}", exc_info=True)
                raise
            
            # Decode the response
            logger.debug("Decoding response...")
            response_text = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            logger.debug(f"Decoded response (first 200 chars): {response_text[:200]}...")
            
            # For chat responses, return a structured format
            if is_chat:
                # Calculate token usage
                prompt_tokens = inputs.input_ids.shape[1]
                completion_tokens = len(outputs[0]) - prompt_tokens
                
                logger.info(f"Generated response with {completion_tokens} tokens")
                
                return {
                    "content": response_text,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }
            else:
                # For simple prompts, just return the text
                logger.info("Returning simple prompt response")
                return response_text
                
        except Exception as e:
            error_msg = f"Error in generate_response: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if is_chat:
                return {
                    "content": f"Error: {error_msg}",
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }
            return f"Error: {error_msg}"
    
    def _format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages into a single prompt string for Gemma.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            Formatted prompt string for Gemma
        """
        # Special handling for Gemma's chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                # Convert messages to the format expected by apply_chat_template
                formatted_messages = []
                for msg in messages:
                    role = msg['role']
                    # Map roles to Gemma's expected format
                    if role == 'system':
                        formatted_messages.append({"role": "user", "content": msg['content']})
                        formatted_messages.append({"role": "assistant", "content": "Understood."})
                    else:
                        formatted_messages.append({"role": role, "content": msg['content']})
                
                # Apply Gemma's chat template
                return self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Error applying chat template: {str(e)}")
                # Fall through to default formatting
        
        # Fallback formatting if chat template is not available
        prompt_parts = []
        for message in messages:
            role = message['role']
            content = message['content'].strip()
            
            if role == 'system':
                prompt_parts.append(f"<start_of_turn>system\n{content}<end_of_turn>")
            elif role == 'user':
                prompt_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == 'assistant':
                prompt_parts.append(f"<start_of_turn>assistant\n{content}<end_of_turn>")
        
        # Add the assistant's response prefix
        prompt_parts.append("<start_of_turn>assistant\n")
        return "".join(prompt_parts)

    async def summarize_case(self, issue: str, root_cause: str, resolution: str = '') -> str:
        """
        Generate a summary for a support case.
        
        Args:
            issue: The main issue description
            root_cause: Analysis of the root cause
            resolution: Resolution steps (optional)
            
        Returns:
            str: Generated summary
        """
        prompt = (
            "### Instruction: Summarize this support case in 3-5 sentences "
            "focusing on the core issue and resolution.\n\n"
            f"### Issue:\n{issue}\n\n"
            f"### Root Cause:\n{root_cause}\n\n"
            f"### Resolution:\n{resolution if resolution else 'No resolution provided'}\n\n"
            "### Summary:"
        )
        return await self.generate_response(prompt, max_length=300)

    async def search_similar_cases(
        self, 
        query: str, 
        k: int = 10,  
        min_score: float = 0.7,  
        include_solutions: bool = False,
        **kwargs
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Search for similar cases and optionally generate AI-powered solutions.
        
        When include_solutions is True:
        - Returns a dictionary with 'ai_summary'
        
        When include_solutions is False:
        - Returns a list of cases with case task number, issue, resolution, and support steps
        """
        logger.info(f"Searching for cases similar to: '{query}' with k={k}, min_score={min_score}")
        
        try:
            if not self.case_service:
                raise ValueError("CaseService not provided. Cannot perform vector search.")
            
            # Get similar documents from vector store
            similar_docs = await self.case_service.find_similar_cases(
                query=query,
                k=k,
                min_score=min_score,
                **kwargs
            )
            
            if not similar_docs:
                logger.info("No similar cases found in the vector database.")
                return {"ai_summary": "No similar cases found."} if include_solutions else []
                
            results = []
            case_task_numbers = []
            case_details = []
            
            for doc in similar_docs:
                try:
                    # Handle both document objects and dictionaries
                    if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                        metadata = doc.metadata
                        content = doc.page_content
                    elif isinstance(doc, dict):
                        metadata = doc.get('metadata', {})
                        # Get content from 'content' field first, fallback to 'page_content'
                        content = doc.get('content', doc.get('page_content', ''))
                        
                        # If we got content from 'content' field, update metadata with any missing fields
                        if 'content' in doc and isinstance(metadata, dict):
                            # Extract fields from content if they're not in metadata
                            content_fields = {
                                'issue': self._extract_field_from_content(content, 'ISSUE'),
                                'root_cause': self._extract_field_from_content(content, 'ROOT_CAUSE'),
                                'resolution': self._extract_field_from_content(content, 'RESOLUTION'),
                                'steps_support': self._extract_field_from_content(content, 'STEPS_SUPPORT')
                            }
                            # Update metadata with extracted fields if they don't exist
                            for field, value in content_fields.items():
                                if value and field not in metadata:
                                    metadata[field] = value
                    else:
                        logger.warning(f"Unexpected document type: {type(doc)}")
                        continue
                    
                    # Get case number from metadata
                    case_number = metadata.get('case_task_number')
                    if not case_number:
                        logger.warning(f"Document has no case_task_number in metadata")
                        continue
                        
                    logger.debug(f"Processing case: {case_number}")
                    
                    # Extract fields from metadata with fallback to empty string
                    issue = metadata.get('issue', '')
                    root_cause = metadata.get('root_cause', '')
                    resolution = metadata.get('resolution', '')
                    steps_support = metadata.get('steps_support', metadata.get('steps_support', ''))
                    
                    # If any field is missing, try to extract from content
                    if not all([issue, root_cause, resolution, steps_support]) and content:
                        logger.debug("Some fields missing from metadata, trying to extract from content")
                        issue = issue or self._extract_field_from_content(content, 'ISSUE')
                        root_cause = root_cause or self._extract_field_from_content(content, 'ROOT_CAUSE')
                        resolution = resolution or self._extract_field_from_content(content, 'RESOLUTION')
                        steps_support = steps_support or self._extract_field_from_content(content, 'STEPS_SUPPORT')
                    
                    # Log if we still couldn't find all fields
                    if not all([issue, root_cause, resolution, steps_support]):
                        missing = [f for f, v in [
                            ('issue', issue),
                            ('root_cause', root_cause),
                            ('resolution', resolution),
                            ('steps_support', steps_support)
                        ] if not v]
                        logger.warning(f"Could not extract fields: {', '.join(missing)}")
                    
                    # Add to case task numbers list
                    case_task_numbers.append(str(case_number))
                    
                    # Prepare the result
                    result = {
                        'case_number': case_number,
                        'similarity_score': doc.get('score', 0.0),
                        'issue': issue,
                        'root_cause': root_cause,
                        'resolution': resolution,
                        'support_steps': steps_support,
                        'metadata': {
                            'case_task_number': case_number,
                            'issue': issue,
                            'root_cause': root_cause,
                            'resolution': resolution,
                            'support_steps': steps_support,
                            'parent_case': metadata.get('parent_case'),
                            'created_at': metadata.get('created_at'),
                            'updated_at': metadata.get('updated_at')
                        }
                    }
                    
                    # Collect case details for AI summary
                    if include_solutions:
                        case_details.append({
                            'case_number': case_number,
                            'issue': issue,
                            'resolution': resolution,
                            'support_steps': steps_support,
                            'root_cause': root_cause
                        })
                        continue
                        
                    # Add to results list
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error processing search result: {str(e)}", exc_info=True)
                    continue
            
            # Generate AI summary if include_solutions is True
            ai_summary = ""
            if include_solutions and case_details:
                try:
                    # Prepare cases text for AI analysis
                    cases_text = "\n\n".join([
                        f"Case {case['case_number']}:\n"
                        f"Root Cause: {case.get('root_cause', 'Not specified')}\n"
                        f"Resolution: {case.get('resolution', 'Not specified')}\n"
                        f"Support Steps: {case.get('support_steps', 'Not specified')}"
                        for case in case_details
                    ])
                    
                    prompt = (
                        "Analyze the following support cases and provide a comprehensive summary "
                        "that focuses on the root causes, resolutions, and support steps. "
                        "Your summary should be based ONLY on the information provided in these cases.\n\n"
                        f"Search Query: {query}\n\n"
                        f"Relevant Cases:\n{cases_text}\n\n"
                        "Provide a well-structured summary with these sections, using ONLY the information from the cases above:\n"
                        "1. Root Causes Identified (list the main causes mentioned in the cases as bullet points without specifying case task numbers)\n"
                        "2. Resolution Steps (summarize the key steps taken to resolve the issues as bullet points without specifying case task numbers)\n"
                        "3. Support Process (describe the support process followed as bullet points without specifying case task numbers)\n"
                        "4. Case References (list of case numbers used in the analysis)\n\n"
                    )
                    
                    # Generate the summary using the LLM
                    summary_response = await self.generate_response(
                        messages=[{"role": "user", "content": prompt}],
                        max_length=1500,  # Increased length for more detailed summary
                        temperature=0.2  # Lower temperature for more focused responses
                    )
                    
                    if isinstance(summary_response, dict) and 'content' in summary_response:
                        ai_summary = summary_response['content'].strip()
                    else:
                        ai_summary = "Unable to generate summary at this time."
                        
                except Exception as e:
                    logger.error(f"Error generating AI summary: {str(e)}", exc_info=True)
                    ai_summary = "An error occurred while generating the summary."
            
            # Return appropriate response based on include_solutions flag
            if include_solutions:
                return {
                    'ai_summary': ai_summary,
                    'query': query,
                    'total_results': len(case_task_numbers)
                }
            else:
                return results
                
        except Exception as e:
            logger.error(f"Error in search_similar_cases: {str(e)}", exc_info=True)
            # Return empty result in the expected format
            return {"ai_summary": "An error occurred while processing your request."} if include_solutions else []
    
    def _extract_field_from_content(self, content: str, field_name: str) -> str:
        """
        Extract a field from the content string.
        
        Args:
            content: The content string to extract from
            field_name: The name of the field to extract
            
        Returns:
            The extracted field value or an empty string if not found
        """
        if not content:
            return ''
            
        # Try different patterns to match the field
        patterns = [
            rf'^{field_name.upper()}:\s*(.*?)(?:\n|$)',  # UPPERCASE:
            rf'^{field_name.title()}:\s*(.*?)(?:\n|$)',  # Title Case:
            rf'^{field_name.replace("_", " ").title()}:\s*(.*?)(?:\n|$)'  # Title Case with spaces:
        ]
        
        for pattern_str in patterns:
            try:
                pattern = re.compile(pattern_str, re.MULTILINE | re.DOTALL)
                match = pattern.search(content)
                if match:
                    result = match.group(1).strip()
                    logger.debug(f"Extracted {field_name} using pattern: {pattern_str}")
                    return result
            except Exception as e:
                logger.debug(f"Error with pattern {pattern_str}: {str(e)}")
        
        logger.debug(f"Could not find {field_name} in content")
        return ''

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
                messages=[{"role": "user", "content": prompt}],
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
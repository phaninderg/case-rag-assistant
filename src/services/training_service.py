import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from typing import List, Dict, Any, Optional
import os
from pathlib import Path

from src.config.settings import settings

logger = logging.getLogger(__name__)

class TrainingService:
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the training service.
        
        Args:
            model_name: Name of the model to train. If None, uses the default from settings.
        """
        self.model_name = model_name or settings.default_llm
        self.tokenizer = None
        self.model = None
        # Initialize LLMService lazily to avoid circular imports
        self._llm_service = None
    
    @property
    def llm_service(self):
        if self._llm_service is None:
            from src.services.llm_service import LLMService
            self._llm_service = LLMService(model_name=self.model_name)
        return self._llm_service
    
    def prepare_training_data(self, cases: List[Dict[str, Any]]) -> Dataset:
        """
        Prepare case data for training using all available fields.
        
        Args:
            cases: List of case dictionaries containing case data
            
        Returns:
            Dataset: Prepared dataset for training with all fields included
        """
        texts = []
        
        for case in cases:
            # Build a structured prompt with all available fields
            prompt_parts = ["Case Information:"]
            
            # Include all fields that have values
            for field, value in case.items():
                if value:  # Only include non-empty fields
                    if isinstance(value, dict):
                        # Handle nested dictionaries
                        nested_items = [f"{k}: {v}" for k, v in value.items() if v]
                        if nested_items:
                            prompt_parts.append(f"{field.capitalize()}:\n  " + "\n  ".join(nested_items))
                    elif isinstance(value, list):
                        # Handle lists (like tags)
                        if value:
                            prompt_parts.append(f"{field.capitalize()}: " + ", ".join(str(v) for v in value))
                    else:
                        prompt_parts.append(f"{field.capitalize()}: {value}")
            
            # Join all parts with newlines for better readability
            full_text = "\n".join(prompt_parts) + "\n"
            texts.append(full_text)
        
        # Create HuggingFace Dataset
        dataset = Dataset.from_dict({
            'text': texts
        })
        
        return dataset
    
    def train_model(self, cases: List[Dict[str, Any]], output_dir: str):
        """
        Train a model on case data.
        
        Args:
            cases: List of case dictionaries containing training data
            output_dir: Directory to save the trained model
        """
        try:
            # Set device to CPU explicitly
            import torch
            device = torch.device("cpu")
            
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Starting model training with {len(cases)} cases")
            
            # Prepare training data
            train_dataset = self.prepare_training_data(cases)
            logger.info(f"Prepared training dataset with {len(train_dataset)} examples")
            
            # Initialize tokenizer with proper configuration
            logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                padding_side="right",
                use_fast=True
            )
            
            # Configure padding token - ensure it's set before model loading
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    # If no EOS token, add a new padding token
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            # Initialize model on CPU
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            self.model = self.model.to(device)  # Move model to CPU
            
            # Resize token embeddings if we added a new pad token
            if self.tokenizer.pad_token == '[PAD]':
                self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Tokenize the dataset
            logger.info("Tokenizing dataset...")
            def tokenize_function(examples):
                # Ensure we have a list of strings
                texts = [text for text in examples['text'] if isinstance(text, str)]
                if not texts:
                    return {}
                    
                # Tokenize with padding and truncation
                tokenized = self.tokenizer(
                    texts,
                    padding='max_length',
                    max_length=128,
                    truncation=True,
                    return_tensors='pt'
                )
                
                # For language modeling, labels should be the same as input_ids
                tokenized['labels'] = tokenized['input_ids'].clone()
                return tokenized
                
            tokenized_dataset = train_dataset.map(
                tokenize_function,
                batched=True,
                batch_size=1,  # Process one example at a time
                remove_columns=['text']
            )
            
            # Filter out any empty examples that might have been created
            tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) > 0)
            
            if len(tokenized_dataset) == 0:
                raise ValueError("No valid examples found after tokenization. Check your input data format.")
            
            logger.info(f"Tokenized dataset contains {len(tokenized_dataset)} examples")
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=str(output_path),
                num_train_epochs=1,  # Start with 1 epoch
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=1,
                warmup_steps=5,  # Reduced warmup steps
                weight_decay=0.01,
                logging_dir=str(output_path / "logs"),
                logging_steps=1,
                save_strategy="steps",
                save_steps=10,
                evaluation_strategy="no",  # Disable evaluation for now
                load_best_model_at_end=False,  # Disable for initial testing
                no_cuda=True,  # Force CPU usage
                dataloader_num_workers=0,  # Set to 0 for CPU
                report_to=[],  # Disable all reporting
                save_total_limit=1,
                fp16=False,  # Disable mixed precision
                bf16=False,   # Disable bfloat16
                # Device settings
                local_rank=-1,
                ddp_find_unused_parameters=False,
                remove_unused_columns=False,  # Important for custom datasets
                # Enable language modeling
                prediction_loss_only=True
            )
            
            # Initialize DataCollator for language modeling
            from transformers import DataCollatorForLanguageModeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False  # Use causal language modeling (not masked language modeling)
            )
            
            # Initialize Trainer with explicit device settings
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator
            )
            
            # Start training
            logger.info("Starting training...")
            trainer.train()
            
            # Save the model and tokenizer
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            logger.info(f"Training completed. Model saved to {output_dir}")
            return {"status": "success", "output_dir": str(output_path.absolute())}
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}", exc_info=True)
            raise

    def load_trained_model(self, model_path: str):
        """
        Load a previously trained model.
        
        Args:
            model_path: Path to the trained model directory
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.llm_service.llm = self.model
        logger.info(f"Loaded trained model from {model_path}")

import logging
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import pandas as pd

from src.config.settings import settings
from src.config.models import DEFAULT_LLM

logger = logging.getLogger(__name__)

class TrainingService:
    def __init__(self, base_model: str = None, llm_service=None):
        """
        Initialize the training service.
        
        Args:
            base_model: Base model name or path to fine-tune. Defaults to 'llama3-8b'
            llm_service: Optional LLMService instance for inference during training
        """
        self.base_model = base_model or DEFAULT_LLM
        self.llm_service = llm_service
        self.device = self._get_device()
        self.tokenizer = None
        self.model = None
        logger.info(f"Initialized TrainingService with base model: {self.base_model}")
    
    def _get_device(self):
        """Get the appropriate device for training."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def _load_model(self):
        """Lazily load the model and tokenizer."""
        if self.model is None or self.tokenizer is None:
            logger.info(f"Loading tokenizer and model for {self.base_model}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.base_model,
                    trust_remote_code=True
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Use bfloat16 if available for faster training
                torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    torch_dtype=torch_dtype,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                logger.info(f"Model loaded on device: {self.model.device}")
                
            except Exception as e:
                logger.error(f"Failed to load model {self.base_model}: {str(e)}")
                raise
    
    def prepare_training_data(self, cases: List[Dict[str, Any]]) -> Dataset:
        """Prepare case task data for instruction fine-tuning using the new fields."""
        training_examples = []
        
        for case in cases:
            # Safely convert all values to strings and handle None/NaN values
            def safe_str(value) -> str:
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    return 'N/A'
                return str(value).strip()
            
            # Get values with safe string conversion
            issue = safe_str(case.get('issue'))
            root_cause = safe_str(case.get('root_cause'))
            resolution = safe_str(case.get('resolution'))
            steps_support = safe_str(case.get('steps_support'))
            
            # Format input with all relevant fields
            input_text = (
                f"Issue: {issue}\n"
                f"Root Cause: {root_cause}\n"
                f"Steps Taken: {steps_support}"
            )
            
            # Create instruction for resolution generation
            if resolution and resolution != 'N/A':
                training_examples.append({
                    "instruction": "Based on the issue and root cause, provide a detailed resolution.",
                    "input": input_text,
                    "output": resolution
                })
            
            # Create instruction for root cause analysis
            if root_cause and root_cause != 'N/A':
                training_examples.append({
                    "instruction": "Analyze the issue and provide the root cause analysis.",
                    "input": f"Issue: {issue}",
                    "output": root_cause
                })
            
            # Create instruction for support steps
            if steps_support and steps_support != 'N/A':
                training_examples.append({
                    "instruction": "Provide the steps taken to support this issue.",
                    "input": f"Issue: {issue}\nRoot Cause: {root_cause}",
                    "output": steps_support
                })
            
            # Create instruction for issue summarization
            if issue and issue != 'N/A':
                training_examples.append({
                    "instruction": "Summarize this case issue concisely.",
                    "input": issue,
                    "output": issue[:500]  # First 500 characters as summary
                })
        
        return Dataset.from_list(training_examples)

    def tokenize_function(self, examples):
        """Tokenize the training examples."""
        prompts = [
            f"### Instruction: {inst}\n\n### Input: {inp}\n\n### Response: {out}"
            for inst, inp, out in zip(
                examples['instruction'],
                examples['input'],
                examples['output']
            )
        ]
        
        return self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

    def train(
        self,
        cases: List[Dict[str, Any]],
        output_dir: str,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 2,
        learning_rate: float = 2e-5,
        **kwargs
    ) -> str:
        """
        Train the model on case task data.
        
        Args:
            cases: List of case task dictionaries
            output_dir: Directory to save the trained model
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            learning_rate: Learning rate for training
            **kwargs: Additional training arguments
            
        Returns:
            str: Path to the saved model
        """
        try:
            # Prepare training data
            dataset = self.prepare_training_data(cases)
            tokenized_dataset = dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            # Load model if not already loaded
            self._load_model()
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                learning_rate=learning_rate,
                save_strategy="epoch",
                evaluation_strategy="no",
                logging_steps=1,
                save_total_limit=1,
                remove_unused_columns=False,
                no_cuda=not torch.cuda.is_available(),
                **kwargs
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False
                )
            )
            
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Train and save
            logger.info("Starting training...")
            trainer.train()
            
            logger.info(f"Training complete. Saving model to {output_path}")
            trainer.save_model(str(output_path))
            self.tokenizer.save_pretrained(str(output_path))
            
            return str(output_path.absolute())
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}", exc_info=True)
            raise

    def load_trained_model(self, model_path: str):
        """
        Load a trained model.
        
        Args:
            model_path: Path to the trained model
        """
        try:
            logger.info(f"Loading trained model from {model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            raise

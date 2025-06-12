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

from src.config.settings import settings

logger = logging.getLogger(__name__)

class TrainingService:
    def __init__(self, base_model: str = "distilgpt2", llm_service=None):
        """
        Initialize the training service.
        
        Args:
            base_model: Base model name or path to fine-tune
            llm_service: Optional LLMService instance for inference during training
        """
        self.base_model = base_model
        self.llm_service = llm_service
        self.device = self._get_device()
        logger.info(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = None
    
    def _get_device(self):
        """Force CPU usage for stability."""
        logger.info("Forcing CPU usage for training stability")
        return "cpu"
    
    def prepare_training_data(self, cases: List[Dict[str, Any]]) -> Dataset:
        """Prepare case data for instruction fine-tuning."""
        training_examples = []
        
        for case in cases:
            # Format input
            input_text = (
                f"Subject: {case.get('subject', '')}\n"
                f"Description: {case.get('description', '')}"
            )
            
            # Create instruction for summarization
            if 'summary' in case:
                training_examples.append({
                    "instruction": "Summarize this support case in 3-5 sentences focusing on the core issue.",
                    "input": input_text,
                    "output": case['summary']
                })
            
            # Create instruction for solution generation
            if 'close_notes' in case:
                training_examples.append({
                    "instruction": "Based on this case, provide a solution for the issue.",
                    "input": input_text,
                    "output": case['close_notes']
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
        Train the model on case data.
        
        Args:
            cases: List of case dictionaries
            output_dir: Directory to save the trained model
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            learning_rate: Learning rate for training
            **kwargs: Additional training arguments
            
        Returns:
            str: Path to the saved model
        """
        try:
            # Ensure we're using CPU
            torch.set_default_device("cpu")
            
            # Prepare data
            dataset = self.prepare_training_data(cases)
            tokenized_dataset = dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            # Initialize model with CPU device
            logger.info(f"Loading model: {self.base_model} on CPU")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float32
            ).to("cpu")  # Explicitly move to CPU
            
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(output_path),
                num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                learning_rate=learning_rate,
                save_strategy="epoch",
                evaluation_strategy="no",
                logging_steps=1,
                save_total_limit=1,
                remove_unused_columns=False,
                no_cuda=True,  # Disable CUDA
                use_mps_device=False,  # Disable MPS
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
            
            # Train and save
            logger.info("Starting training on CPU...")
            trainer.train()
            
            logger.info("Training complete. Saving model...")
            trainer.save_model(str(output_path))
            self.tokenizer.save_pretrained(str(output_path))
            
            logger.info(f"Model saved to {output_path}")
            return str(output_path)
            
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

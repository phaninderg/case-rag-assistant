# Case RAG Assistant

A powerful Retrieval-Augmented Generation (RAG) system for case task analysis, similarity search, and intelligent case management. This application helps support engineers quickly find relevant information from past cases and generate intelligent responses using state-of-the-art language models.

## 🚀 Current Status

✅ **Core Features Implemented**
- Multi-model LLM support (OpenAI, HuggingFace, LLaMA)
- Vector-based semantic search with ChromaDB
- Case management with metadata support
- RESTful API with pagination and filtering
- Asynchronous request handling
- Configurable model parameters
- Fine-tuning support for custom models
- CPU-optimized training pipeline

🔄 **Recent Updates**
- Added fine-tuning capabilities for case-specific models
- Improved tokenization and data preprocessing
- Enhanced error handling and logging
- CPU-optimized training pipeline for Apple Silicon
- Support for custom case data formatting
- Integration with HuggingFace Transformers for fine-tuning

## 🛠️ Features

### Core Functionality
- **Multi-Model Support**: Choose from various LLM providers (OpenAI, HuggingFace, LLaMA)
- **Semantic Search**: Find similar cases using vector similarity search
- **Case Management**: Store, retrieve, and manage case data with metadata
- **Fine-Tuning**: Train custom models on your case data
- **RESTful API**: Comprehensive API documentation with Swagger UI

### Technical Features
- **CPU Optimization**: Runs efficiently on CPU-only environments
- **Asynchronous Processing**: Efficient handling of multiple concurrent requests
- **Flexible Storage**: Local vector store with ChromaDB
- **Environment Configuration**: Easy setup with environment variables
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip (Python package manager)
- [Optional] For training: At least 8GB RAM recommended

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/case-rag-assistant.git
   cd case-rag-assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Running the Application

1. Start the FastAPI server:
   ```bash
   uvicorn src.app:app --reload
   ```

2. Access the API documentation:
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Training a Custom Model

1. Prepare your case data in JSON format (see `cases.json` for example)

2. Start the training process:
   ```bash
   curl -X 'POST' \
     'http://localhost:8000/train' \
     -H 'accept: application/json' \
     -F 'cases_file=@./path/to/your/cases.json'
   ```

3. Monitor training progress in the logs

## 🧩 Project Structure

```
case-rag-assistant/
├── src/
│   ├── config/         # Configuration files
│   ├── models/         # Data models and embeddings
│   ├── services/       # Core services
│   │   ├── case_service.py    # Case management
│   │   ├── llm_service.py     # LLM integration
│   │   └── training_service.py # Model training
│   ├── utils/          # Utility functions
│   └── app.py          # FastAPI application
├── tests/              # Test cases
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 📚 API Documentation

### Endpoints

- `GET /api/cases` - List all cases
- `GET /api/cases/{case_id}` - Get case details
- `POST /api/cases/search` - Search for similar cases
- `POST /train` - Train a new model on case data
- `GET /models` - List available models

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [LangChain](https://python.langchain.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [ChromaDB](https://www.trychroma.com/)

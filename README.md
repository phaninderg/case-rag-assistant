# Case RAG Assistant

A powerful Retrieval-Augmented Generation (RAG) system for case task analysis, similarity search, and intelligent case management. This application helps support engineers quickly find relevant information from past cases and generate intelligent responses using state-of-the-art language models.

## 🚀 Current Status

✅ **Core Features Implemented**
- Multi-model LLM support (Gemma-2b-it, HuggingFace, LLaMA)
- Vector-based semantic search with ChromaDB
- Case management with metadata support
- RESTful API with pagination and filtering
- Asynchronous request handling
- Configurable model parameters
- CPU-optimized inference pipeline

🔄 **Recent Updates**
- Upgraded to Gemma-2b-it as the default LLM
- Enhanced tokenization and data preprocessing
- Improved error handling and logging
- CPU-optimized inference pipeline for Apple Silicon
- Support for custom case data formatting
- Integration with HuggingFace Transformers for inference

## 🛠️ Features

### Core Functionality
- **Multi-Model Support**: Choose from various LLM providers (Gemma-2b-it, HuggingFace, LLaMA)
- **Semantic Search**: Find similar cases using vector similarity search
- **Case Management**: Store, retrieve, and manage case data with metadata
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
- [Optional] For inference: At least 8GB RAM recommended

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

## 🧩 Project Structure

```
case-rag-assistant/
├── src/
│   ├── config/         # Configuration files
│   ├── models/         # Data models and embeddings
│   ├── services/       # Core services
│   │   ├── case_service.py    # Case management
│   │   ├── llm_service.py     # LLM integration
│   │   └── embedding_service.py # Vector embeddings
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

- [Google Gemma](https://ai.google.dev/gemma)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [ChromaDB](https://www.trychroma.com/)

## 🚀 Model Selection and Embedding Capabilities

### Model Selection

The system supports multiple LLM models with Gemma-2b-it as the default. You can switch between different models using the `model_name` parameter in the API requests.

### Embedding Models

The system supports different embedding models for vector search. The default is set to 'all-mpnet-base-v2' for optimal performance.

```python
# Example: Update embedding model
embedding_service.update_embedding_model(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_path=None  # Optional local path
)
```

### Using Gemma-2b-it

To use Gemma-2b-it as your LLM, ensure you have the necessary access and set it in your configuration:

```json
{
  "model_name": "google/gemma-2b-it"
}
```

## 🔧 Configuration

### Environment Variables

```env
# Required
DEFAULT_LLM=google/gemma-2b-it
DEFAULT_EMBEDDING=sentence-transformers/all-mpnet-base-v2

# Optional
HUGGINGFACE_API_KEY=your-hf-token
MODEL_CACHE_DIR=./model_cache

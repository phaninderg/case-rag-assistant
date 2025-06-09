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

🔄 **Recent Updates**
- Upgraded to latest dependency versions
- Improved error handling and logging
- Enhanced search functionality with similarity scoring
- Added pagination and sorting for case listings
- Integration with HuggingFace Hub for model management

## 🛠️ Features

- **Multi-Model Support**: Choose from various LLM providers (OpenAI, HuggingFace, LLaMA) and embedding models
- **Semantic Search**: Find similar cases using vector similarity search with configurable thresholds
- **AI-Powered Analysis**: Generate intelligent case summaries and relevance analysis
- **RESTful API**: Comprehensive API documentation with Swagger UI and ReDoc
- **Asynchronous Processing**: Efficient handling of multiple concurrent requests
- **Flexible Storage**: Local vector store with ChromaDB and file-based case storage
- **Environment Configuration**: Easy setup with environment variables

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip (Python package manager)
- [Optional] NVIDIA GPU for local model acceleration

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

4. Create a `.env` file with your API keys (copy from `.env.example`):
   ```bash
   # Required for OpenAI models
   OPENAI_API_KEY=your-openai-key
   
   # Required for HuggingFace Hub models
   HUGGINGFACE_API_KEY=your-hf-key
   
   # Model Defaults
   DEFAULT_LLM=tinyllama  # or gpt-3.5-turbo, mistral-7b, etc.
   DEFAULT_EMBEDDING=all-mpnet-base-v2
   
   # Server Configuration
   HOST=0.0.0.0
   PORT=8000
   DEBUG=true
   LOG_LEVEL=INFO
   
   # Data Directories
   DATA_DIR=./data
   ```

## 🏃 Running the Application

Start the FastAPI development server:
```bash
uvicorn src.app:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🧩 API Endpoints

### Cases
- `GET /api/cases` - List all cases with pagination
- `GET /api/cases/{case_id}` - Get case details
- `POST /api/cases` - Create a new case
- `POST /api/cases/search` - Search for similar cases
- `POST /api/cases/{case_id}/summarize` - Generate case summary

### Models
- `GET /api/models` - List available models
- `POST /api/llm/config` - Update LLM configuration

## 🏗️ Project Structure

```
case-rag-assistant/
├── src/
│   ├── app.py                 # Main FastAPI application
│   ├── config/                # Configuration settings
│   │   ├── __init__.py
│   │   ├── models.py         # Model configurations
│   │   └── settings.py       # Application settings
│   │
│   ├── models/             # ML models and embeddings
│   │   ├── __init__.py
│   │   ├── embeddings.py     # Embedding service
│   │   └── factory.py        # Model factory
│   │
│   ├── services/           # Business logic
│   │   ├── __init__.py
│   │   ├── case_service.py   # Case management
│   │   ├── llm_service.py    # LLM interactions
│   │   └── web_crawler.py    # Web crawling functionality
│   │
│   └── utils/              # Utility functions
│       ├── __init__.py
│       └── helpers.py        # Helper functions
│
├── data/                   # Case data storage (gitignored)
│   ├── cases/              # Case JSON files
│   └── embeddings/         # Vector database storage
├── .env.example           # Example environment variables
├── .gitignore
├── README.md
└── requirements.txt       # Python dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/) for LLM orchestration
- [Chroma](https://www.trychroma.com/) for vector storage
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [HuggingFace](https://huggingface.co/) for open-source models

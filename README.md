# Case RAG Assistant

A powerful Retrieval-Augmented Generation (RAG) system for case task analysis, similarity search, and intelligent case management. This application helps support engineers quickly find relevant information from past cases and generate intelligent responses using state-of-the-art language models.

## 🚀 Current Status

✅ **Core Features Implemented**
- Multi-model LLM support (Gemma-2b-it, TinyLlama, LLaMA)
- Vector-based semantic search with ChromaDB
- Case management with metadata support
- RESTful API with FastAPI
- Modern React frontend with Material-UI
- Model training and fine-tuning capabilities
- CPU/GPU optimized inference pipeline

## 🛠️ Features

### Backend
- **LLM Integration**: Support for multiple models (Gemma-2b-it, TinyLlama, LLaMA)
- **Vector Search**: Semantic search using ChromaDB
- **Case Management**: CRUD operations for case tasks
- **Model Training**: Fine-tune models on custom datasets
- **RESTful API**: Comprehensive API documentation with Swagger UI
- **Asynchronous Processing**: Efficient request handling

### Frontend
- **Interactive UI**: Built with React and Material-UI
- **Case Search**: Find similar cases using natural language
- **Chat Interface**: Interactive chat with the AI assistant
- **Responsive Design**: Works on desktop and mobile
- **Real-time Updates**: Dynamic UI with live search results

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 16+ (for frontend)
- pip (Python package manager)
- [Optional] CUDA for GPU acceleration

### Backend Setup

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

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

## 🏗️ Project Structure

```
case-rag-assistant/
├── src/                    # Backend source code
│   ├── config/            # Configuration files
│   ├── models/            # Data models and database schemas
│   ├── services/          # Core services
│   │   ├── case_service.py    # Case management logic
│   │   ├── llm_service.py     # LLM integration
│   │   └── training_service.py # Model training
│   └── app.py             # FastAPI application
├── frontend/              # React frontend
│   ├── public/            # Static assets
│   └── src/               # React source
│       ├── components/    # Reusable UI components
│       ├── services/      # API service layer
│       └── App.tsx        # Main application component
├── data/                  # Data storage
│   ├── cases/             # Case data files
│   └── vector_store/      # Embedding vectors
├── tests/                 # Test cases
├── sample_case_task.csv   # Sample case tasks for testing
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 📋 Sample Data

A sample CSV file (`sample_case_task.csv`) is included in the root directory with example case tasks that demonstrate the expected format for training data. The file includes the following columns:

- `issue`: Description of the reported problem
- `root_cause`: Analysis of what was causing the issue
- `resolution`: Steps taken to resolve the issue
- `steps_support`: Detailed support steps with numbered actions

You can use this file as a reference when preparing your own case task data for training or testing the system.

## 📚 API Endpoints

### Case Management
- `GET /api/cases` - List all cases
- `POST /api/cases` - Create a new case
- `GET /api/cases/{case_id}` - Get case details
- `PUT /api/cases/{case_id}` - Update a case
- `DELETE /api/cases/{case_id}` - Delete a case

### Search & Analysis
- `POST /api/cases/search` - Search for similar cases
- `GET /api/cases/{case_id}/summary` - Generate case summary
- `POST /api/chat` - Chat with the AI assistant

### Models
- `GET /api/models` - List available models
- `POST /api/train` - Train/fine-tune a model
- `POST /api/load-model` - Load a trained model

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [FastAPI](https://fastapi.tiangolo.com/) for the backend API
- [React](https://reactjs.org/) and [Material-UI](https://mui.com/) for the frontend

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

## 📸 Screenshots

### Frontend Interface
![Frontend Search Interface](/docs/images/frontend-search.png)
*The main search interface where users can query for similar cases*

![Frontend Chat Interface](/docs/images/frontend-chat.png)
*Interactive chat interface with the AI assistant*

### Backend Interface
![API Documentation](/docs/images/backend-api-docs.png)
*FastAPI Swagger documentation showing available endpoints*

![Prometheus Dashboard](/docs/images/prometheus-dashboard.png)
*Metrics dashboard for monitoring application performance*

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
- [Optional] Docker and Docker Compose for containerized deployment
- [Optional] Hugging Face API key for accessing gated models like Gemma

### Development Setup

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

3. Install Python dependencies for development:
   ```bash
   # Install PyTorch first (important for dependency resolution)
   pip install torch torchvision torchaudio
   
   # For macOS:
   pip install -r requirements.txt -r dev-requirements.txt
   
   # For Linux/Windows with CUDA:
   # pip install -r requirements-linux.txt -r dev-requirements.txt
   ```

4. Run the backend server:
   ```bash
   python -m src.app
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

### Setting Up Hugging Face API Key

To access gated models like Gemma, you need to set up a Hugging Face API key:

1. Create an account on [Hugging Face](https://huggingface.co/) if you don't have one
2. Go to your profile settings and create an API token
3. Accept the terms for the models you want to use (e.g., [Gemma](https://huggingface.co/google/gemma-2b-it))
4. Add your API key to the `.env` file:
   ```
   HUGGINGFACE_API_KEY=your_actual_api_key_here
   ```

**Note:** If you don't provide a valid API key, the application will automatically fall back to using TinyLlama, which is a non-gated model.

### Production Deployment

#### Using Docker (Recommended)

1. Set up your environment variables in the `.env` file:
   ```
   HUGGINGFACE_API_KEY=your_actual_api_key_here
   ```

2. Build the backend and frontend Docker images:
   ```bash
   # Build the backend image
   docker build -t case-rag-backend .
   
   # Build the frontend image
   docker build -t case-rag-frontend ./frontend
   ```

3. Create a Docker network for the application:
   ```bash
   docker network create case-rag-network
   ```

4. Run the backend container:
   ```bash
   docker run -d \
     --name case-rag-assistant \
     --network case-rag-network \
     -p 8000:8000 \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/offload:/app/offload \
     -v ~/.cache/huggingface:/root/.cache/huggingface \
     -e LOG_LEVEL=INFO \
     -e HOST=0.0.0.0 \
     -e PORT=8000 \
     -e DATA_DIR=/app/data \
     -e DEFAULT_LLM=google/gemma-2b-it \
     -e DEFAULT_EMBEDDING=all-mpnet-base-v2 \
     -e HUGGINGFACE_API_KEY=${HUGGINGFACE_API_KEY} \
     case-rag-backend
   ```

5. Run the frontend container:
   ```bash
   docker run -d \
     --name case-rag-frontend \
     --network case-rag-network \
     -p 3000:80 \
     case-rag-frontend
   ```

6. Run Prometheus for monitoring (optional):
   ```bash
   docker run -d \
     --name prometheus \
     --network case-rag-network \
     -p 9090:9090 \
     -v $(pwd)/prometheus.yml:/etc/prometheus/prometheus.yml \
     prom/prometheus:latest \
     --config.file=/etc/prometheus/prometheus.yml \
     --storage.tsdb.path=/prometheus \
     --web.console.libraries=/usr/share/prometheus/console_libraries \
     --web.console.templates=/usr/share/prometheus/consoles
   ```

7. Access the application:
   - Frontend UI: `http://localhost:3000`
   - Backend API: `http://localhost:8000`
   - API Documentation: `http://localhost:8000/docs`
   - Metrics Dashboard: `http://localhost:9090` (Prometheus)

#### Using docker-compose (Alternative)

If you prefer using docker-compose:

```bash
docker-compose up -d --build
```

#### Manual Deployment

1. Install production dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Build the frontend:
   ```bash
   cd frontend
   npm run build
   ```

3. Run with Gunicorn for production:
   ```bash
   gunicorn src.app:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
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
├── dev-requirements.txt   # Development dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── prometheus.yml        # Prometheus configuration
└── README.md             # This file
```

## 🚀 Performance Optimization

### Memory Management

The application includes several optimizations for efficient memory usage:

- **Model Cleanup**: Automatic cleanup of previous models when loading new ones
- **Garbage Collection**: Strategic use of Python's garbage collector to free memory
- **Quantization**: Support for 4-bit and 8-bit quantization of LLMs on compatible hardware
- **Device Optimization**: Automatic detection and utilization of available hardware (CUDA, MPS, CPU)

### Production Optimizations

- **Asynchronous Processing**: Non-blocking API endpoints for better concurrency
- **Efficient Data Loading**: Optimized CSV processing with memory-efficient options
- **Connection Pooling**: Database connection reuse for better performance
- **Caching**: Strategic caching of embeddings and model outputs

## 📊 Monitoring and Metrics

The application includes comprehensive monitoring capabilities:

### Prometheus Metrics

Access metrics at `/metrics` endpoint or through the Prometheus UI at `http://localhost:9090` when using Docker deployment.

Key metrics include:

- **HTTP Request Metrics**: Count, latency, and status codes
- **Model Performance**: Load time and inference latency
- **Memory Usage**: Real-time memory consumption tracking
- **Active Requests**: Concurrent request monitoring

### Health Checks

A health check endpoint is available at `/api/health` to verify system status.

## 📋 Sample Data

A sample CSV file (`sample_case_task.csv`) is included in the root directory with example case tasks that demonstrate the expected format for training data. The file includes the following columns:

- `issue`: Description of the reported problem
- `root_cause`: Analysis of what was causing the issue
- `resolution`: Steps taken to resolve the issue
- `steps_support`: Detailed support steps with numbered actions

You can use this file as a reference when preparing your own case task data for training or testing the system.

## 📚 API Endpoints

### Case Management
- `POST /api/cases` - Create a new case
- `GET /api/cases/{case_id}` - Get case details

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

# Document Assistant

> **AI-powered document assistant with MCP Server integration**

A sophisticated document analysis system that combines cutting-edge AI with modern architecture. Built for financial compliance analysis with CBUAE standards processing capabilities.

## ✨ Features

- **MCP Server**: High-performance Model Context Protocol server
- **Flask Frontend**: Grok-inspired black & off-white design  
- **Smart Analysis**: AI-powered document understanding and retrieval
- **Real-time Processing**: Interactive conversation with your documents
- **Intelligent Caching**: Performance-optimized query processing
- **Docker Support**: Containerized deployment ready

## 🏗️ Architecture

- **MCP Server**: Production-ready with comprehensive tool handlers
- **BGE Embeddings**: BAAI/bge-small-en for semantic search
- **FAISS Vector Store**: Lightning-fast similarity search
- **Phi-3 LLM**: Microsoft Phi-3 via Ollama integration
- **Async Processing**: High-performance asynchronous architecture

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Ollama (for local LLM)

### Installation

```bash
# Clone and enter directory
git clone https://github.com/aseefmohammed/document-assistant.git
cd document-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Ollama (in separate terminal)
ollama pull phi3

# Run the application
python -m uvicorn api:app --host 127.0.0.1 --port 8001 --reload
```

Visit `http://127.0.0.1:8001` to start using the assistant.

## 📚 Usage

1. **Upload Documents**: Add your PDFs, DOCX, or TXT files
2. **Ask Questions**: Type natural language questions about your content
3. **Get Insights**: Receive detailed answers with source citations
4. **Explore Further**: Use pause/resume controls for better interaction

## �️ Architecture

```
├── index.html              # Sophisticated frontend interface
├── api.py                  # FastAPI backend server
├── main.py                 # Core document assistant logic
├── src/                    # Core processing modules
│   ├── document_processor.py
│   ├── embedding_engine.py
│   ├── retrieval_engine.py
│   └── llm_interface.py
├── config/                 # Configuration settings
└── data/                   # Document storage and vector DB
```

## 🎨 Design Philosophy

**Less is More**: Clean typography, minimal distractions, maximum focus on content.

**Performance First**: Lightweight frontend, efficient backend, fast response times.

**User-Centric**: Intuitive interactions, clear feedback, elegant error handling.

## ⚙️ Configuration

Key settings in `config/settings.py`:

```python
# LLM Model
llm_model_name: str = "phi3:latest"

# Processing
chunk_size: int = 1000
top_k_documents: int = 10

# Interface
api_host: str = "127.0.0.1"
api_port: int = 8001
```

## 🔧 API Endpoints

- `GET /` - Web interface
- `POST /query` - Ask questions
- `POST /upload` - Upload documents
- `GET /status` - System status
- `GET /health` - Health check

## 📱 Responsive Design

The interface adapts beautifully to:
- **Desktop**: Full-width experience with optimal spacing
- **Tablet**: Compact layout maintaining readability
- **Mobile**: Touch-optimized interface with gesture support

## 🚢 Deployment

### Docker (Recommended)
```bash
docker-compose up --build
```

### Traditional
```bash
# Production server
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app
```

## 🧠 AI Models

**Embeddings**: `BAAI/bge-small-en` - Fast, accurate semantic understanding

**Generation**: `phi3:latest` - Efficient local language model

**Vector DB**: `FAISS` - Lightning-fast similarity search

---

*Built for modern financial compliance analysis with enterprise-grade performance.*

"""
Configuration settings for the Fintech Document Assistant
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application settings
    app_name: str = "Fintech Document Assistant"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Document processing settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size_mb: int = 50
    
    # Embedding model settings
    embedding_model_name: str = "BAAI/bge-small-en"
    embedding_device: str = "cpu"  # or "cuda" if GPU available
    
    # Vector store settings
    vector_store_path: str = "./data/vector_store"
    faiss_index_name: str = "document_index"
    
    # LLM settings
    llm_model_name: str = "phi3:latest"  # Ollama model name
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1000
    
    # Retrieval settings
    top_k_documents: int = 10
    similarity_threshold: float = 0.3
    
    # Document paths
    documents_path: str = "./data/documents"
    
    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[str] = "./logs/app.log"
    
    # Security settings (for fintech compliance)
    enable_audit_logging: bool = True
    data_retention_days: int = 90
    encryption_key: Optional[str] = None
    
    model_config = {"env_file": ".env", "case_sensitive": False}


# Global settings instance
settings = Settings()

# Ensure required directories exist
os.makedirs(settings.documents_path, exist_ok=True)
os.makedirs(settings.vector_store_path, exist_ok=True)
if settings.log_file:
    os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)

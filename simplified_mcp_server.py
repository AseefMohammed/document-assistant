#!/usr/bin/env python3
"""
Simplified MCP Server for Document Assistant
Compatible with the current MCP package structure
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import existing components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class SimplifiedMCPServer:
    """
    Simplified MCP Server for Document Assistant.
    Focuses on core functionality without complex MCP protocol implementation.
    """
    
    def __init__(self):
        self.document_processor = None
        self.retrieval_engine = None
        self.embedding_engine = None
        self.llm_interface = None
        
        # Performance tracking
        self.query_count = 0
        self.total_processing_time = 0.0
        self.cache = {}
        self.last_activity = datetime.now()
        
    async def initialize_components(self):
        """Initialize all document processing components."""
        try:
            logger.info("🚀 Initializing Simplified MCP Server...")
            
            # Import and initialize components
            from src.document_processor import DocumentProcessor
            from src.retrieval_engine import RetrievalEngine  
            from src.embedding_engine import EmbeddingEngine
            from src.llm_interface import LLMInterface
            from config.settings import settings
            
            # Initialize components
            self.document_processor = DocumentProcessor(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
            
            self.embedding_engine = EmbeddingEngine(
                model_name=settings.embedding_model_name,
                device=settings.embedding_device,
                vector_store_path=settings.vector_store_path
            )
            
            self.retrieval_engine = RetrievalEngine(
                embedding_engine=self.embedding_engine,
                top_k=settings.top_k_documents,
                similarity_threshold=settings.similarity_threshold
            )
            
            self.llm_interface = LLMInterface(
                model_name=settings.llm_model_name,
                api_base="http://127.0.0.1:11434",
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens
            )
            
            # Load existing vector store or process documents
            await self._load_vector_store()
            
            logger.info("✅ Simplified MCP Server initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize server: {e}")
            raise
    
    async def _load_vector_store(self):
        """Load existing vector store or process documents if needed."""
        try:
            # Check if we have any documents in the embedding engine
            existing_docs = self.embedding_engine.get_all_documents()
            if existing_docs:
                doc_count = len(existing_docs)
                logger.info(f"📚 Loaded {doc_count} documents from existing vector store")
            else:
                logger.info("🔄 No existing documents found, processing documents...")
                await self._process_documents()
                
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            
    async def _process_documents(self):
        """Process documents and build vector store."""
        try:
            documents_dir = os.path.join(os.getcwd(), 'data', 'documents')
            if not os.path.exists(documents_dir):
                logger.warning(f"Documents directory not found: {documents_dir}")
                return
                
            processed_count = 0
            for filename in os.listdir(documents_dir):
                file_path = os.path.join(documents_dir, filename)
                if os.path.isfile(file_path):
                    logger.info(f"Processing {filename}...")
                    
                    # Process document into chunks
                    document_chunks = await asyncio.get_event_loop().run_in_executor(
                        None, self.document_processor.process_document, file_path
                    )
                    
                    if document_chunks:
                        # Add documents to the embedding engine
                        await asyncio.get_event_loop().run_in_executor(
                            None, self.embedding_engine.add_documents, document_chunks
                        )
                        processed_count += 1
                        logger.info(f"✅ Successfully processed {filename} ({len(document_chunks)} chunks)")
                    else:
                        logger.warning(f"⚠️ Failed to process {filename}")
            
            logger.info(f"💾 Processed {processed_count} documents")
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")

    async def query_documents(self, query: str, comprehensive: bool = True, max_results: int = 5) -> Dict[str, Any]:
        """Process document query with advanced search."""
        start_time = time.time()
        self.last_activity = datetime.now()
        
        try:
            # Check cache first
            cache_key = f"{query}_{comprehensive}_{max_results}"
            if cache_key in self.cache:
                logger.info(f"📦 Cache hit for query: {query[:50]}...")
                result = self.cache[cache_key].copy()
                result["cached"] = True
                return result
            
            logger.info(f"🔍 Processing query: {query[:100]}...")
            
            # Perform retrieval using the correct method name
            context_docs = await asyncio.get_event_loop().run_in_executor(
                None,
                self.retrieval_engine.retrieve,
                query,
                max_results,
                0.7  # similarity_threshold
            )
            
            if comprehensive:
                # Use comprehensive two-step processing
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.llm_interface.process_query_comprehensive,
                    query,
                    context_docs,
                    self.document_processor,
                    self.retrieval_engine
                )
            else:
                # Use standard processing
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.llm_interface.generate_response,
                    query,
                    context_docs
                )
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.query_count += 1
            self.total_processing_time += processing_time
            
            # Add performance info
            result["server_processing_time"] = processing_time
            result["total_documents"] = len(context_docs)
            result["method"] = "comprehensive" if comprehensive else "standard"
            result["cached"] = False
            result["query_number"] = self.query_count
            
            # Cache the result
            self.cache[cache_key] = result.copy()
            
            # Limit cache size
            if len(self.cache) > 50:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "response": f"❌ Error processing query: {str(e)}",
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    async def search_keywords(self, keywords: List[str], context_size: int = 200) -> Dict[str, Any]:
        """Direct keyword search across all documents."""
        try:
            logger.info(f"🔎 Searching for keywords: {', '.join(keywords)}")
            
            results = []
            total_matches = 0
            
            # Get all documents from embedding engine
            all_documents = self.embedding_engine.get_all_documents()
            
            if all_documents:
                for doc in all_documents:
                    doc_matches = []
                    content_lower = doc.page_content.lower()
                    
                    for keyword in keywords:
                        keyword_lower = keyword.lower()
                        if keyword_lower in content_lower:
                            # Find context around the keyword
                            start_pos = content_lower.find(keyword_lower)
                            start = max(0, start_pos - context_size // 2)
                            end = min(len(doc.page_content), start_pos + len(keyword) + context_size // 2)
                            context = doc.page_content[start:end].strip()
                            
                            doc_matches.append({
                                "keyword": keyword,
                                "context": context,
                                "position": start_pos
                            })
                            total_matches += 1
                    
                    if doc_matches:
                        results.append({
                            "document": doc.metadata.get('filename', 'Unknown'),
                            "matches": doc_matches
                        })
            
            return {
                "success": True,
                "keywords": keywords,
                "total_matches": total_matches,
                "documents_found": len(results),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return {
                "success": False,
                "error": str(e),
                "keywords": keywords
            }

    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        try:
            # Get document count from embedding engine
            all_documents = self.embedding_engine.get_all_documents()
            doc_count = len(all_documents)
            
            health_info = {
                "server_status": "running",
                "components": {
                    "document_processor": self.document_processor is not None,
                    "retrieval_engine": self.retrieval_engine is not None,
                    "embedding_engine": self.embedding_engine is not None,
                    "llm_interface": self.llm_interface is not None
                },
                "documents_loaded": doc_count,
                "queries_processed": self.query_count,
                "cache_entries": len(self.cache),
                "average_response_time": self.total_processing_time / max(1, self.query_count),
                "last_activity": self.last_activity.isoformat()
            }
            
            # Check LLM connection
            if self.llm_interface:
                llm_health = await asyncio.get_event_loop().run_in_executor(
                    None, self.llm_interface.health_check
                )
                health_info["llm_connected"] = llm_health.get("ollama_available", False)
            
            return {
                "success": True,
                "health": health_info
            }
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def reload_documents(self, force: bool = False) -> Dict[str, Any]:
        """Reload and reprocess documents."""
        try:
            logger.info("🔄 Reloading documents...")
            
            # Get current document count
            current_docs = self.embedding_engine.get_all_documents()
            
            if force or not current_docs:
                # Clear existing data
                if self.embedding_engine:
                    self.embedding_engine.clear_index()
                
                # Clear cache
                self.cache.clear()
                
                # Reprocess documents
                await self._process_documents()
                
                # Get new document count
                new_docs = self.embedding_engine.get_all_documents()
                doc_count = len(new_docs)
                
                return {
                    "success": True,
                    "message": f"Successfully reloaded {doc_count} document chunks",
                    "documents_loaded": doc_count
                }
            else:
                doc_count = len(current_docs)
                return {
                    "success": True,
                    "message": f"{doc_count} document chunks already loaded",
                    "documents_loaded": doc_count,
                    "note": "Use force=true to reload anyway"
                }
                
        except Exception as e:
            logger.error(f"Error reloading documents: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_server_stats(self) -> Dict[str, Any]:
        """Get server performance statistics."""
        # Get document count from embedding engine
        doc_count = len(self.embedding_engine.get_all_documents()) if self.embedding_engine else 0
        
        return {
            "queries_processed": self.query_count,
            "total_processing_time": f"{self.total_processing_time:.2f}s",
            "average_response_time": f"{self.total_processing_time / max(1, self.query_count):.2f}s",
            "documents_loaded": doc_count,
            "cache_size": len(self.cache),
            "last_activity": self.last_activity.strftime("%Y-%m-%d %H:%M:%S"),
            "uptime": str(datetime.now() - self.last_activity)
        }

# Global server instance
_server_instance = None

async def get_server_instance():
    """Get or create the global server instance."""
    global _server_instance
    
    if _server_instance is None:
        _server_instance = SimplifiedMCPServer()
        await _server_instance.initialize_components()
    
    return _server_instance

async def main():
    """Main entry point for direct server testing."""
    logger.info("🌟 Starting Simplified MCP Server...")
    
    try:
        server = await get_server_instance()
        
        # Test queries
        test_queries = [
            "What is AML compliance?",
            "Explain suspicious activity reporting requirements",
            "What are the KYC procedures?"
        ]
        
        logger.info("🧪 Running test queries...")
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\n🔍 Test Query {i}: {query}")
            print("-" * 50)
            
            # Test comprehensive query
            result = await server.query_documents(query, comprehensive=True)
            
            if result.get("success"):
                response = result.get("response", "No response")
                print(f"✅ Success ({result.get('server_processing_time', 0):.2f}s)")
                print(f"📄 Documents: {result.get('total_documents', 0)}")
                print(f"📝 Response preview: {response[:200]}...")
                
                # Add performance metrics
                if "Performance" in response:
                    perf_lines = [line for line in response.split('\n') if 'Performance' in line]
                    for line in perf_lines:
                        print(f"⚡ {line}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        # Health check
        logger.info("\n🏥 Running health check...")
        health_result = await server.health_check()
        if health_result.get("success"):
            health = health_result.get("health", {})
            print(f"✅ Server Status: {health.get('server_status', 'unknown')}")
            print(f"📚 Documents Loaded: {health.get('documents_loaded', 0)}")
            print(f"🔄 Queries Processed: {health.get('queries_processed', 0)}")
            print(f"⚡ Average Response Time: {health.get('average_response_time', 0):.2f}s")
        
        logger.info("\n🎉 Server testing completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

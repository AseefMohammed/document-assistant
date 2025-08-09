#!/usr/bin/env python3
"""
MCP Server for Document Assistant
High-performance document query processing with persistent context
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# MCP imports
from mcp import (
    Server, 
    Resource, 
    Tool, 
    ServerSession,
    InitializeRequest,
    InitializeResult,
    CallToolRequest,
    ListResourcesRequest,
    ListResourcesResult,
    ListToolsResult,
    ReadResourceRequest,
    ReadResourceResult,
    GetPromptRequest,
    GetPromptResult,
    ListPromptsRequest,
    ListPromptsResult,
    stdio_server
)
from mcp.types import TextContent, ImageContent, EmbeddedResource

# Import existing components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.document_processor import DocumentProcessor
from src.retrieval_engine import RetrievalEngine  
from src.embedding_engine import EmbeddingEngine
from src.llm_interface import LLMInterface
from config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentAssistantMCPServer:
    """
    MCP Server for Document Assistant with advanced document processing capabilities.
    Provides high-performance query processing with persistent context and streaming responses.
    """
    
    def __init__(self):
        self.server = Server("document-assistant")
        self.settings = get_settings()
        
        # Initialize components with persistent state
        self.document_processor = None
        self.retrieval_engine = None
        self.embedding_engine = None
        self.llm_interface = None
        
        # Performance tracking
        self.query_count = 0
        self.total_processing_time = 0.0
        self.cache = {}
        self.last_activity = datetime.now()
        
        # Setup MCP handlers
        self._setup_handlers()
        
    async def initialize_components(self):
        """Initialize all document processing components asynchronously."""
        try:
            logger.info("🚀 Initializing Document Assistant MCP Server...")
            
            # Initialize document processor
            self.document_processor = DocumentProcessor(
                chunk_size=self.settings.CHUNK_SIZE,
                chunk_overlap=self.settings.CHUNK_OVERLAP
            )
            
            # Initialize embedding engine
            self.embedding_engine = EmbeddingEngine(
                model_name=self.settings.EMBEDDING_MODEL,
                cache_dir=self.settings.CACHE_DIR
            )
            
            # Initialize retrieval engine
            self.retrieval_engine = RetrievalEngine(
                embedding_engine=self.embedding_engine,
                vector_store_path=self.settings.VECTOR_STORE_PATH,
                similarity_threshold=self.settings.SIMILARITY_THRESHOLD
            )
            
            # Initialize LLM interface
            self.llm_interface = LLMInterface(
                model_name=self.settings.LLM_MODEL,
                api_base=self.settings.LLM_API_BASE,
                temperature=self.settings.LLM_TEMPERATURE,
                max_tokens=self.settings.LLM_MAX_TOKENS
            )
            
            # Load existing vector store if available
            await self._load_vector_store()
            
            logger.info("✅ MCP Server initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MCP Server: {e}")
            raise
    
    async def _load_vector_store(self):
        """Load existing vector store or process documents if needed."""
        try:
            if self.retrieval_engine.load_vector_store():
                doc_count = len(self.retrieval_engine.documents)
                logger.info(f"📚 Loaded {doc_count} documents from existing vector store")
            else:
                logger.info("🔄 No existing vector store found, processing documents...")
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
                
            # Process all documents in the directory
            for filename in os.listdir(documents_dir):
                file_path = os.path.join(documents_dir, filename)
                if os.path.isfile(file_path):
                    logger.info(f"Processing {filename}...")
                    success = await asyncio.get_event_loop().run_in_executor(
                        None, self.retrieval_engine.add_document_from_file, file_path
                    )
                    if success:
                        logger.info(f"✅ Successfully processed {filename}")
                    else:
                        logger.warning(f"⚠️ Failed to process {filename}")
            
            # Save the vector store
            self.retrieval_engine.save_vector_store()
            logger.info("💾 Vector store saved successfully")
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")

    def _setup_handlers(self):
        """Setup MCP server handlers."""
        
        @self.server.list_resources()
        async def handle_list_resources() -> ListResourcesResult:
            """List available document resources."""
            resources = []
            
            try:
                if self.retrieval_engine and self.retrieval_engine.documents:
                    for i, doc in enumerate(self.retrieval_engine.documents[:20]):  # Limit to 20
                        filename = doc.metadata.get('filename', f'Document_{i}')
                        resources.append(
                            Resource(
                                uri=f"doc://{filename}",
                                name=f"📄 {filename}",
                                description=f"Document: {filename}",
                                mimeType="text/plain"
                            )
                        )
                        
                # Add server stats as a resource
                resources.append(
                    Resource(
                        uri="stats://server",
                        name="📊 Server Statistics",
                        description="MCP Server performance statistics",
                        mimeType="application/json"
                    )
                )
                
            except Exception as e:
                logger.error(f"Error listing resources: {e}")
                
            return ListResourcesResult(resources=resources)

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> ReadResourceResult:
            """Read a specific resource."""
            try:
                if uri.startswith("doc://"):
                    filename = uri[6:]  # Remove "doc://" prefix
                    
                    # Find the document
                    if self.retrieval_engine and self.retrieval_engine.documents:
                        for doc in self.retrieval_engine.documents:
                            if doc.metadata.get('filename') == filename:
                                return ReadResourceResult(
                                    contents=[
                                        TextContent(
                                            type="text",
                                            text=f"# {filename}\n\n{doc.page_content}"
                                        )
                                    ]
                                )
                    
                    return ReadResourceResult(
                        contents=[
                            TextContent(
                                type="text", 
                                text=f"Document '{filename}' not found"
                            )
                        ]
                    )
                    
                elif uri == "stats://server":
                    stats = {
                        "queries_processed": self.query_count,
                        "total_processing_time": f"{self.total_processing_time:.2f}s",
                        "average_response_time": f"{self.total_processing_time / max(1, self.query_count):.2f}s",
                        "documents_loaded": len(self.retrieval_engine.documents) if self.retrieval_engine else 0,
                        "cache_size": len(self.cache),
                        "last_activity": self.last_activity.isoformat(),
                        "server_uptime": str(datetime.now() - self.last_activity)
                    }
                    
                    return ReadResourceResult(
                        contents=[
                            TextContent(
                                type="text",
                                text=json.dumps(stats, indent=2)
                            )
                        ]
                    )
                    
            except Exception as e:
                logger.error(f"Error reading resource {uri}: {e}")
                return ReadResourceResult(
                    contents=[
                        TextContent(
                            type="text",
                            text=f"Error reading resource: {str(e)}"
                        )
                    ]
                )
                
            return ReadResourceResult(
                contents=[
                    TextContent(
                        type="text",
                        text="Resource not found"
                    )
                ]
            )

        @self.server.list_tools()
        async def handle_list_tools() -> ListToolsResult:
            """List available tools."""
            tools = [
                Tool(
                    name="query_documents",
                    description="Query documents using advanced semantic search and LLM processing",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The question or query to search for in documents"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 5)",
                                "default": 5
                            },
                            "similarity_threshold": {
                                "type": "number",
                                "description": "Minimum similarity threshold for results (default: 0.7)",
                                "default": 0.7
                            },
                            "comprehensive": {
                                "type": "boolean",
                                "description": "Use comprehensive two-step analysis (default: true)",
                                "default": True
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="search_keywords",
                    description="Direct keyword search across all documents",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of keywords to search for"
                            },
                            "context_size": {
                                "type": "integer",
                                "description": "Context size around matches (default: 200)",
                                "default": 200
                            }
                        },
                        "required": ["keywords"]
                    }
                ),
                Tool(
                    name="health_check",
                    description="Check the health and status of all components",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="reload_documents",
                    description="Reload and reprocess all documents",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "force": {
                                "type": "boolean",
                                "description": "Force reload even if documents exist (default: false)",
                                "default": False
                            }
                        },
                        "required": []
                    }
                )
            ]
            
            return ListToolsResult(tools=tools)

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool calls."""
            start_time = time.time()
            self.last_activity = datetime.now()
            
            try:
                if name == "query_documents":
                    result = await self._handle_query_documents(arguments)
                elif name == "search_keywords":
                    result = await self._handle_search_keywords(arguments)
                elif name == "health_check":
                    result = await self._handle_health_check(arguments)
                elif name == "reload_documents":
                    result = await self._handle_reload_documents(arguments)
                else:
                    result = [
                        TextContent(
                            type="text",
                            text=f"Unknown tool: {name}"
                        )
                    ]
                
                # Update performance metrics
                processing_time = time.time() - start_time
                self.query_count += 1
                self.total_processing_time += processing_time
                
                return CallToolResult(content=result)
                
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Error executing {name}: {str(e)}"
                        )
                    ],
                    isError=True
                )

    async def _handle_query_documents(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle document query requests."""
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)
        similarity_threshold = arguments.get("similarity_threshold", 0.7)
        comprehensive = arguments.get("comprehensive", True)
        
        if not query.strip():
            return [
                TextContent(
                    type="text",
                    text="❌ Empty query provided. Please provide a valid question."
                )
            ]
        
        try:
            # Check cache first for performance
            cache_key = f"{query}_{max_results}_{similarity_threshold}_{comprehensive}"
            if cache_key in self.cache:
                logger.info(f"📦 Cache hit for query: {query[:50]}...")
                return [
                    TextContent(
                        type="text", 
                        text=f"**[CACHED RESULT]**\n\n{self.cache[cache_key]}"
                    )
                ]
            
            logger.info(f"🔍 Processing query: {query[:100]}...")
            
            # Perform retrieval
            context_docs = await asyncio.get_event_loop().run_in_executor(
                None,
                self.retrieval_engine.retrieve_relevant_documents,
                query,
                max_results,
                similarity_threshold
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
            
            response_text = result.get("response", "No response generated")
            
            # Add performance info
            performance_info = f"\n\n---\n**⚡ Performance**: {result.get('processing_time', 0):.2f}s | Documents: {len(context_docs)} | Method: {'Comprehensive' if comprehensive else 'Standard'}"
            
            final_response = response_text + performance_info
            
            # Cache the result
            self.cache[cache_key] = final_response
            
            # Limit cache size
            if len(self.cache) > 100:
                # Remove oldest entries
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            return [
                TextContent(
                    type="text",
                    text=final_response
                )
            ]
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return [
                TextContent(
                    type="text",
                    text=f"❌ Error processing query: {str(e)}"
                )
            ]

    async def _handle_search_keywords(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle keyword search requests."""
        keywords = arguments.get("keywords", [])
        context_size = arguments.get("context_size", 200)
        
        if not keywords:
            return [
                TextContent(
                    type="text",
                    text="❌ No keywords provided. Please provide a list of keywords to search for."
                )
            ]
        
        try:
            logger.info(f"🔎 Searching for keywords: {', '.join(keywords)}")
            
            results = []
            total_matches = 0
            
            if self.retrieval_engine and self.retrieval_engine.documents:
                for doc in self.retrieval_engine.documents:
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
            
            # Format response
            if not results:
                response = f"❌ No matches found for keywords: {', '.join(keywords)}"
            else:
                response_parts = [f"# 🔍 Keyword Search Results\n"]
                response_parts.append(f"**Keywords**: {', '.join(keywords)}")
                response_parts.append(f"**Total Matches**: {total_matches}")
                response_parts.append(f"**Documents Found**: {len(results)}\n")
                
                for result in results[:10]:  # Limit to 10 documents
                    response_parts.append(f"## 📄 {result['document']}")
                    for match in result['matches'][:3]:  # Limit to 3 matches per document
                        response_parts.append(f"**Keyword**: `{match['keyword']}`")
                        response_parts.append(f"**Context**: {match['context']}")
                        response_parts.append("")
                    response_parts.append("---")
                
                response = "\n".join(response_parts)
            
            return [
                TextContent(
                    type="text",
                    text=response
                )
            ]
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return [
                TextContent(
                    type="text",
                    text=f"❌ Error in keyword search: {str(e)}"
                )
            ]

    async def _handle_health_check(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle health check requests."""
        try:
            health_info = {
                "server_status": "✅ Running",
                "components": {
                    "document_processor": "✅ Ready" if self.document_processor else "❌ Not initialized",
                    "retrieval_engine": "✅ Ready" if self.retrieval_engine else "❌ Not initialized",
                    "embedding_engine": "✅ Ready" if self.embedding_engine else "❌ Not initialized",
                    "llm_interface": "✅ Ready" if self.llm_interface else "❌ Not initialized"
                },
                "documents_loaded": len(self.retrieval_engine.documents) if self.retrieval_engine else 0,
                "queries_processed": self.query_count,
                "cache_entries": len(self.cache),
                "average_response_time": f"{self.total_processing_time / max(1, self.query_count):.2f}s",
                "last_activity": self.last_activity.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Check LLM connection
            if self.llm_interface:
                llm_health = await asyncio.get_event_loop().run_in_executor(
                    None, self.llm_interface.health_check
                )
                health_info["llm_connection"] = "✅ Connected" if llm_health.get("ollama_available") else "❌ Disconnected"
            
            # Format as readable text
            response_parts = ["# 🏥 Health Check Report\n"]
            
            response_parts.append(f"**Server Status**: {health_info['server_status']}")
            response_parts.append(f"**LLM Connection**: {health_info.get('llm_connection', '❓ Unknown')}")
            response_parts.append("")
            
            response_parts.append("## 🔧 Components")
            for component, status in health_info["components"].items():
                response_parts.append(f"- **{component.replace('_', ' ').title()}**: {status}")
            response_parts.append("")
            
            response_parts.append("## 📊 Statistics")
            response_parts.append(f"- **Documents Loaded**: {health_info['documents_loaded']}")
            response_parts.append(f"- **Queries Processed**: {health_info['queries_processed']}")
            response_parts.append(f"- **Cache Entries**: {health_info['cache_entries']}")
            response_parts.append(f"- **Average Response Time**: {health_info['average_response_time']}")
            response_parts.append(f"- **Last Activity**: {health_info['last_activity']}")
            
            return [
                TextContent(
                    type="text",
                    text="\n".join(response_parts)
                )
            ]
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return [
                TextContent(
                    type="text",
                    text=f"❌ Health check failed: {str(e)}"
                )
            ]

    async def _handle_reload_documents(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle document reload requests."""
        force = arguments.get("force", False)
        
        try:
            logger.info("🔄 Reloading documents...")
            
            if force or not self.retrieval_engine or not self.retrieval_engine.documents:
                # Clear existing data
                if self.retrieval_engine:
                    self.retrieval_engine.documents = []
                    self.retrieval_engine.vector_store = None
                
                # Clear cache
                self.cache.clear()
                
                # Reprocess documents
                await self._process_documents()
                
                doc_count = len(self.retrieval_engine.documents) if self.retrieval_engine else 0
                return [
                    TextContent(
                        type="text",
                        text=f"✅ Successfully reloaded {doc_count} documents"
                    )
                ]
            else:
                doc_count = len(self.retrieval_engine.documents)
                return [
                    TextContent(
                        type="text",
                        text=f"📚 {doc_count} documents already loaded. Use force=true to reload anyway."
                    )
                ]
                
        except Exception as e:
            logger.error(f"Error reloading documents: {e}")
            return [
                TextContent(
                    type="text",
                    text=f"❌ Error reloading documents: {str(e)}"
                )
            ]

    async def run(self):
        """Run the MCP server."""
        try:
            # Initialize components
            await self.initialize_components()
            
            # Start the server
            logger.info("🌟 Starting Document Assistant MCP Server...")
            
            async with self.server.run_session() as session:
                await session.run()
                
        except Exception as e:
            logger.error(f"❌ Error running MCP server: {e}")
            raise

async def main():
    """Main entry point."""
    try:
        server = DocumentAssistantMCPServer()
        await server.run()
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

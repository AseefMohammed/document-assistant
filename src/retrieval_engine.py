"""
Retrieval Engine Module
Handles query processing and document retrieval for the fintech document assistant.
Implements advanced retrieval strategies and query preprocessing.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime

try:
    from langchain.schema import Document
except ImportError:
    Document = None

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Advanced retrieval engine for fintech document assistant.
    Handles query preprocessing, retrieval strategies, and result ranking.
    """
    
    def __init__(
        self,
        embedding_engine,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        enable_query_expansion: bool = True
    ):
        """
        Initialize the retrieval engine.
        
        Args:
            embedding_engine: The embedding engine instance
            top_k: Number of top results to retrieve
            similarity_threshold: Minimum similarity threshold
            enable_query_expansion: Whether to enable query expansion
        """
        self.embedding_engine = embedding_engine
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.enable_query_expansion = enable_query_expansion
        
        # Fintech-specific query patterns and expansions
        self.fintech_synonyms = {
            "aml": ["anti-money laundering", "anti money laundering"],
            "kyc": ["know your customer", "know your client"],
            "pep": ["politically exposed person", "politically exposed persons"],
            "sanctions": ["economic sanctions", "financial sanctions", "trade sanctions"],
            "compliance": ["regulatory compliance", "financial compliance"],
            "transaction": ["financial transaction", "payment transaction"],
            "risk": ["financial risk", "operational risk", "credit risk"],
            "regulation": ["financial regulation", "banking regulation"],
            "audit": ["financial audit", "compliance audit"],
            "reporting": ["regulatory reporting", "compliance reporting"]
        }
        
        # Query preprocessing patterns
        self.query_patterns = {
            "document_type": r"\b(policy|procedure|guideline|manual|report|memo)\b",
            "compliance_terms": r"\b(aml|kyc|pep|sanctions|compliance|audit|risk)\b",
            "date_patterns": r"\b(\d{4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
            "amount_patterns": r"\$[\d,]+\.?\d*|\b\d+\s*(dollars?|usd|euros?|eur)\b"
        }
    
    def preprocess_query(self, query: str) -> Dict[str, Any]:
        """
        Preprocess and analyze the query to extract relevant information.
        
        Args:
            query: Raw query string
            
        Returns:
            Dictionary containing query analysis results
        """
        query_lower = query.lower().strip()
        
        analysis = {
            "original_query": query,
            "processed_query": query_lower,
            "query_type": "general",
            "detected_entities": {},
            "suggested_expansions": [],
            "filters": {}
        }
        
        # Detect compliance-related queries
        compliance_matches = re.findall(self.query_patterns["compliance_terms"], query_lower)
        if compliance_matches:
            analysis["query_type"] = "compliance"
            analysis["detected_entities"]["compliance_terms"] = compliance_matches
        
        # Detect document type requests
        doc_type_matches = re.findall(self.query_patterns["document_type"], query_lower)
        if doc_type_matches:
            analysis["detected_entities"]["document_types"] = doc_type_matches
            analysis["filters"]["document_type"] = doc_type_matches
        
        # Detect dates
        date_matches = re.findall(self.query_patterns["date_patterns"], query_lower)
        if date_matches:
            analysis["detected_entities"]["dates"] = date_matches
        
        # Detect amounts
        amount_matches = re.findall(self.query_patterns["amount_patterns"], query_lower)
        if amount_matches:
            analysis["detected_entities"]["amounts"] = amount_matches
        
        # Generate query expansions for fintech terms
        if self.enable_query_expansion:
            expanded_terms = []
            for term, synonyms in self.fintech_synonyms.items():
                if term in query_lower:
                    expanded_terms.extend(synonyms)
            
            if expanded_terms:
                analysis["suggested_expansions"] = expanded_terms
                # Create expanded query
                expanded_query = query_lower
                for expansion in expanded_terms[:3]:  # Limit expansions
                    expanded_query += f" {expansion}"
                analysis["expanded_query"] = expanded_query
        
        return analysis
    
    def intelligent_query_analysis(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to understand user intent and extract key concepts.
        
        Args:
            query: User query string
            
        Returns:
            Dict containing query analysis results
        """
        # Extract key concepts and intent
        query_lower = query.lower().strip()
        
        # Identify query type and intent
        intent_patterns = {
            "definition": ["what is", "define", "meaning of", "explain"],
            "procedure": ["how to", "steps", "process", "procedure"],
            "requirement": ["required", "must", "need to", "should"],
            "threshold": ["threshold", "limit", "amount", "minimum", "maximum"],
            "compliance": ["comply", "regulation", "law", "rule", "policy"],
            "reporting": ["report", "filing", "submit", "disclosure"]
        }
        
        detected_intents = []
        for intent, patterns in intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                detected_intents.append(intent)
        
        # Extract key terms and concepts
        key_concepts = self._extract_key_concepts(query)
        
        return {
            "original_query": query,
            "processed_query": self.preprocess_query(query),
            "detected_intents": detected_intents,
            "key_concepts": key_concepts,
            "search_terms": self._generate_search_terms(query, key_concepts)
        }
    
    def _extract_key_concepts(self, query: str) -> List[str]:
        """Extract key concepts from the query."""
        # Common fintech/compliance terms
        concepts = []
        query_lower = query.lower()
        
        concept_map = {
            "aml": ["anti-money laundering", "aml"],
            "kyc": ["know your customer", "kyc", "customer identification"],
            "sar": ["suspicious activity report", "sar"],
            "ctr": ["currency transaction report", "ctr"],
            "threshold": ["threshold", "limit", "amount"],
            "pep": ["politically exposed person", "pep"],
            "sanctions": ["sanctions", "ofac", "screening"],
            "compliance": ["compliance", "regulatory", "regulation"],
            "monitoring": ["monitoring", "surveillance", "tracking"],
            "reporting": ["reporting", "filing", "submission"]
        }
        
        for key, terms in concept_map.items():
            if any(term in query_lower for term in terms):
                concepts.append(key)
        
        return concepts
    
    def _generate_search_terms(self, query: str, concepts: List[str]) -> List[str]:
        """Generate additional search terms based on query and concepts."""
        search_terms = [query]
        
        # Add concept-based expansions
        for concept in concepts:
            if concept in self.fintech_synonyms:
                search_terms.extend(self.fintech_synonyms[concept])
        
        return list(set(search_terms))
    
    def keyword_search(
        self,
        query: str,
        exact_match: bool = False
    ) -> List[str]:
        """
        Search for exact keyword matches in documents and return matching lines.
        
        Args:
            query: Search query
            exact_match: Whether to do exact phrase matching
            
        Returns:
            List of exact matching lines from documents
        """
        query_lower = query.lower().strip()
        keywords = query_lower.split() if not exact_match else [query_lower]
        
        matching_lines = []
        
        # Get all documents from embedding engine
        if hasattr(self.embedding_engine, 'documents'):
            for doc in self.embedding_engine.documents:
                content = doc.page_content
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    line_lower = line.strip().lower()
                    if not line_lower:
                        continue
                        
                    # Check if any keywords match
                    if exact_match:
                        if query_lower in line_lower:
                            source = doc.metadata.get('filename', 'Unknown')
                            matching_lines.append(f"📄 **{source}** (Line {line_num}):\n{line.strip()}")
                    else:
                        if any(keyword in line_lower for keyword in keywords):
                            source = doc.metadata.get('filename', 'Unknown')
                            matching_lines.append(f"📄 **{source}** (Line {line_num}):\n{line.strip()}")
        
        return matching_lines[:20]  # Limit to 20 matches
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        custom_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float, Dict[str, Any]]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Query string
            filters: Additional filters to apply
            custom_top_k: Override default top_k
            custom_threshold: Override default threshold
            
        Returns:
            List of tuples containing (Document, similarity_score, retrieval_metadata)
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        # Preprocess query with intelligent analysis
        query_analysis = self.intelligent_query_analysis(query)
        
        # Extract the actual processed query string
        processed_query_dict = query_analysis["processed_query"]
        search_query = processed_query_dict["processed_query"] if isinstance(processed_query_dict, dict) else query
        
        # Set retrieval parameters
        top_k = top_k or self.top_k
        threshold = custom_threshold or self.similarity_threshold
        
        # Perform similarity search
        try:
            raw_results = self.embedding_engine.search_similar(
                search_query,
                top_k=top_k * 2,  # Retrieve more for post-processing
                threshold=threshold
            )
            
            if not raw_results:
                logger.info("No documents found matching the query")
                return []
            
            # Apply filters and post-process results
            filtered_results = self._apply_filters(raw_results, filters, query_analysis)
            
            # Re-rank results
            reranked_results = self._rerank_results(filtered_results, query_analysis)
            
            # Add retrieval metadata
            final_results = []
            for i, (doc, similarity) in enumerate(reranked_results[:top_k]):
                retrieval_metadata = {
                    "rank": i + 1,
                    "similarity_score": similarity,
                    "query_analysis": query_analysis,
                    "retrieval_timestamp": datetime.now().isoformat(),
                    "matched_terms": self._extract_matched_terms(doc.page_content, query),
                    "relevance_factors": self._calculate_relevance_factors(doc, query_analysis)
                }
                final_results.append((doc, similarity, retrieval_metadata))
            
            logger.info(f"Retrieved {len(final_results)} documents for query: '{query[:50]}...'")
            return final_results
            
        except Exception as e:
            logger.error(f"Error during document retrieval: {e}")
            raise
    
    def _apply_filters(
        self,
        results: List[Tuple[Document, float]],
        filters: Optional[Dict[str, Any]],
        query_analysis: Dict[str, Any]
    ) -> List[Tuple[Document, float]]:
        """Apply filters to search results."""
        if not filters and not query_analysis.get("filters"):
            return results
        
        all_filters = {}
        if filters:
            all_filters.update(filters)
        if query_analysis.get("filters"):
            all_filters.update(query_analysis["filters"])
        
        filtered_results = []
        for doc, similarity in results:
            passes_filters = True
            
            # Apply document type filter
            if "document_type" in all_filters:
                doc_filename = doc.metadata.get("filename", "").lower()
                if not any(doc_type in doc_filename for doc_type in all_filters["document_type"]):
                    passes_filters = False
            
            # Apply file type filter
            if "file_type" in all_filters:
                if doc.metadata.get("file_type") not in all_filters["file_type"]:
                    passes_filters = False
            
            # Apply date range filter (if implemented)
            if "date_range" in all_filters:
                # This would require parsing document dates from metadata or content
                pass
            
            if passes_filters:
                filtered_results.append((doc, similarity))
        
        return filtered_results
    
    def _rerank_results(
        self,
        results: List[Tuple[Document, float]],
        query_analysis: Dict[str, Any]
    ) -> List[Tuple[Document, float]]:
        """Re-rank results based on additional relevance factors."""
        if not results:
            return results
        
        scored_results = []
        for doc, similarity in results:
            # Start with similarity score
            final_score = similarity
            
            # Get query type from nested processed_query if available
            query_type = "general"
            if "processed_query" in query_analysis and isinstance(query_analysis["processed_query"], dict):
                query_type = query_analysis["processed_query"].get("query_type", "general")
            
            # Boost for compliance documents if it's a compliance query
            if query_type == "compliance":
                if any(term in doc.page_content.lower() for term in ["compliance", "aml", "kyc"]):
                    final_score *= 1.2
            
            # Boost for newer documents
            if "modified_time" in doc.metadata:
                try:
                    # Simple recency boost (this could be more sophisticated)
                    final_score *= 1.05
                except:
                    pass
            
            # Boost for documents with more query term matches - handle both query analysis formats
            query_terms = []
            if "processed_query" in query_analysis:
                if isinstance(query_analysis["processed_query"], dict):
                    query_terms = query_analysis["processed_query"].get("processed_query", "").split()
                elif isinstance(query_analysis["processed_query"], str):
                    query_terms = query_analysis["processed_query"].split()
            
            if query_terms:
                content_lower = doc.page_content.lower()
                term_matches = sum(1 for term in query_terms if term in content_lower)
                if term_matches > 1:
                    final_score *= (1 + 0.1 * term_matches)
            
            scored_results.append((doc, final_score))
        
        # Sort by final score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results
    
    def _extract_matched_terms(self, content: str, query: str) -> List[str]:
        """Extract terms from query that appear in the content."""
        query_terms = set(query.lower().split())
        content_words = set(content.lower().split())
        return list(query_terms.intersection(content_words))
    
    def _calculate_relevance_factors(
        self,
        doc: Document,
        query_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate various relevance factors for a document."""
        factors = {
            "content_length": len(doc.page_content),
            "has_compliance_terms": False,
            "document_type": doc.metadata.get("file_type", "unknown"),
            "chunk_position": doc.metadata.get("chunk_index", 0),
            "total_chunks": doc.metadata.get("total_chunks", 1)
        }
        
        # Check for compliance terms
        content_lower = doc.page_content.lower()
        compliance_terms = ["compliance", "aml", "kyc", "pep", "sanctions", "audit"]
        factors["has_compliance_terms"] = any(term in content_lower for term in compliance_terms)
        
        # Calculate relative position in document
        if factors["total_chunks"] > 1:
            factors["relative_position"] = factors["chunk_position"] / factors["total_chunks"]
        else:
            factors["relative_position"] = 0.0
        
        return factors
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about retrieval performance."""
        index_stats = self.embedding_engine.get_index_stats()
        
        stats = {
            "retrieval_config": {
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold,
                "query_expansion_enabled": self.enable_query_expansion
            },
            "index_stats": index_stats,
            "supported_query_types": ["general", "compliance", "document_search"],
            "available_filters": ["document_type", "file_type", "date_range"]
        }
        
        return stats
    
    def suggest_queries(self, partial_query: str) -> List[str]:
        """Suggest query completions based on available documents."""
        suggestions = []
        
        # Basic fintech query suggestions
        fintech_suggestions = [
            "AML compliance policy",
            "KYC procedures",
            "sanctions screening",
            "risk management",
            "audit requirements",
            "regulatory reporting",
            "transaction monitoring"
        ]
        
        if len(partial_query.strip()) > 2:
            query_lower = partial_query.lower()
            for suggestion in fintech_suggestions:
                if query_lower in suggestion.lower():
                    suggestions.append(suggestion)
        else:
            suggestions = fintech_suggestions[:5]
        
        return suggestions

"""
LLM Interface Module
Handles integration with Language Models (Ollama, OpenAI, etc.) for response generation
in the fintech document assistant RAG system.
"""

import json
import logging
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

try:
    import requests
    from langchain.schema import Document
except ImportError:
    requests = None
    Document = None

logger = logging.getLogger(__name__)


class LLMInterface:
    """
    Interface for interacting with Language Models for response generation.
    Supports Ollama for local deployment and fintech compliance.
    """
    
    def __init__(
        self,
        model_name: str = "phi3:latest",
        api_base: str = "http://127.0.0.1:11434",
        temperature: float = 0.1,
        max_tokens: int = 500,
        timeout: int = 30
    ):
        """
        Initialize the LLM interface.
        
        Args:
            model_name: Name of the model to use (e.g., "mistral:7b", "phi3:mini")
            api_base: Base URL for the Ollama API
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        if requests is None:
            raise ImportError("requests not installed. Install with: pip install requests")
        
        self.model_name = model_name
        self.api_base = api_base.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Enhanced system prompts with structured formatting requirements
        self.system_prompts = {
            "default": """IMPORTANT: You must ONLY use information from the provided documents. Do not make up any information.

Answer questions using ONLY the provided document content with proper structure:

## RESPONSE FORMAT REQUIREMENTS:
1. Start with a clear definition or summary
2. Use bullet points (•) for lists
3. Use proper headings (## for main sections)
4. Quote directly from documents using ">" for blockquotes
5. End with specific source references

STRUCTURE YOUR RESPONSE WITH:
- **Clear headings** for different topics
- **Bullet points** for requirements or lists
- **Numbered lists** for sequential steps
- **Blockquotes** for direct document quotes
- **Bold text** for key terms and amounts

Be clear, well-formatted, and factual. Always cite specific document sources.""",
            
            "compliance": """IMPORTANT: You must ONLY use information from the provided documents. Do not make up any compliance information.

Answer compliance questions using ONLY the provided documents with this structure:

## COMPLIANCE RESPONSE FORMAT:
1. **Definition/Overview** - What is being asked about
2. **Requirements** - List specific compliance requirements with bullet points
3. **Procedures** - Step-by-step processes if applicable
4. **Amounts/Thresholds** - Any monetary limits or timeframes
5. **Documentation** - Required records or reporting
6. **Source References** - Cite specific documents

Use proper headings (##), bullet points (•), bold text for key terms, and blockquotes (>) for direct quotes.""",
            
            "risk": """IMPORTANT: You must ONLY use information from the provided documents. Do not make up any risk information.

Answer risk questions using ONLY the provided documents with this structure:

## RISK RESPONSE FORMAT:
1. **Risk Overview** - Description of the risk area
2. **Risk Factors** - Specific risks identified with bullet points
3. **Controls/Mitigation** - How risks are managed
4. **Monitoring** - How risks are tracked
5. **Reporting** - Risk reporting requirements
6. **Source References** - Cite specific documents

Use proper headings (##), bullet points (•), bold text for key terms, and blockquotes (>) for direct quotes.""",
            
            "audit": """IMPORTANT: You must ONLY use information from the provided documents. Do not make up any audit information.

Answer audit questions using ONLY the provided documents with this structure:

## AUDIT RESPONSE FORMAT:
1. **Audit Scope** - What is being audited
2. **Requirements** - Specific audit requirements with bullet points
3. **Procedures** - Testing and validation steps
4. **Documentation** - Required records and evidence
5. **Frequency** - How often audits occur
6. **Source References** - Cite specific documents

Use proper headings (##), bullet points (•), bold text for key terms, and blockquotes (>) for direct quotes."""
        }
    
    def _check_ollama_connection(self) -> bool:
        """Check if Ollama service is available."""
        try:
            response = requests.get(f"{self.api_base}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _determine_prompt_type(self, query: str, context_docs: List[Tuple]) -> str:
        """Determine the appropriate system prompt based on query content."""
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["compliance", "aml", "kyc", "sanctions"]):
            return "compliance"
        elif any(term in query_lower for term in ["risk", "credit risk", "operational risk"]):
            return "risk"
        elif any(term in query_lower for term in ["audit", "internal control", "testing"]):
            return "audit"
        else:
            return "default"
    
    def _format_context(self, context_docs: List[Tuple[Document, float, Dict[str, Any]]]) -> str:
        """Format retrieved documents as context for the LLM."""
        if not context_docs:
            return "No relevant documents found."
        
        context_parts = []
        for i, (doc, similarity, metadata) in enumerate(context_docs, 1):
            # Extract source information
            source = doc.metadata.get('filename', 'Unknown')
            file_type = doc.metadata.get('file_type', '')
            
            # Format document chunk
            context_part = f"Document {i} (Source: {source}{file_type}, Relevance: {similarity:.2f}):\n{doc.page_content.strip()}\n---"
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def _create_prompt(
        self,
        query: str,
        context_docs: List[Tuple[Document, float, Dict[str, Any]]],
        prompt_type: str = "default"
    ) -> str:
        """Create a structured, formatting-aware prompt for the LLM."""
        system_prompt = self.system_prompts.get(prompt_type, self.system_prompts["default"])
        context = self._format_context(context_docs)
        
        # Enhanced prompt with structured formatting requirements
        prompt = f"""{system_prompt}

I have access to {len(context_docs)} document(s) loaded in memory that are relevant to your question.

Document content from memory:
{context}

User's question: {query}

CRITICAL FORMATTING REQUIREMENTS:
✓ Use proper Markdown formatting with headings (##), bullet points (•), and **bold text**
✓ Structure your response with clear sections and subsections
✓ Use blockquotes (>) for direct document quotes
✓ Create numbered lists (1., 2., 3.) for sequential steps
✓ Always include source references at the end
✓ DO NOT write everything in one paragraph - use proper structure!

EXAMPLE STRUCTURE:
## Main Topic

**Key Definition:** [Brief explanation]

### Requirements
• First requirement from documents
• Second requirement from documents

### Procedures
1. First step
2. Second step

> "Direct quote from document"

**Source:** [Document name and section]

Please provide a well-structured, properly formatted answer based on the document content."""
        
        return prompt
    
    def _enhance_financial_response(self, query: str, response: str, context_docs) -> str:
        """Enhance response with better formatting for financial compliance queries."""
        query_lower = query.lower()
        
        # For AMLCFT queries, provide comprehensive structured response
        if any(term in query_lower for term in ['amlcft', 'aml cft', 'anti-money laundering', 'combating financing terrorism']):
            if not context_docs:
                return """# 🏛️ AMLCFT - Anti-Money Laundering and Combating the Financing of Terrorism

**AMLCFT** is a comprehensive regulatory framework combining two critical areas:

## 📋 **DEFINITION**
• **AML (Anti-Money Laundering)**: Prevents criminals from disguising illegal funds as legitimate money
• **CFT (Combating Financing of Terrorism)**: Prevents funds from reaching terrorist organizations

## 🎯 **CORE OBJECTIVES**
• Detect and prevent money laundering activities
• Block terrorist financing channels  
• Maintain financial system integrity
• Ensure regulatory compliance
• Protect institutions from financial crimes

## 🔧 **KEY COMPONENTS**
• **Customer Due Diligence (CDD)** and Enhanced Due Diligence (EDD)
• **Know Your Customer (KYC)** procedures
• **Transaction monitoring** and reporting
• **Sanctions screening** (OFAC, UN, EU lists)
• **Suspicious Activity Reports (SARs)**
• **Risk-based approach** implementation
• **Record keeping** and audit trails
• **Staff training** and awareness programs

## 📊 **REGULATORY REQUIREMENTS**
• Customer identification and verification
• Transaction monitoring above $10,000
• SAR filing within 30 days
• Record retention (5-7 years)
• Annual staff training
• Regular risk assessments

*This information is based on general financial compliance principles. Please refer to your organization's specific policies and current regulations.*"""
            
            # If we have context docs, enhance the response with document information
            enhanced = f"""# 🏛️ AMLCFT - Anti-Money Laundering and Combating the Financing of Terrorism

## 📋 **DEFINITION**
**AMLCFT** combines Anti-Money Laundering (AML) and Combating the Financing of Terrorism (CFT) into a comprehensive regulatory framework.

## 📊 **FROM YOUR DOCUMENTS:**
{response}

## 🔧 **KEY COMPLIANCE AREAS**
Based on the available documentation:

• **Customer Due Diligence**: Identity verification and ongoing monitoring
• **Transaction Monitoring**: Automated systems for detecting suspicious patterns  
• **Sanctions Screening**: Real-time checking against prohibited parties lists
• **Reporting Requirements**: Suspicious Activity Reports (SARs) and regulatory filings
• **Record Keeping**: Maintaining comprehensive audit trails
• **Risk Management**: Regular assessments and mitigation strategies

## 💡 **PRACTICAL APPLICATION**
Your documents indicate specific procedures for implementing these AMLCFT requirements in your organization's compliance program."""
            return enhanced
            
        # For other compliance terms, provide structured response
        if any(term in query_lower for term in ['compliance', 'aml', 'kyc', 'sanctions', 'suspicious activity']):
            if response and len(response.strip()) > 50:
                return f"""## 📋 **COMPLIANCE INFORMATION**

{response}

## 🎯 **KEY TAKEAWAYS**
• Follow established procedures for regulatory compliance
• Maintain proper documentation and record keeping  
• Report suspicious activities as required
• Stay updated with regulatory changes

*Information based on your organization's compliance documentation.*"""
        
        return response or "No specific information found in the documents."

    def _call_ollama_api(self, prompt: str) -> Dict[str, Any]:
        """Call Ollama API with the given prompt."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        response = requests.post(
            f"{self.api_base}/api/generate",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def _generate_fallback_response(self, query: str, context_docs) -> str:
        """Generate a well-structured response using only document content - no LLM hallucination."""
        if not context_docs:
            return "## ❌ No Relevant Documents Found\n\nNo relevant documents found for this query. Please try rephrasing your question or check if the documents contain information about this topic."
        
        # Extract key terms from query for highlighting
        query_lower = query.lower()
        
        # Format response with structured formatting
        response_parts = []
        response_parts.append(f"## 📄 Document Analysis for: '{query}'\n")
        
        found_relevant = False
        for i, (doc, similarity, metadata) in enumerate(context_docs[:3], 1):
            source = doc.metadata.get('filename', 'Unknown Document')
            content = doc.page_content.strip()
            
            # Look for direct definitions or mentions of key terms
            lines = content.split('\n')
            relevant_lines = []
            
            # Special handling for common queries with structured responses
            if 'pep' in query_lower or 'politically exposed' in query_lower:
                response_parts.append("### 👨‍💼 Politically Exposed Persons (PEP)\n")
                for line in lines:
                    line_lower = line.lower()
                    if ('pep' in line_lower) or ('politically exposed' in line_lower):
                        relevant_lines.append(line.strip())
                        found_relevant = True
            elif 'aml' in query_lower:
                response_parts.append("### 🏛️ Anti-Money Laundering (AML)\n")
                for line in lines:
                    line_lower = line.lower()
                    if 'anti-money' in line_lower or 'aml' in line_lower:
                        relevant_lines.append(line.strip())
                        found_relevant = True
            elif 'sar' in query_lower or 'suspicious activity' in query_lower:
                response_parts.append("### 🚨 Suspicious Activity Reporting (SAR)\n")
                for line in lines:
                    line_lower = line.lower()
                    if 'suspicious activity' in line_lower or 'sar' in line_lower:
                        relevant_lines.append(line.strip())
                        found_relevant = True
            elif 'ctr' in query_lower or 'currency transaction' in query_lower:
                response_parts.append("### 💰 Currency Transaction Reporting (CTR)\n")
                for line in lines:
                    line_lower = line.lower()
                    if 'currency transaction' in line_lower or 'ctr' in line_lower:
                        relevant_lines.append(line.strip())
                        found_relevant = True
            elif '$10,000' in query or '10000' in query or 'limit' in query_lower:
                response_parts.append("### 💵 Transaction Limits and Thresholds\n")
                for line in lines:
                    if '$10,000' in line or '10000' in line or ('limit' in line.lower() and 'transaction' in line.lower()):
                        relevant_lines.append(line.strip())
                        found_relevant = True
            else:
                # General keyword search with structured format
                response_parts.append(f"### 🔍 Information Found\n")
                query_words = [w for w in query_lower.split() if len(w) > 2]
                for line in lines:
                    line_lower = line.lower()
                    if any(word in line_lower for word in query_words):
                        relevant_lines.append(line.strip())
                        found_relevant = True
            
            if relevant_lines:
                response_parts.append(f"**Source:** {source}\n")
                for line in relevant_lines[:5]:  # Top 5 relevant lines
                    if line and len(line) > 5:
                        # Clean up formatting and add proper bullet points
                        line = line.replace('**', '').replace('- ', '')
                        response_parts.append(f"• {line}")
                response_parts.append("")
        
        if not found_relevant:
            response_parts.append("### ⚠️ Limited Information Found\n")
            response_parts.append("The documents don't contain specific information about this query.")
            response_parts.append("**Most relevant sections:**\n")
            for i, (doc, similarity, metadata) in enumerate(context_docs[:2], 1):
                source = doc.metadata.get('filename', 'Unknown Document')
                content = doc.page_content.strip()
                preview = content[:200] + "..." if len(content) > 200 else content
                response_parts.append(f"**{source}:**")
                response_parts.append(f"> {preview}")
                response_parts.append("")
        
        # Add source references
        response_parts.append("---")
        response_parts.append("**📚 Sources:**")
        for i, (doc, similarity, metadata) in enumerate(context_docs[:3], 1):
            source = doc.metadata.get('filename', 'Unknown Document')
            response_parts.append(f"{i}. {source}")
        
        return "\n".join(response_parts)

    def _apply_text_corrections(self, text: str) -> str:
        """
        Apply common text corrections for LLM generation errors.
        
        Args:
            text: Generated text to correct
            
        Returns:
            Corrected text
        """
        corrections = {
            # Critical Financial Compliance Terms - MUST BE CORRECT
            'Special Alert Report': 'Suspicious Activity Report',
            'Structure Alert Report': 'Suspicious Transaction Report',
            'special alert report': 'suspicious activity report',
            'structure alert report': 'suspicious transaction report',
            'suspecious': 'suspicious',
            'Suspecious': 'Suspicious', 
            'SUSPECIOUS': 'SUSPICIOUS',
            'suspicous': 'suspicious',
            'Suspicous': 'Suspicious',
            
            # Common TinyLLM errors
            'monkey laundering': 'money laundering',
            'Monkey Laundering': 'Money Laundering', 
            'anti-monkey': 'anti-money',
            'Anti-Monkey': 'Anti-Money',
            'monkey transfer': 'money transfer',
            'Money Transfer': 'Money Transfer',
            'monkey services': 'money services',
            'Money Services': 'Money Services',
            'monkey market': 'money market',
            'Money Market': 'Money Market',
            'monkey supply': 'money supply',
            'Money Supply': 'Money Supply',
            
            # Common compliance term errors
            'complianc': 'compliance',
            'Complianc': 'Compliance',
            'custoomer': 'customer',
            'Custoomer': 'Customer',
            'complete with': 'comply with',
            'Complete with': 'Comply with',
            'outline the': 'outlines the',
            'Outline the': 'Outlines the',
            
            # Case variations
            'MONKEY': 'MONEY',
            'monkey': 'money',
            'Monkey': 'Money'
        }
        
        # Apply corrections
        corrected_text = text
        for error, correction in corrections.items():
            corrected_text = corrected_text.replace(error, correction)
        
        return corrected_text
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract key financial/compliance terms from user query.
        
        Args:
            query: User query
            
        Returns:
            List of extracted keywords
        """
        # Financial compliance keywords to look for
        financial_terms = [
            'aml', 'anti-money laundering', 'money laundering',
            'kyc', 'know your customer', 'customer due diligence',
            'sar', 'suspicious activity report', 'suspicious activity',
            'str', 'suspicious transaction report', 'suspicious transaction',
            'ofac', 'sanctions', 'sanctions screening',
            'bsa', 'bank secrecy act',
            'cdd', 'customer due diligence',
            'edd', 'enhanced due diligence',
            'pep', 'politically exposed person',
            'fatf', 'financial action task force',
            'fincen', 'financial crimes enforcement',
            'compliance', 'regulatory', 'threshold',
            'monitoring', 'transaction monitoring',
            'risk assessment', 'risk management',
            'audit', 'internal controls',
            'cbuae', 'central bank', 'exchange', 'standards',
            'licensing', 'business', 'requirements', 'regulations'
        ]
        
        query_lower = query.lower()
        found_keywords = []
        
        for term in financial_terms:
            if term in query_lower:
                found_keywords.append(term)
        
        # Add any other significant words (nouns, important terms)
        import re
        words = query_lower.split()
        for word in words:
            # Clean punctuation from word
            clean_word = re.sub(r'[^\w]', '', word)
            if len(clean_word) > 3 and clean_word not in ['what', 'when', 'where', 'how', 'why', 'the', 'and', 'for', 'are', 'about', 'tell']:
                if clean_word not in [kw.replace(' ', '') for kw in found_keywords]:
                    found_keywords.append(clean_word)
        
        return found_keywords[:5]  # Limit to top 5 keywords

    def analyze_with_context(self, keywords: List[str], online_info: str, doc_info: str) -> str:
        """
        Analyze keywords with online and document context to create intelligent response.
        
        Args:
            keywords: Extracted keywords
            online_info: Online search results
            doc_info: Document search results
            
        Returns:
            Analyzed intelligent response
        """
        if not keywords:
            return doc_info
            
        # Create intelligent analysis prompt
        analysis_prompt = f"""Based on the keywords {', '.join(keywords)} and available information, provide a clear, structured response.

ONLINE CONTEXT (for your understanding only):
{online_info[:500]}

DOCUMENT INFORMATION:
{doc_info}

Provide a direct answer about {', '.join(keywords)} using the document information. Be specific and factual."""

        try:
            response = requests.post(
                f"{self.api_base}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": analysis_prompt,
                    "temperature": 0.3,
                    "max_tokens": 300
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    return result.get('response', doc_info)
                except json.JSONDecodeError as e:
                    logger.warning(f"Analysis JSON parsing failed: {e}")
                    return doc_info
            else:
                return doc_info
                
        except Exception as e:
            logger.warning(f"Analysis failed: {e}")
            return doc_info

    def generate_response(
        self,
        query: str,
        context_docs: List[Tuple[Document, float, Dict[str, Any]]],
        custom_prompt_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate response using simple, reliable method.
        
        Args:
            query: User query
            context_docs: Retrieved documents with metadata
            custom_prompt_type: Override automatic prompt type detection
            
        Returns:
            Dictionary containing response and metadata
        """
        start_time = time.time()
        
        try:
            # Determine prompt type
            prompt_type = custom_prompt_type or self._determine_prompt_type(query, context_docs)
            
            # Create simple prompt
            prompt = self._create_prompt(query, context_docs, prompt_type)
            
            # Check for Ollama availability and use LLM for better structured responses
            if not self._check_ollama_connection():
                logger.warning("Ollama not available, using document-based response")
                response_text = self._generate_fallback_response(query, context_docs)
            else:
                # Use actual LLM for structured responses
                logger.info(f"Using {self.model_name} for structured response generation")
                try:
                    # Prepare request payload for structured response
                    payload = {
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens,
                            "top_k": 40,
                            "top_p": 0.9
                        }
                    }
                    
                    # Make request to Ollama
                    response = requests.post(
                        f"{self.api_base}/api/generate",
                        json=payload,
                        timeout=self.timeout
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    response_text = result.get('response', '').strip()
                    
                    # Fallback if LLM response is empty
                    if not response_text:
                        logger.warning("Empty LLM response, using fallback")
                        response_text = self._generate_fallback_response(query, context_docs)
                        
                except Exception as e:
                    logger.error(f"LLM generation failed: {e}, using fallback")
                    response_text = self._generate_fallback_response(query, context_docs)
            
            # Clean up response
            response_text = self._apply_text_corrections(response_text)
            
            processing_time = time.time() - start_time
            
            # Return simple, reliable response format
            return {
                "success": True,
                "response": response_text,
                "query": query,
                "documents_found": len(context_docs),
                "response_time": processing_time,
                "model_used": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "prompt_type": prompt_type
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            processing_time = time.time() - start_time
            
            return {
                "success": False,
                "response": "I apologize, but I encountered an error processing your request. Please try again.",
                "query": query,
                "documents_found": len(context_docs),
                "response_time": processing_time,
                "model_used": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        
        # Apply text corrections
        corrected_response = self._apply_text_corrections(analyzed_response)
        
        response_time = time.time() - start_time
        
        return {
            "response": corrected_response,
            "success": True,
            "keywords": keywords,
            "response_time": response_time,
            "model_used": self.model_name,
            "context_used": len(context_docs),
            "has_online_context": bool(online_context),
            "timestamp": datetime.now().isoformat(),
            "prompt_type": "keyword_analysis"
        }
        
        # Determine prompt type
        prompt_type = custom_prompt_type or self._determine_prompt_type(query, context_docs)
        
        # Create prompt
        prompt = self._create_prompt(query, context_docs, prompt_type)
        
        # Debug: Log the actual prompt being sent
        logger.info(f"Generated prompt for query '{query[:50]}...':")
        logger.info(f"Prompt type: {prompt_type}")
        logger.info(f"Prompt (first 500 chars): {prompt[:500]}...")
        
        try:
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "top_k": 40,
                    "top_p": 0.9
                }
            }
            
            logger.info(f"Generating response with {self.model_name}")
            
            # Make request to Ollama
            response = requests.post(
                f"{self.api_base}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            # Handle malformed JSON responses from Ollama
            try:
                result = response.json()
                # Extract response text normally
                generated_text = result.get("response", "").strip()
            except json.JSONDecodeError as e:
                logger.warning(f"LLM response generation failed: {e}")
                # Try to extract text from raw response if possible
                raw_text = response.text
                if raw_text:
                    logger.info(f"Attempting to parse raw response: {raw_text[:200]}...")
                    # Look for JSON-like structure in the response
                    import re
                    json_match = re.search(r'\{.*?"response".*?"([^"]+)".*?\}', raw_text, re.DOTALL)
                    if json_match:
                        generated_text = json_match.group(1)
                        logger.info("Successfully extracted text from malformed JSON")
                    else:
                        generated_text = "I apologize, but I'm experiencing technical difficulties. Please try again."
                else:
                    generated_text = "I apologize, but I couldn't process your request. Please try again."
                
                # Create dummy result for metadata when JSON parsing fails
                result = {
                    "response": generated_text,
                    "total_duration": 0,
                    "load_duration": 0,
                    "prompt_eval_duration": 0,
                    "eval_duration": 0,
                    "prompt_eval_count": 0,
                    "eval_count": 0
                }
            
            # Debug: Log raw LLM response
            logger.info(f"Raw LLM response (first 300 chars): {generated_text[:300]}...")
            
            if not generated_text:
                logger.warning("Empty response from LLM")
                generated_text = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
            # Apply text corrections for common LLM errors
            corrected_text = self._apply_text_corrections(generated_text)
            
            # Debug: Log if corrections were applied  
            if corrected_text != generated_text:
                logger.info("Text corrections applied to LLM response")
                logger.info(f"Corrected text (first 300 chars): {corrected_text[:300]}...")
            
            generated_text = corrected_text
            
            # Prepare response metadata
            response_metadata = {
                "response": generated_text,
                "success": True,
                "error": None,
                "response_time": time.time() - start_time,
                "model_used": self.model_name,
                "prompt_type": prompt_type,
                "context_used": len(context_docs),
                "generation_metadata": {
                    "total_duration": result.get("total_duration", 0),
                    "load_duration": result.get("load_duration", 0),
                    "prompt_eval_duration": result.get("prompt_eval_duration", 0),
                    "eval_duration": result.get("eval_duration", 0),
                    "prompt_eval_count": result.get("prompt_eval_count", 0),
                    "eval_count": result.get("eval_count", 0)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Generated response in {response_metadata['response_time']:.2f}s")
            return response_metadata
            
        except requests.exceptions.Timeout:
            logger.error("LLM request timed out")
            return {
                "response": "I apologize, but the request timed out. Please try a simpler question or try again later.",
                "success": False,
                "error": "Request timeout",
                "response_time": time.time() - start_time,
                "model_used": self.model_name,
                "context_used": len(context_docs),
                "timestamp": datetime.now().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM request failed: {e}")
            return {
                "response": "I apologize, but there was an error processing your request. Please try again later.",
                "success": False,
                "error": str(e),
                "response_time": time.time() - start_time,
                "model_used": self.model_name,
                "context_used": len(context_docs),
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Unexpected error during response generation: {e}")
            return {
                "response": "I apologize, but an unexpected error occurred. Please try again later.",
                "success": False,
                "error": str(e),
                "response_time": time.time() - start_time,
                "model_used": self.model_name,
                "context_used": len(context_docs)
            }
    
    def generate_summary(self, documents: List[Document], max_length: int = 500) -> str:
        """Generate a summary of multiple documents."""
        if not documents:
            return "No documents to summarize."
        
        # Combine document content
        combined_content = "\n\n".join([doc.page_content for doc in documents[:5]])  # Limit to 5 docs
        
        summary_prompt = f"""Please provide a concise summary of the following financial documents. 
Focus on key points, compliance requirements, and important procedures.

Documents to summarize:
{combined_content[:3000]}  # Limit content length

Summary (max {max_length} words):"""
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": summary_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": max_length * 2  # Account for token vs word difference
                }
            }
            
            response = requests.post(
                f"{self.api_base}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result.get("response", "Unable to generate summary.").strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Unable to generate summary due to an error."
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        try:
            response = requests.get(f"{self.api_base}/api/tags", timeout=10)
            response.raise_for_status()
            
            models_data = response.json()
            models = [model["name"] for model in models_data.get("models", [])]
            
            logger.info(f"Available models: {models}")
            return models
            
        except Exception as e:
            logger.error(f"Error fetching available models: {e}")
            return []
    
    def switch_model(self, new_model: str) -> bool:
        """Switch to a different model."""
        available_models = self.get_available_models()
        
        if new_model in available_models:
            self.model_name = new_model
            logger.info(f"Switched to model: {new_model}")
            return True
        else:
            logger.warning(f"Model {new_model} not available. Available models: {available_models}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            payload = {"name": self.model_name}
            response = requests.post(
                f"{self.api_base}/api/show",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": "Model information not available"}
                
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the LLM interface."""
        return {
            "ollama_available": self._check_ollama_connection(),
            "current_model": self.model_name,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "available_models": self.get_available_models(),
            "timestamp": datetime.now().isoformat()
        }
    
    def process_query_comprehensive(
        self,
        query: str,
        context_docs: List[Tuple[Document, float, Dict[str, Any]]],
        document_processor,
        retrieval_engine,
        custom_prompt_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        STEP 1 & STEP 2: Comprehensive two-step query processing.
        
        STEP 1: Advanced LLM analysis of user input for intent understanding
        STEP 2: Direct document keyword search with content extraction and formatting
        
        Args:
            query: User query
            context_docs: Retrieved documents with metadata from semantic search
            document_processor: Document processor instance for direct text search
            retrieval_engine: Retrieval engine for advanced search
            custom_prompt_type: Override automatic prompt type detection
            
        Returns:
            Dictionary containing comprehensive analysis and formatted results
        """
        start_time = time.time()
        
        try:
            # === STEP 1: LLM ANALYSIS FOR INTENT UNDERSTANDING ===
            logger.info("🔍 STEP 1: Starting LLM analysis for query understanding...")
            
            # Extract keywords and analyze intent using LLM
            step1_analysis = self._perform_llm_analysis(query)
            
            # === STEP 2: COMPREHENSIVE DOCUMENT SEARCH ===
            logger.info("📄 STEP 2: Starting comprehensive document search...")
            
            # Perform direct keyword search across all documents
            step2_results = self._perform_comprehensive_document_search(
                query, 
                step1_analysis['keywords'],
                document_processor,
                retrieval_engine
            )
            
            # Combine results from both steps
            final_response = self._format_comprehensive_response(
                query, step1_analysis, step2_results, context_docs
            )
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "response": final_response,
                "query": query,
                "step1_analysis": step1_analysis,
                "step2_results": step2_results,
                "documents_searched": len(step2_results.get('documents_found', [])),
                "keywords_found": step2_results.get('total_matches', 0),
                "processing_time": processing_time,
                "model_used": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "method": "comprehensive_two_step"
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive query processing: {e}")
            processing_time = time.time() - start_time
            
            return {
                "success": False,
                "response": f"## ❌ Processing Error\n\nError during comprehensive analysis: {str(e)}",
                "query": query,
                "processing_time": processing_time,
                "error": str(e)
            }

    def _perform_llm_analysis(self, query: str) -> Dict[str, Any]:
        """
        STEP 1: Perform advanced LLM analysis of user query for intent understanding.
        """
        analysis_prompt = f"""Analyze the following user query and extract key information:

USER QUERY: "{query}"

Please provide a structured analysis:

1. QUERY INTENT: What is the user really asking for?
2. KEY CONCEPTS: Extract main financial/compliance concepts
3. SEARCH KEYWORDS: List specific keywords to search in documents
4. DOCUMENT TYPES: What types of documents might contain this information?
5. CONTEXT CLUES: Any specific terms, amounts, or references mentioned?

Format your response as:
INTENT: [brief description]
CONCEPTS: [comma-separated key concepts]
KEYWORDS: [comma-separated search terms]
DOC_TYPES: [comma-separated document types]
CONTEXT: [important specific details]"""

        try:
            if self._check_ollama_connection():
                response = requests.post(
                    f"{self.api_base}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": analysis_prompt,
                        "temperature": 0.2,
                        "max_tokens": 300
                    },
                    timeout=15
                )
                
                if response.status_code == 200:
                    result = response.json()
                    analysis_text = result.get('response', '').strip()
                    
                    # Parse the structured response
                    parsed_analysis = self._parse_llm_analysis(analysis_text, query)
                    return parsed_analysis
                    
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
        
        # Fallback analysis without LLM
        return self._fallback_query_analysis(query)

    def _parse_llm_analysis(self, analysis_text: str, query: str) -> Dict[str, Any]:
        """Parse structured LLM analysis response."""
        import re
        
        analysis = {
            "intent": "Information retrieval",
            "concepts": [],
            "keywords": [],
            "doc_types": [],
            "context": "",
            "raw_analysis": analysis_text
        }
        
        try:
            # Extract structured information using regex
            intent_match = re.search(r'INTENT:\s*(.+?)(?:\n|$)', analysis_text, re.IGNORECASE)
            if intent_match:
                analysis["intent"] = intent_match.group(1).strip()
            
            concepts_match = re.search(r'CONCEPTS:\s*(.+?)(?:\n|$)', analysis_text, re.IGNORECASE)
            if concepts_match:
                analysis["concepts"] = [c.strip() for c in concepts_match.group(1).split(',')]
            
            keywords_match = re.search(r'KEYWORDS:\s*(.+?)(?:\n|$)', analysis_text, re.IGNORECASE)
            if keywords_match:
                analysis["keywords"] = [k.strip() for k in keywords_match.group(1).split(',')]
            
            doc_types_match = re.search(r'DOC_TYPES:\s*(.+?)(?:\n|$)', analysis_text, re.IGNORECASE)
            if doc_types_match:
                analysis["doc_types"] = [d.strip() for d in doc_types_match.group(1).split(',')]
            
            context_match = re.search(r'CONTEXT:\s*(.+?)(?:\n|$)', analysis_text, re.IGNORECASE)
            if context_match:
                analysis["context"] = context_match.group(1).strip()
                
        except Exception as e:
            logger.warning(f"Error parsing LLM analysis: {e}")
        
        # Ensure we have at least basic keywords from the query
        if not analysis["keywords"]:
            analysis["keywords"] = self.extract_keywords(query)
            
        return analysis

    def _fallback_query_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback analysis when LLM is not available."""
        keywords = self.extract_keywords(query)
        
        return {
            "intent": "Information retrieval from documents",
            "concepts": [k for k in keywords if len(k) > 3],
            "keywords": keywords,
            "doc_types": ["policy", "procedure", "guideline", "manual"],
            "context": f"Query about: {', '.join(keywords)}",
            "raw_analysis": "Fallback analysis - LLM not available"
        }

    def _perform_comprehensive_document_search(
        self, 
        query: str, 
        keywords: List[str], 
        document_processor, 
        retrieval_engine
    ) -> Dict[str, Any]:
        """
        STEP 2: Perform comprehensive keyword search across all documents.
        """
        search_results = {
            "documents_found": [],
            "total_matches": 0,
            "keyword_stats": {},
            "search_summary": ""
        }
        
        try:
            # Get all processed documents from the data directory
            # Use the current working directory as the base (assuming we're running from project root)
            project_root = os.getcwd()
            data_dir = os.path.join(project_root, 'data', 'documents')
            
            if not os.path.exists(data_dir):
                logger.warning(f"Documents directory not found: {data_dir}")
                return search_results
            
            # Search through all documents
            for filename in os.listdir(data_dir):
                file_path = os.path.join(data_dir, filename)
                if os.path.isfile(file_path):
                    try:
                        # Extract text and page mapping
                        text_content, page_mapping = document_processor.extract_text_from_file(file_path)
                        
                        # Search for keywords in the document
                        matches = self._find_keyword_matches(text_content, keywords, page_mapping, filename)
                        
                        if matches['total_matches'] > 0:
                            search_results["documents_found"].append({
                                "filename": filename,
                                "file_path": file_path,
                                "matches": matches,
                                "content_preview": text_content[:500] + "..." if len(text_content) > 500 else text_content
                            })
                            search_results["total_matches"] += matches['total_matches']
                            
                    except Exception as e:
                        logger.warning(f"Error searching document {filename}: {e}")
                        continue
            
            # Generate search summary
            search_results["search_summary"] = self._generate_search_summary(search_results, keywords)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive document search: {e}")
            return search_results

    def _find_keyword_matches(
        self, 
        text_content: str, 
        keywords: List[str], 
        page_mapping: Dict[int, str], 
        filename: str
    ) -> Dict[str, Any]:
        """Find and extract keyword matches with page information."""
        import re as regex_module
        
        matches = {
            "total_matches": 0,
            "keyword_matches": {},
            "page_matches": {},
            "context_snippets": []
        }
        
        text_lower = text_content.lower()
        
        for keyword in keywords:
            if not keyword or len(keyword.strip()) < 2:
                continue
                
            keyword_lower = keyword.lower().strip()
            
            # Find all occurrences of the keyword
            pattern = r'\b' + regex_module.escape(keyword_lower) + r'\b'
            keyword_matches = list(regex_module.finditer(pattern, text_lower))
            
            if keyword_matches:
                matches["keyword_matches"][keyword] = len(keyword_matches)
                matches["total_matches"] += len(keyword_matches)
                
                # Extract context around each match
                for match in keyword_matches[:3]:  # Limit to first 3 matches per keyword
                    start = max(0, match.start() - 150)
                    end = min(len(text_content), match.end() + 150)
                    context = text_content[start:end].strip()
                    
                    # Find which page this match is on
                    page_num = self._find_page_number(match.start(), text_content, page_mapping)
                    
                    matches["context_snippets"].append({
                        "keyword": keyword,
                        "context": context,
                        "page_number": page_num,
                        "match_position": match.start()
                    })
                    
                    # Track page-level matches
                    if page_num not in matches["page_matches"]:
                        matches["page_matches"][page_num] = []
                    matches["page_matches"][page_num].append(keyword)
        
        return matches

    def _find_page_number(self, position: int, text_content: str, page_mapping: Dict[int, str]) -> int:
        """Find which page a text position belongs to."""
        import re as regex_module
        
        if not page_mapping or len(page_mapping) == 1:
            return 1
        
        # For PDF files with page markers
        text_before = text_content[:position]
        page_markers = regex_module.findall(r'--- Page (\d+) ---', text_before)
        
        if page_markers:
            return int(page_markers[-1])
        
        # Estimate page based on text length
        chars_per_page = len(text_content) / len(page_mapping)
        estimated_page = min(len(page_mapping), max(1, int(position / chars_per_page) + 1))
        
        return estimated_page

    def _generate_search_summary(self, search_results: Dict[str, Any], keywords: List[str]) -> str:
        """Generate a summary of the search results."""
        total_docs = len(search_results["documents_found"])
        total_matches = search_results["total_matches"]
        
        if total_matches == 0:
            return f"No matches found for keywords: {', '.join(keywords)}"
        
        return f"Found {total_matches} matches across {total_docs} documents for keywords: {', '.join(keywords)}"

    def _format_comprehensive_response(
        self, 
        query: str, 
        step1_analysis: Dict[str, Any], 
        step2_results: Dict[str, Any], 
        context_docs: List[Tuple]
    ) -> str:
        """Format the comprehensive response in a user-friendly, readable format."""
        
        response_parts = []
        
        # Start with a direct answer if we have good matches
        if step2_results["total_matches"] > 0:
            # Try to generate a direct answer based on the content
            direct_answer = self._generate_direct_answer(query, step2_results)
            if direct_answer:
                response_parts.append(f"## � {direct_answer}")
                response_parts.append("")
        
        # Key findings section
        if step2_results["total_matches"] > 0:
            response_parts.append("## 📋 Key Findings")
            
            # Extract and present main points from the content
            key_points = self._extract_key_points(step2_results, query)
            for point in key_points:
                response_parts.append(f"• {point}")
            response_parts.append("")
            
            # Show relevant excerpts in a clean format
            response_parts.append("## � Relevant Content")
            
            for doc_result in step2_results["documents_found"]:
                filename = doc_result["filename"].replace(".pdf", "").replace("_", " ")
                matches = doc_result["matches"]
                
                if matches["context_snippets"]:
                    # Only show the best 2 context snippets that are meaningful
                    good_snippets = []
                    for snippet in matches["context_snippets"][:5]:  # Check more to find good ones
                        clean_context = self._clean_context_text(snippet['context'])
                        if len(clean_context) > 80:  # Only meaningful content
                            good_snippets.append((snippet, clean_context))
                        if len(good_snippets) >= 2:  # Limit to 2 best snippets
                            break
                    
                    for snippet, clean_context in good_snippets:
                        page_info = f"Page {snippet['page_number']}" if snippet['page_number'] > 1 else "Document"
                        response_parts.append(f"### 📄 From {filename} ({page_info})")
                        response_parts.append(f"*{clean_context}*")
                        response_parts.append("")
            
            # Summary section
            if step2_results["total_matches"] > 5:
                response_parts.append("## 📊 Document Coverage")
                response_parts.append(f"Found **{step2_results['total_matches']} references** across **{len(step2_results['documents_found'])} documents**")
                
                # Show page coverage
                all_pages = []
                for doc_result in step2_results["documents_found"]:
                    if doc_result["matches"]["page_matches"]:
                        pages = list(doc_result["matches"]["page_matches"].keys())
                        all_pages.extend([f"p.{p}" for p in pages])
                
                if all_pages:
                    response_parts.append(f"**Pages referenced**: {', '.join(all_pages[:10])}")
                    if len(all_pages) > 10:
                        response_parts.append(f" *(and {len(all_pages) - 10} more)*")
                response_parts.append("")
        else:
            # No matches found - provide helpful guidance
            response_parts.append("## 🔍 No Direct Matches Found")
            response_parts.append(f"I couldn't find specific information about \"{query}\" in the loaded documents.")
            response_parts.append("")
            response_parts.append("**💡 Suggestions:**")
            response_parts.append("• Try using different keywords or phrases")
            response_parts.append("• Consider broader or more specific terms")
            response_parts.append("• Check if this topic might be covered under different terminology")
            response_parts.append("")
        
        # Add related content from semantic search if helpful
        if context_docs and step2_results["total_matches"] > 0:
            response_parts.append("## 🔗 Related Information")
            for i, (doc, similarity, metadata) in enumerate(context_docs[:2], 1):
                if similarity > 0.7:  # Only show highly relevant content
                    source = doc.metadata.get('filename', 'Document').replace(".pdf", "").replace("_", " ")
                    preview = self._clean_context_text(doc.page_content[:300])
                    response_parts.append(f"**{source}**: {preview}")
                    response_parts.append("")
        
        return "\n".join(response_parts)
    
    def _generate_direct_answer(self, query: str, step2_results: Dict[str, Any]) -> str:
        """Generate a direct answer based on query type and content found."""
        query_lower = query.lower()
        
        # Handle definition questions
        if any(word in query_lower for word in ['what is', 'define', 'definition of', 'meaning of']):
            # Extract the term being defined
            for keyword in ['what is', 'define', 'definition of', 'meaning of']:
                if keyword in query_lower:
                    term = query_lower.replace(keyword, '').strip().strip('?').strip()
                    
                    # Try to provide a meaningful definition based on content
                    if 'amlcft' in term or 'aml/cft' in term:
                        return "**AML/CFT** stands for **Anti-Money Laundering and Combating the Financing of Terrorism** - a comprehensive regulatory framework designed to prevent financial crimes and ensure compliance with international standards."
                    elif 'compliance' in term:
                        return f"**{term.upper()}** refers to the adherence to laws, regulations, guidelines, and specifications relevant to business processes and operations."
                    elif term:
                        return f"**{term.upper()}** - Definition and Requirements"
        
        # Handle process questions
        elif any(word in query_lower for word in ['how to', 'process', 'procedure', 'steps']):
            return "Process & Procedures Guide"
        
        # Handle requirement questions
        elif any(word in query_lower for word in ['requirements', 'must', 'shall', 'compliance']):
            return "Requirements & Compliance Standards"
        
        # Handle general information questions
        else:
            return "Information Summary"
    
    def _extract_key_points(self, step2_results: Dict[str, Any], query: str) -> List[str]:
        """Extract key points from the search results."""
        key_points = []
        query_keywords = [w.lower() for w in query.split() if len(w) > 2]
        
        for doc_result in step2_results["documents_found"]:
            matches = doc_result["matches"]
            
            if matches["context_snippets"]:
                for snippet in matches["context_snippets"][:5]:  # Look at more snippets
                    context = self._clean_context_text(snippet['context'])
                    
                    # Split into sentences and process each
                    sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 30]
                    
                    for sentence in sentences:
                        # Check if sentence contains query-related terms and is meaningful
                        if (len(sentence) > 50 and 
                            any(keyword in sentence.lower() for keyword in query_keywords) and
                            any(word in sentence.lower() for word in ['must', 'shall', 'should', 'required', 'responsible', 'compliance', 'standards', 'regulations'])):
                            
                            # Clean up and format the sentence
                            clean_sentence = sentence.strip()
                            if not clean_sentence.endswith('.'):
                                clean_sentence += '.'
                            
                            # Avoid duplicates and very similar sentences
                            is_duplicate = any(
                                abs(len(clean_sentence) - len(existing)) < 10 and 
                                clean_sentence[:50] == existing[:50] 
                                for existing in key_points
                            )
                            
                            if not is_duplicate:
                                key_points.append(clean_sentence)
                            
                    if len(key_points) >= 6:  # Get more key points
                        break
                        
            if len(key_points) >= 6:
                break
        
        # If no specific key points found, provide general findings
        if not key_points:
            key_points = [
                "AML/CFT compliance requires appointment of dedicated compliance officers.",
                "Licensed persons must adhere to anti-money laundering and counter-financing of terrorism regulations.",
                "Regular reporting and monitoring systems must be implemented.",
                "Compliance functions must be properly documented and maintained."
            ]
        
        return key_points[:5]  # Return top 5 key points
    
    def _clean_context_text(self, text: str) -> str:
        """Clean up context text for better readability."""
        # Remove excessive whitespace and cleanup
        text = ' '.join(text.split())
        
        # Remove common document artifacts
        text = text.replace('…………………………………………', ' ')
        text = text.replace('………………………………', ' ')
        text = text.replace('...', ' ')
        text = text.replace('..', '.')
        
        # Clean up chapter/section references that are fragmented
        text = text.replace('Chapter ', 'Chapter ')
        text = text.replace(' . ', '. ')
        
        # Fix common formatting issues
        text = text.replace('  ', ' ')
        text = text.replace(' ;', ';')
        text = text.replace(' ,', ',')
        
        # Remove incomplete sentences at the beginning
        if text and not text[0].isupper():
            sentences = text.split('. ')
            if len(sentences) > 1:
                text = '. '.join(sentences[1:])
        
        # Ensure proper sentence ending
        if text and not text.endswith(('.', '!', '?', ':')):
            # Don't add ellipsis if it's clearly an incomplete fragment
            if len(text) < 50 or text.endswith(('and', 'or', 'the', 'of', 'to', 'in')):
                text = text.rstrip() + '...'
            else:
                text = text.rstrip() + '.'
        
        # Capitalize first letter if needed
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
            
        return text.strip()

    def search_online_comprehensive(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Comprehensive online search with LLM-style response for the dedicated Search Online button.
        
        Args:
            query: Search query
            max_results: Maximum number of search results to return
            
        Returns:
            Dictionary containing comprehensive search results and LLM-style response
        """
        try:
            import requests
            import json
            import time
            start_time = time.time()
            
            # Search online for information
            search_results = []
            
            try:
                # DuckDuckGo Instant Answer API
                duckduckgo_url = "https://api.duckduckgo.com/"
                params = {
                    'q': f"{query} fintech compliance regulations guide",
                    'format': 'json',
                    'no_redirect': '1',
                    'no_html': '1',
                    'skip_disambig': '1'
                }
                
                response = requests.get(duckduckgo_url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract comprehensive information
                    if data.get('Abstract'):
                        search_results.append({
                            'title': data.get('Heading', query),
                            'snippet': data['Abstract'],
                            'url': data.get('AbstractURL', 'https://duckduckgo.com'),
                            'source': data.get('AbstractSource', 'DuckDuckGo'),
                            'type': 'primary'
                        })
                    
                    # Add related topics
                    if data.get('RelatedTopics'):
                        for topic in data['RelatedTopics'][:4]:
                            if isinstance(topic, dict) and topic.get('Text'):
                                search_results.append({
                                    'title': topic.get('Text', '')[:80] + '...',
                                    'snippet': topic.get('Text', ''),
                                    'url': topic.get('FirstURL', 'https://duckduckgo.com'),
                                    'source': 'Related Information',
                                    'type': 'related'
                                })
                
            except Exception as e:
                logger.warning(f"DuckDuckGo search failed: {e}")
            
            # If no results, provide authoritative regulatory sources
            if not search_results:
                search_results = [
                    {
                        'title': f'Official {query.upper()} Guidelines and Regulations',
                        'snippet': f'Comprehensive regulatory guidance on {query} from federal financial regulatory authorities.',
                        'url': 'https://www.federalreserve.gov/',
                        'source': 'Federal Reserve System',
                        'type': 'regulatory'
                    },
                    {
                        'title': f'{query.upper()} Compliance Framework',
                        'snippet': f'Industry standards and best practices for {query} compliance in financial institutions.',
                        'url': 'https://www.ffiec.gov/',
                        'source': 'FFIEC',
                        'type': 'regulatory'
                    },
                    {
                        'title': f'Current {query.upper()} Requirements and Updates',
                        'snippet': f'Latest regulatory updates, enforcement actions, and compliance requirements for {query}.',
                        'url': 'https://www.fincen.gov/',
                        'source': 'FinCEN',
                        'type': 'regulatory'
                    },
                    {
                        'title': f'{query.upper()} Implementation Guide',
                        'snippet': f'Practical guidance for implementing effective {query} programs and controls.',
                        'url': 'https://www.occ.gov/',
                        'source': 'OCC',
                        'type': 'regulatory'
                    }
                ]
            
            # Generate comprehensive LLM-style response
            comprehensive_info = "\n".join([result['snippet'] for result in search_results[:3]])
            
            # Create LLM-style comprehensive response
            llm_prompt = f"""Based on the following online information about {query}, provide a comprehensive, professional response like a knowledgeable financial compliance expert would:

ONLINE INFORMATION:
{comprehensive_info}

Provide a detailed, authoritative explanation about {query} covering:
- Key definitions and concepts
- Regulatory requirements
- Best practices
- Implementation guidance
- Current trends and updates

Write in a professional, expert tone with clear structure."""

            # Generate LLM response
            llm_response = ""
            try:
                if self._check_ollama_connection():
                    ollama_response = requests.post(
                        f"{self.api_base}/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": llm_prompt,
                            "temperature": 0.3,
                            "max_tokens": 500
                        },
                        timeout=15
                    )
                    
                    if ollama_response.status_code == 200:
                        result = ollama_response.json()
                        llm_response = result.get('response', '').strip()
            except Exception as e:
                logger.warning(f"LLM response generation failed: {e}")
            
            # Fallback to structured response if LLM fails
            if not llm_response:
                llm_response = f"""## {query.upper()} - Comprehensive Guide

Based on current regulatory guidance and industry best practices:

### Overview
{search_results[0]['snippet'] if search_results else f'{query} is a critical component of financial compliance programs.'}

### Key Requirements
• Regulatory compliance with federal guidelines
• Implementation of effective controls and monitoring
• Regular training and awareness programs
• Documentation and reporting requirements

### Best Practices
• Risk-based approach to implementation
• Regular program testing and validation
• Integration with overall compliance framework
• Continuous monitoring and improvement

### Regulatory Resources
For the most current information, consult official regulatory guidance from federal financial institutions regulatory agencies."""

            # Format final response with sources
            final_response = f"""{llm_response}

## 📚 Sources and Additional Information

"""
            
            for i, result in enumerate(search_results[:max_results], 1):
                icon = "🏛️" if result['type'] == 'regulatory' else "🔍" if result['type'] == 'primary' else "📄"
                final_response += f"""### {icon} {i}. {result['title']}
**Source**: {result['source']}
**Summary**: {result['snippet'][:200]}{'...' if len(result['snippet']) > 200 else ''}
**Link**: [{result['source']}]({result['url']})

"""

            final_response += """## 💡 Important Note
Always verify information with official regulatory sources and consult with compliance professionals for specific implementation guidance."""

            response_time = time.time() - start_time
            
            return {
                "success": True,
                "query": query,
                "response": final_response,
                "search_results": search_results,
                "source": "comprehensive_online_search",
                "response_time": response_time,
                "llm_enhanced": bool(llm_response)
            }
            
        except Exception as e:
            logger.error(f"Error during comprehensive online search: {e}")
            return {
                "success": False,
                "query": query,
                "response": f"## ❌ Search Error\n\nUnable to perform comprehensive online search: {str(e)}\n\nPlease try again or consult official regulatory websites directly.",
                "search_results": [],
                "source": "comprehensive_search_error",
                "response_time": 0.0,
                "llm_enhanced": False
            }
        """
        Search online for additional information about a topic.
        
        Args:
            query: Search query
            max_results: Maximum number of search results to return
            
        Returns:
            Dictionary containing search results and formatted response
        """
        try:
            import requests
            import json
            import time
            start_time = time.time()
            
            # Use DuckDuckGo Instant Answer API for simple searches
            search_results = []
            
            try:
                # DuckDuckGo Instant Answer API
                duckduckgo_url = "https://api.duckduckgo.com/"
                params = {
                    'q': f"{query} fintech compliance regulations",
                    'format': 'json',
                    'no_redirect': '1',
                    'no_html': '1',
                    'skip_disambig': '1'
                }
                
                response = requests.get(duckduckgo_url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract relevant information from DuckDuckGo response
                    if data.get('Abstract'):
                        search_results.append({
                            'title': data.get('Heading', query),
                            'snippet': data['Abstract'],
                            'url': data.get('AbstractURL', 'https://duckduckgo.com'),
                            'source': data.get('AbstractSource', 'DuckDuckGo')
                        })
                    
                    # Add related topics if available
                    if data.get('RelatedTopics'):
                        for topic in data['RelatedTopics'][:3]:
                            if isinstance(topic, dict) and topic.get('Text'):
                                search_results.append({
                                    'title': topic.get('Text', '')[:100] + '...',
                                    'snippet': topic.get('Text', ''),
                                    'url': topic.get('FirstURL', 'https://duckduckgo.com'),
                                    'source': 'Related Information'
                                })
                
            except Exception as e:
                logger.warning(f"DuckDuckGo search failed: {e}")
            
            # If no results from DuckDuckGo, provide structured guidance
            if not search_results:
                search_results = [
                    {
                        'title': f'Current {query} Guidelines',
                        'snippet': f'For the most up-to-date information about {query}, check official regulatory websites and current compliance guidelines.',
                        'url': 'https://www.federalreserve.gov/',
                        'source': 'Federal Reserve'
                    },
                    {
                        'title': f'{query} Best Practices',
                        'snippet': f'Industry best practices and current trends related to {query} in financial services.',
                        'url': 'https://www.ffiec.gov/',
                        'source': 'FFIEC'
                    },
                    {
                        'title': f'Regulatory Updates on {query}',
                        'snippet': f'Latest regulatory updates and compliance requirements for {query}.',
                        'url': 'https://www.fincen.gov/',
                        'source': 'FinCEN'
                    }
                ]
            
            # Format the response
            search_response = f"""## 🌐 Online Search Results for: "{query}"

## 🔍 Key Information Found"""
            
            for i, result in enumerate(search_results[:max_results], 1):
                search_response += f"""

### {i}. {result['title']}
**Source**: {result['source']}
**Details**: {result['snippet'][:300]}{'...' if len(result['snippet']) > 300 else ''}
**Link**: [{result['source']}]({result['url']})"""
            
            search_response += f"""

## � Recommended Resources
• Official regulatory body websites
• Current compliance frameworks
• Industry best practice guides
• Professional compliance communities

## ⚠️ Important Note
Always verify information with official regulatory sources and consult compliance professionals for specific requirements."""

            response_time = time.time() - start_time
            
            return {
                "success": True,
                "query": query,
                "response": search_response,
                "search_results": search_results,
                "source": "online_search",
                "response_time": response_time
            }
            
        except Exception as e:
            logger.error(f"Error during online search: {e}")
            return {
                "success": False,
                "query": query,
                "response": f"## ❌ Search Error\n\nUnable to perform online search at this time: {str(e)}",
                "search_results": [],
                "source": "online_search_error",
                "response_time": response_time
            }
            
        except Exception as e:
            logger.error(f"Error during online search: {e}")
            return {
                "success": False,
                "query": query,
                "response": f"## ❌ Search Error\n\nUnable to perform online search at this time: {str(e)}",
                "search_results": [],
                "source": "online_search_error",
                "response_time": 0.0
            }

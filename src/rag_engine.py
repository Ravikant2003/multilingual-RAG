import logging
from typing import List, Dict, Any
import re
from config import config
from src.vector_store import VectorStore
from src.embedding_manager import EmbeddingManager
from src.gemini_client import GeminiClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    """RAG engine using Google Gemini with improved retrieval"""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.gemini_client = GeminiClient() if config.USE_GEMINI else None
        self.chat_history = []
    
    def generate_response(self, query: str, chat_history: List[Dict] = None) -> tuple[str, List[str]]:
        """Generate response using Gemini with RAG"""
        try:
            # Retrieve relevant documents
            retrieved_docs = self._retrieve_documents([query])
            context = self._prepare_context(retrieved_docs)
            
            # Generate with Gemini
            if self.gemini_client:
                response = self.gemini_client.chat_with_context(context, query, chat_history)
            else:
                response = self._generate_fallback_response(context, query)
            
            # Update chat history
            self._update_chat_history(query, response)
            
            # Extract sources
            sources = [doc['content'][:200] + "..." for doc in retrieved_docs[:3]]
            
            return response, sources
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._generate_error_fallback(), []
    
    def _retrieve_documents(self, queries: List[str]) -> List[Dict]:
        """Retrieve relevant documents with better search"""
        all_documents = []
        
        for query in queries:
            # Try different search strategies
            strategies = [
                # Regular hybrid search
                lambda: self.vector_store.hybrid_search(
                    query=query,
                    top_k=config.TOP_K_RESULTS,
                    semantic_weight=config.SEMANTIC_WEIGHT,
                    keyword_weight=config.KEYWORD_WEIGHT
                ),
                # Broader semantic search
                lambda: self.vector_store.semantic_search(
                    query=query,
                    top_k=config.TOP_K_RESULTS * 2
                ),
                # Keyword-only search
                lambda: self.vector_store.keyword_search(
                    query=query,
                    top_k=config.TOP_K_RESULTS
                )
            ]
            
            for strategy in strategies:
                try:
                    documents = strategy()
                    if documents:
                        all_documents.extend(documents)
                        break  # Use first successful strategy
                except Exception as e:
                    logger.warning(f"Search strategy failed: {e}")
                    continue
        
        # If no documents found, get some random documents
        if not all_documents and hasattr(self.vector_store, 'collection'):
            try:
                # Get some random documents to have something to work with
                results = self.vector_store.collection.get(limit=3)
                if results and 'documents' in results:
                    for i, content in enumerate(results['documents']):
                        all_documents.append({
                            'content': content,
                            'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                            'score': 0.1
                        })
            except Exception as e:
                logger.warning(f"Could not get random documents: {e}")
        
        # Remove duplicates
        unique_documents = []
        seen_content = set()
        
        for doc in all_documents:
            content_preview = doc['content'][:50]  # Use shorter preview for dedup
            if content_preview not in seen_content:
                seen_content.add(content_preview)
                unique_documents.append(doc)
        
        unique_documents.sort(key=lambda x: x.get('score', 0), reverse=True)
        return unique_documents[:config.TOP_K_RESULTS * 2]
    
    def _prepare_context(self, documents: List[Dict]) -> str:
        """Prepare context with better formatting"""
        if not documents:
            return "No documents available for analysis."
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            content = doc['content']
            metadata = doc.get('metadata', {})
            
            # Clean the content
            content = self._clean_context_content(content)
            
            # Only include if content is meaningful
            if len(content.strip()) > 10:
                source = metadata.get('source', 'Document')
                page = metadata.get('page', 'N/A')
                language = metadata.get('language', 'unknown')
                
                context_part = f"""--- DOCUMENT {i} ---
Source: {source}
Page: {page}
Language: {language}
Content: {content}
"""
                context_parts.append(context_part)
        
        if not context_parts:
            return "Documents available but content is too short or unclear."
        
        context = "\n".join(context_parts)
        
        # Add instruction header
        context = f"""The following are excerpts from the uploaded document. Some text might be unclear due to OCR limitations. Please analyze what information is available.

{context}"""
        
        return context

    def _clean_context_content(self, content: str) -> str:
        """Clean context content but preserve multilingual text"""
        if not content:
            return ""
        
        # Remove excessive whitespace but preserve line breaks for structure
        content = re.sub(r'[ \t]+', ' ', content)
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        return content.strip()
    
    def _generate_fallback_response(self, context: str, query: str) -> str:
        """Fallback response if Gemini fails"""
        if not context or "no relevant documents" in context.lower():
            return "I couldn't find relevant information in the document to answer your question."
        
        return "I found some relevant information but cannot generate a detailed response at the moment."
    
    def _generate_error_fallback(self) -> str:
        """Error fallback response"""
        return "I encountered an error while processing your question. Please try again."
    
    def _update_chat_history(self, query: str, response: str):
        """Update chat history"""
        if not config.ENABLE_CHAT_MEMORY:
            return
        
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": response})
        
        if len(self.chat_history) > config.MAX_CHAT_HISTORY * 2:
            self.chat_history = self.chat_history[-(config.MAX_CHAT_HISTORY * 2):]
    
    def get_chat_history(self) -> List[Dict]:
        return self.chat_history.copy()
    
    def clear_chat_history(self):
        self.chat_history.clear()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = {
            "llm": "Google Gemini Pro",
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
            "vector_db": "ChromaDB",
            "context_length": "~30,000 tokens",
            "supported_languages": ["Urdu", "Bengali", "English", "Hindi", "Chinese + 100+ more"],
            "features": {
                "multilingual": "Excellent",
                "conversation": "State-of-the-art",
                "reasoning": "Excellent"
            },
            "api_based": True
        }
        
        return info

# Test function
def test_gemini_rag():
    """Test Gemini RAG engine"""
    try:
        from src.embedding_manager import create_embedding_manager
        from src.vector_store import VectorStore
        
        print("🧪 Testing Google Gemini RAG Engine...")
        
        embedding_manager = create_embedding_manager()
        vector_store = VectorStore(embedding_manager)
        rag_engine = RAGEngine(vector_store, embedding_manager)
        
        info = rag_engine.get_model_info()
        print("✅ Gemini RAG Engine Info:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gemini test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_gemini_rag()
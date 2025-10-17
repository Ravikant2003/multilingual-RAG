import logging
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import uuid
import time
from datetime import datetime
from config import config
from src.embedding_manager import EmbeddingManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Manages vector database operations for document storage and retrieval"""
    
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.client = None
        self.collection = None
        self.collection_name = config.COLLECTION_NAME
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Updated ChromaDB settings for new version
            self.client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)
            logger.info(f"ChromaDB client initialized. Persistence path: {config.VECTOR_DB_PATH}")
            
            # Create or get collection
            self._setup_collection()
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
            raise
        
    def _setup_collection(self):
        """Setup or get existing collection"""
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
            
        except Exception as e:
            # Create new collection if it doesn't exist
            logger.info(f"Creating new collection: {self.collection_name}")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Multilingual document store", "created_at": datetime.now().isoformat()}
            )
    
    def _get_chroma_embedding_function(self):
        """Create embedding function compatible with ChromaDB"""
        def chroma_embedding_function(texts: List[str]) -> List[List[float]]:
            embeddings = self.embedding_manager.encode_texts(texts)
            return embeddings.tolist()
        
        return chroma_embedding_function
    
    def add_documents(self, documents: List[Dict]):
        """
        Add documents to the vector store
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        try:
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            metadatas = []
            documents_text = []
            
            for doc in documents:
                doc_id = str(uuid.uuid4())
                content = doc['content']
                metadata = doc['metadata']
                
                # Generate embedding
                embedding = self.embedding_manager.encode_single_text(content)
                
                # Only add if embedding is valid
                if self.embedding_manager.validate_embedding(embedding):
                    ids.append(doc_id)
                    embeddings.append(embedding)
                    metadatas.append(metadata)
                    documents_text.append(content)
            
            if not ids:
                logger.warning("No valid documents to add after embedding validation")
                return
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents_text,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully added {len(ids)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def semantic_search(self, query: str, top_k: int = None, filter_metadata: Dict = None) -> List[Dict]:
        """
        Perform semantic search using embeddings
        """
        if top_k is None:
            top_k = config.TOP_K_RESULTS
        
        try:
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filter_metadata,
                include=["documents", "metadatas", "distances", "embeddings"]
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else None,
                        'score': 1 - (results['distances'][0][i] if results['distances'] else 0),  # Convert to similarity score
                        'embedding': results['embeddings'][0][i] if results['embeddings'] else None
                    })
            
            logger.info(f"Semantic search found {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []
    
    def hybrid_search(self, query: str, top_k: int = None, 
                     semantic_weight: float = None, keyword_weight: float = None) -> List[Dict]:
        """
        Perform hybrid search combining semantic and keyword search
        """
        if top_k is None:
            top_k = config.TOP_K_RESULTS
        if semantic_weight is None:
            semantic_weight = config.SEMANTIC_WEIGHT
        if keyword_weight is None:
            keyword_weight = config.KEYWORD_WEIGHT
        
        try:
            # Get semantic results
            semantic_results = self.semantic_search(query, top_k * 2)  # Get more for reranking
            
            # Get keyword results (using ChromaDB's built-in text search)
            keyword_results = self.keyword_search(query, top_k * 2)
            
            # Combine and rerank results
            combined_results = self._combine_and_rerank_results(
                semantic_results, keyword_results, 
                semantic_weight, keyword_weight, query
            )
            
            # Return top_k results
            final_results = combined_results[:top_k]
            logger.info(f"Hybrid search returned {len(final_results)} results")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return self.semantic_search(query, top_k)  # Fallback to semantic search
    
    def keyword_search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Perform keyword-based search using ChromaDB's where filter
        """
        if top_k is None:
            top_k = config.TOP_K_RESULTS
        
        try:
            # Simple keyword search using ChromaDB's text matching
            # This is a basic implementation - for production, consider integrating with BM25
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else None,
                        'score': self._compute_keyword_score(query, results['documents'][0][i]),
                        'search_type': 'keyword'
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
            return []
    
    def _compute_keyword_score(self, query: str, document: str) -> float:
        """
        Compute simple keyword matching score
        """
        query_terms = set(query.lower().split())
        doc_terms = set(document.lower().split())
        
        if not query_terms:
            return 0.0
        
        # Simple Jaccard similarity
        intersection = len(query_terms.intersection(doc_terms))
        union = len(query_terms.union(doc_terms))
        
        return intersection / union if union > 0 else 0.0
    
    def _combine_and_rerank_results(self, semantic_results: List[Dict], keyword_results: List[Dict],
                                  semantic_weight: float, keyword_weight: float, query: str) -> List[Dict]:
        """
        Combine and rerank results from semantic and keyword search
        """
        # Create a combined pool of unique documents
        all_documents = {}
        
        # Add semantic results
        for result in semantic_results:
            content = result['content']
            if content not in all_documents:
                all_documents[content] = {
                    'content': content,
                    'metadata': result['metadata'],
                    'semantic_score': result.get('score', 0),
                    'keyword_score': 0,
                    'combined_score': 0
                }
        
        # Add keyword results and update scores
        for result in keyword_results:
            content = result['content']
            if content in all_documents:
                all_documents[content]['keyword_score'] = result.get('score', 0)
            else:
                all_documents[content] = {
                    'content': content,
                    'metadata': result['metadata'],
                    'semantic_score': 0,
                    'keyword_score': result.get('score', 0),
                    'combined_score': 0
                }
        
        # Calculate combined scores
        for doc in all_documents.values():
            doc['combined_score'] = (
                semantic_weight * doc['semantic_score'] + 
                keyword_weight * doc['keyword_score']
            )
        
        # Convert to list and sort by combined score
        combined_list = list(all_documents.values())
        combined_list.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Apply reranking if enabled
        if config.ENABLE_RERANKING:
            combined_list = self._rerank_results(combined_list, query)
        
        return combined_list
    
    def _rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Apply cross-encoder reranking for better relevance
        """
        try:
            # Simple implementation - in production, use a cross-encoder model
            # For now, we'll use a combination of existing scores and length normalization
            
            reranked_results = []
            for result in results:
                # Simple reranking: boost shorter, more relevant documents
                content = result['content']
                length_penalty = min(1.0, 500 / len(content))  # Prefer documents around 500 chars
                
                reranked_score = result['combined_score'] * length_penalty
                reranked_results.append({
                    **result,
                    'reranked_score': reranked_score
                })
            
            # Sort by reranked score
            reranked_results.sort(key=lambda x: x['reranked_score'], reverse=True)
            return reranked_results
            
        except Exception as e:
            logger.warning(f"Reranking failed: {str(e)}")
            return results
    
    def search_with_filters(self, query: str, filters: Dict, top_k: int = None) -> List[Dict]:
        """
        Search with metadata filters
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k or config.TOP_K_RESULTS,
                where=filters,
                include=["documents", "metadatas", "distances"]
            )
            
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                        'distance': results['distances'][0][i] if results['distances'] else None,
                        'score': 1 - (results['distances'][0][i] if results['distances'] else 0)
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in filtered search: {str(e)}")
            return []
    
    def get_document_count(self) -> int:
        """Get total number of documents in the collection"""
        try:
            # Get a sample to check if collection has documents
            results = self.collection.get(limit=1)
            return len(results['ids']) if results and 'ids' in results else 0
        except Exception as e:
            logger.error(f"Error getting document count: {str(e)}")
            return 0
        
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            count = self.get_document_count()
            
            # Sample some documents to get language distribution
            sample_results = self.collection.query(
                query_texts=[""],
                n_results=min(100, count),
                include=["metadatas"]
            )
            
            language_dist = {}
            if sample_results['metadatas']:
                for metadata in sample_results['metadatas'][0]:
                    lang = metadata.get('language', 'unknown')
                    language_dist[lang] = language_dist.get(lang, 0) + 1
            
            return {
                "total_documents": count,
                "collection_name": self.collection_name,
                "embedding_dimension": self.embedding_manager.get_embedding_dimension(),
                "language_distribution": language_dist,
                "persistence_path": config.VECTOR_DB_PATH
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}
    
    def delete_documents(self, document_ids: List[str]):
        """Delete documents by IDs"""
        try:
            self.collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents")
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
    
    def clear_collection(self):
        """Clear all documents from the collection"""
        try:
            # Get all IDs first
            results = self.collection.get(include=[])
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info("Cleared all documents from collection")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
    
    def persist(self):
        """Persist the vector store to disk"""
        try:
            # ChromaDB automatically persists, but we can force it
            if hasattr(self.client, 'persist'):
                self.client.persist()
            logger.info("Vector store persisted to disk")
        except Exception as e:
            logger.error(f"Error persisting vector store: {str(e)}")


# Test function
def test_vector_store():
    """Test the vector store functionality"""
    try:
        from src.embedding_manager import create_embedding_manager
        
        # Initialize components
        embedding_manager = create_embedding_manager()
        vector_store = VectorStore(embedding_manager)
        
        # Test data
        test_documents = [
            {
                'content': "This is a test document about artificial intelligence and machine learning.",
                'metadata': {'source': 'test.pdf', 'page': 1, 'language': 'english'}
            },
            {
                'content': "机器学习是人工智能的一个重要分支。",
                'metadata': {'source': 'test.pdf', 'page': 2, 'language': 'chinese'}
            },
            {
                'content': "কৃত্রিম বুদ্ধিমত্তা আধুনিক প্রযুক্তির একটি গুরুত্বপূর্ণ ক্ষেত্র।",
                'metadata': {'source': 'test.pdf', 'page': 3, 'language': 'bengali'}
            }
        ]
        
        # Add documents
        vector_store.add_documents(test_documents)
        
        # Test search
        results = vector_store.semantic_search("artificial intelligence", top_k=2)
        print(f"Semantic search found {len(results)} results")
        
        # Test hybrid search
        hybrid_results = vector_store.hybrid_search("machine learning", top_k=2)
        print(f"Hybrid search found {len(hybrid_results)} results")
        
        # Test stats
        stats = vector_store.get_collection_stats()
        print("Collection stats:", stats)
        
        return True
        
    except Exception as e:
        print(f"Vector store test failed: {str(e)}")
        return False


if __name__ == "__main__":
    test_vector_store()
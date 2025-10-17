import logging
import numpy as np
from typing import List, Dict, Any, Optional
import torch
from sentence_transformers import SentenceTransformer
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages embedding models for multilingual text processing"""
    
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.device = device or config.EMBEDDING_DEVICE
        self.model = None
        self.embedding_dimension = config.EMBEDDING_MODEL_DIMENSION
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Check if GPU is available
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Using device: {self.device}")
            
            # Load the model with optimizations
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
            
            # Verify model loaded correctly
            if hasattr(self.model, 'get_sentence_embedding_dimension'):
                self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Successfully loaded embedding model. Dimension: {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback model if primary fails"""
        fallback_models = config.ALTERNATIVE_EMBEDDING_MODELS
        
        for fallback_model in fallback_models:
            try:
                logger.info(f"Trying fallback model: {fallback_model}")
                self.model_name = fallback_model
                self.model = SentenceTransformer(
                    fallback_model,
                    device=self.device
                )
                
                if hasattr(self.model, 'get_sentence_embedding_dimension'):
                    self.embedding_dimension = self.model.get_sentence_embedding_dimension()
                
                logger.info(f"Successfully loaded fallback model: {fallback_model}")
                return
                
            except Exception as e:
                logger.error(f"Failed to load fallback model {fallback_model}: {str(e)}")
                continue
        
        # If all models fail, raise exception
        raise RuntimeError("All embedding models failed to load")
    
    def encode_texts(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """
        Encode a list of texts into embeddings
        """
        if not texts:
            return np.array([])
        
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        
        try:
            # Normalize texts
            cleaned_texts = [self._clean_text(text) for text in texts]
            
            # Remove empty texts
            valid_texts = [text for text in cleaned_texts if text.strip()]
            if not valid_texts:
                return np.array([])
            
            logger.info(f"Encoding {len(valid_texts)} texts with batch size {batch_size}")
            
            # Generate embeddings
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Important for similarity search
                device=self.device
            )
            
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            # Return zero vectors as fallback
            return np.zeros((len(texts), self.embedding_dimension))
    
    def encode_single_text(self, text: str) -> np.ndarray:
        """Encode a single text into embedding"""
        return self.encode_texts([text])[0]
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text before encoding
        """
        if not text:
            return ""
        
        # Basic cleaning
        text = ' '.join(text.split())  # Remove extra whitespace
        
        # Truncate very long texts (most embedding models have token limits)
        max_length = 1000  # Conservative limit
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        return text
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        """
        try:
            # Ensure embeddings are normalized
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return 0.0
    
    def batch_compute_similarity(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> List[float]:
        """
        Compute similarities between query and multiple document embeddings
        """
        try:
            # Normalize embeddings
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            document_embeddings = document_embeddings / np.linalg.norm(document_embeddings, axis=1, keepdims=True)
            
            similarities = np.dot(document_embeddings, query_embedding)
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"Error batch computing similarities: {str(e)}")
            return [0.0] * len(document_embeddings)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model"""
        if not self.model:
            return {"error": "Model not loaded"}
        
        info = {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": self.device,
            "model_type": "sentence-transformer",
            "multilingual": True,
            "supported_languages": ["urdu", "chinese", "bengali", "english", "hindi"]
        }
        
        # Add model-specific info if available
        if hasattr(self.model, 'tokenizer'):
            info["vocab_size"] = len(self.model.tokenizer)
        
        return info
    
    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """
        Validate if embedding is correct shape and not all zeros
        """
        if embedding is None:
            return False
        
        if not isinstance(embedding, np.ndarray):
            return False
        
        if embedding.shape != (self.embedding_dimension,):
            return False
        
        # Check if embedding is not all zeros
        if np.all(embedding == 0):
            return False
        
        # Check for NaN values
        if np.any(np.isnan(embedding)):
            return False
        
        return True
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        return self.embedding_dimension
    
    def switch_model(self, new_model_name: str):
        """Switch to a different embedding model"""
        try:
            logger.info(f"Switching to model: {new_model_name}")
            self.model_name = new_model_name
            self._load_model()
        except Exception as e:
            logger.error(f"Failed to switch model: {str(e)}")
            raise


class CachedEmbeddingManager(EmbeddingManager):
    """Embedding manager with simple caching mechanism"""
    
    def __init__(self, model_name: str = None, device: str = None):
        super().__init__(model_name, device)
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def encode_texts(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """Encode texts with caching"""
        if not config.ENABLE_CACHING:
            return super().encode_texts(texts, batch_size)
        
        uncached_texts = []
        uncached_indices = []
        cached_embeddings = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            text_hash = self._text_hash(text)
            if text_hash in self.embedding_cache:
                self.cache_hits += 1
                cached_embeddings.append(self.embedding_cache[text_hash])
            else:
                self.cache_misses += 1
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Encode uncached texts
        if uncached_texts:
            new_embeddings = super().encode_texts(uncached_texts, batch_size)
            
            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                text_hash = self._text_hash(text)
                self.embedding_cache[text_hash] = embedding
        
        # Combine cached and new embeddings
        all_embeddings = [None] * len(texts)
        
        # Place cached embeddings
        cache_idx = 0
        for i, text in enumerate(texts):
            text_hash = self._text_hash(text)
            if text_hash in self.embedding_cache:
                all_embeddings[i] = self.embedding_cache[text_hash]
        
        # Place new embeddings
        for idx, embedding in zip(uncached_indices, new_embeddings):
            all_embeddings[idx] = embedding
        
        return np.array(all_embeddings)
    
    def _text_hash(self, text: str) -> str:
        """Create a simple hash for text caching"""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.embedding_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate
        }
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.embedding_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


# Factory function to create embedding manager
def create_embedding_manager(use_caching: bool = True, **kwargs) -> EmbeddingManager:
    """Create an embedding manager instance"""
    if use_caching and config.ENABLE_CACHING:
        return CachedEmbeddingManager(**kwargs)
    else:
        return EmbeddingManager(**kwargs)


# Test function
def test_embedding_manager():
    """Test the embedding manager"""
    try:
        manager = create_embedding_manager()
        
        # Test model info
        info = manager.get_model_info()
        print("Model Info:", info)
        
        # Test encoding
        test_texts = [
            "Hello world",  # English
            "नमस्ते दुनिया",  # Hindi
            "你好世界",  # Chinese
            "হ্যালো বিশ্ব",  # Bengali
        ]
        
        embeddings = manager.encode_texts(test_texts)
        print(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
        
        # Test similarity
        if len(embeddings) >= 2:
            similarity = manager.compute_similarity(embeddings[0], embeddings[1])
            print(f"Similarity between first two texts: {similarity:.4f}")
        
        # Test cache stats if using cached manager
        if isinstance(manager, CachedEmbeddingManager):
            cache_stats = manager.get_cache_stats()
            print("Cache Stats:", cache_stats)
        
        return True
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False


if __name__ == "__main__":
    test_embedding_manager()
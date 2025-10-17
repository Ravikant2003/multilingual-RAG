import pytest
import os
import tempfile
import sys
from unittest.mock import Mock, patch
import numpy as np

# Add the parent directory to Python path so we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Now import from src
from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine
from src.utils import PerformanceTimer, clean_text, validate_pdf_file , detect_language_simple , sanitize_filename , format_file_size
import config

class TestDocumentProcessor:
    """Test document processor functionality"""
    
    def test_clean_text(self):
        """Test text cleaning utility"""
        # Test extra whitespace removal
        dirty_text = "  Hello   World  \n\nThis is a test.  "
        cleaned = clean_text(dirty_text)
        assert cleaned == "Hello World This is a test."
        
        # Test multilingual text preservation
        multilingual_text = "Hello 世界 नमस्ते হ্যালো"
        cleaned_multilingual = clean_text(multilingual_text)
        assert "世界" in cleaned_multilingual
        assert "नमस्ते" in cleaned_multilingual
        assert "হ্যালো" in cleaned_multilingual
    
    def test_document_processor_init(self):
        """Test document processor initialization"""
        processor = DocumentProcessor()
        assert processor is not None
        assert hasattr(processor, 'supported_languages')
        assert 'urdu' in processor.supported_languages
        assert 'chinese' in processor.supported_languages
        assert 'bengali' in processor.supported_languages
    
    @patch('document_processor.fitz.open')
    def test_process_single_document_digital(self, mock_fitz):
        """Test processing digital PDF"""
        # Mock PDF document
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = "This is sample text from a digital PDF."
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz.return_value = mock_doc
        
        processor = DocumentProcessor()
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            documents = processor.process_single_document(temp_path)
            assert documents is not None
            assert len(documents) > 0
            assert 'content' in documents[0]
            assert 'metadata' in documents[0]
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_validate_pdf_file(self):
        """Test PDF file validation"""
        # Test with non-existent file
        is_valid, message = validate_pdf_file("nonexistent.pdf")
        assert not is_valid
        assert "does not exist" in message
        
        # Test with valid temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_path = temp_file.name
            # Write some PDF-like content
            temp_file.write(b"%PDF-1.4 fake pdf content")
        
        try:
            is_valid, message = validate_pdf_file(temp_path)
            # This might fail due to invalid PDF content, but should handle gracefully
            print(f"Validation result: {is_valid}, {message}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestEmbeddingManager:
    """Test embedding manager functionality"""
    
    def test_embedding_manager_init(self):
        """Test embedding manager initialization"""
        # Use a small test model if available, or mock it
        with patch('embedding_manager.SentenceTransformer') as mock_model:
            mock_instance = Mock()
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_model.return_value = mock_instance
            
            manager = EmbeddingManager()
            assert manager is not None
    
    def test_clean_text_method(self):
        """Test text cleaning in embedding manager"""
        manager = EmbeddingManager()
        
        # Test text truncation
        long_text = "A" * 2000
        cleaned = manager._clean_text(long_text)
        assert len(cleaned) <= 1000 + 3  # Account for "..."
        assert "..." in cleaned
        
        # Test normal text
        normal_text = "Hello world"
        cleaned_normal = manager._clean_text(normal_text)
        assert cleaned_normal == normal_text
    
    def test_embedding_validation(self):
        """Test embedding validation"""
        manager = EmbeddingManager()
        
        # Test valid embedding
        valid_embedding = np.ones(config.EMBEDDING_MODEL_DIMENSION)
        assert manager.validate_embedding(valid_embedding)
        
        # Test invalid embedding (wrong shape)
        wrong_shape_embedding = np.ones(100)
        assert not manager.validate_embedding(wrong_shape_embedding)
        
        # Test invalid embedding (all zeros)
        zero_embedding = np.zeros(config.EMBEDDING_MODEL_DIMENSION)
        assert not manager.validate_embedding(zero_embedding)


class TestVectorStore:
    """Test vector store functionality"""
    
    def test_vector_store_init(self):
        """Test vector store initialization"""
        mock_embedding_manager = Mock()
        mock_embedding_manager.get_embedding_dimension.return_value = 384
        
        with patch('vector_store.chromadb.Client') as mock_client:
            mock_collection = Mock()
            mock_client.return_value.get_collection.return_value = mock_collection
            
            vector_store = VectorStore(mock_embedding_manager)
            assert vector_store is not None
            assert vector_store.collection is not None
    
    def test_keyword_score_calculation(self):
        """Test keyword scoring"""
        mock_embedding_manager = Mock()
        vector_store = VectorStore(mock_embedding_manager)
        
        query = "machine learning"
        document = "This is about machine learning and artificial intelligence"
        
        score = vector_store._compute_keyword_score(query, document)
        assert 0 <= score <= 1
        assert score > 0  # Should have some match
    
    def test_result_combination(self):
        """Test combining search results"""
        mock_embedding_manager = Mock()
        vector_store = VectorStore(mock_embedding_manager)
        
        semantic_results = [
            {'content': 'Doc 1', 'score': 0.8, 'metadata': {}},
            {'content': 'Doc 2', 'score': 0.6, 'metadata': {}}
        ]
        
        keyword_results = [
            {'content': 'Doc 2', 'score': 0.9, 'metadata': {}},
            {'content': 'Doc 3', 'score': 0.7, 'metadata': {}}
        ]
        
        combined = vector_store._combine_and_rerank_results(
            semantic_results, keyword_results, 0.7, 0.3, "test query"
        )
        
        assert len(combined) == 3  # Should have all unique documents
        # Doc 2 should have highest combined score
        scores = [doc['combined_score'] for doc in combined]
        assert max(scores) == combined[0]['combined_score']


class TestRAGEngine:
    """Test RAG engine functionality"""
    
    def test_rag_engine_init(self):
        """Test RAG engine initialization"""
        mock_vector_store = Mock()
        mock_embedding_manager = Mock()
        
        with patch('rag_engine.AutoModelForCausalLM') as mock_model, \
             patch('rag_engine.AutoTokenizer') as mock_tokenizer:
            
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.pad_token = None
            mock_tokenizer_instance.eos_token = '<eos>'
            mock_tokenizer.return_value = mock_tokenizer_instance
            
            rag_engine = RAGEngine(mock_vector_store, mock_embedding_manager)
            assert rag_engine is not None
    
    def test_query_decomposition(self):
        """Test query decomposition"""
        mock_vector_store = Mock()
        mock_embedding_manager = Mock()
        rag_engine = RAGEngine(mock_vector_store, mock_embedding_manager)
        
        # Test comparison query
        comparison_query = "compare machine learning and deep learning"
        sub_queries = rag_engine._decompose_query(comparison_query)
        assert len(sub_queries) > 1
        assert "machine learning" in sub_queries[0].lower()
        assert "deep learning" in sub_queries[1].lower()
        
        # Test simple query (should not be decomposed)
        simple_query = "what is ai"
        simple_sub_queries = rag_engine._decompose_query(simple_query)
        assert len(simple_sub_queries) == 1
        assert simple_query in simple_sub_queries
    
    def test_response_cleaning(self):
        """Test response cleaning"""
        mock_vector_store = Mock()
        mock_embedding_manager = Mock()
        rag_engine = RAGEngine(mock_vector_store, mock_embedding_manager)
        
        # Test repetitive text cleaning
        repetitive_response = "This is a test. This is a test. This is a test."
        cleaned = rag_engine._clean_response(repetitive_response)
        assert "This is a test." in cleaned
        assert cleaned.count("This is a test.") == 1
        
        # Test long response truncation
        long_response = "A" * 1500
        cleaned_long = rag_engine._clean_response(long_response)
        assert len(cleaned_long) <= 1000 + 3  # Account for "..."
        
        # Test punctuation addition
        no_punctuation = "This has no punctuation"
        cleaned_punct = rag_engine._clean_response(no_punctuation)
        assert cleaned_punct.endswith('.')


class TestUtils:
    """Test utility functions"""
    
    def test_performance_timer(self):
        """Test performance timer"""
        with PerformanceTimer("Test Operation") as timer:
            import time
            time.sleep(0.1)
        
        elapsed = timer.get_elapsed_time()
        assert elapsed >= 0.1
        assert elapsed < 1.0  # Should be reasonable
    
    def test_language_detection_simple(self):
        """Test simple language detection"""
        
        # Test English
        english_text = "This is English text"
        assert detect_language_simple(english_text) == "english"
        
        # Test Chinese
        chinese_text = "这是中文文本"
        assert detect_language_simple(chinese_text) == "chinese"
        
        # Test unknown (mixed or too short)
        mixed_text = "A"
        assert detect_language_simple(mixed_text) == "unknown"
    
    def test_file_size_formatting(self):
        """Test file size formatting""" 
        
        assert format_file_size(500) == "500.0 B"
        assert format_file_size(1500) == "1.5 KB"
        assert format_file_size(1500000) == "1.4 MB"
        assert "GB" in format_file_size(1500000000)
    
    def test_sanitize_filename(self):
        """Test filename sanitization"""
        
        
        dangerous_name = "../../etc/passwd"
        safe_name = sanitize_filename(dangerous_name)
        assert ".." not in safe_name
        assert "/" not in safe_name
        
        long_name = "a" * 300 + ".pdf"
        safe_long_name = sanitize_filename(long_name)
        assert len(safe_long_name) <= 255


class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_end_to_end_processing(self):
        """Test complete document processing pipeline"""
        # This is a high-level integration test
        # We'll use mocks to avoid actual model loading
        
        with patch('embedding_manager.SentenceTransformer') as mock_embedding, \
             patch('vector_store.chromadb.Client') as mock_chroma, \
             patch('rag_engine.AutoModelForCausalLM') as mock_llm:
            
            # Setup mocks
            mock_embedding_instance = Mock()
            mock_embedding_instance.encode.return_value = np.ones((1, 384))
            mock_embedding.return_value = mock_embedding_instance
            
            mock_collection = Mock()
            mock_chroma.return_value.get_collection.return_value = mock_collection
            mock_chroma.return_value.create_collection.return_value = mock_collection
            
            # Initialize components
            embedding_manager = EmbeddingManager()
            vector_store = VectorStore(embedding_manager)
            
            # Test document addition
            test_documents = [
                {
                    'content': 'Test document content',
                    'metadata': {'source': 'test.pdf', 'page': 1, 'language': 'english'}
                }
            ]
            
            vector_store.add_documents(test_documents)
            assert mock_collection.add.called
            
            # Test search
            mock_collection.query.return_value = {
                'documents': [['Test document content']],
                'metadatas': [[{'source': 'test.pdf', 'page': 1}]],
                'distances': [[0.1]]
            }
            
            results = vector_store.semantic_search("test query")
            assert len(results) > 0
            assert 'content' in results[0]


# Run tests
if __name__ == "__main__":
    # Import numpy for tests
    import numpy as np
    
    print("Running RAG System Tests...")
    
    # Run individual test classes
    test_classes = [
        TestDocumentProcessor,
        TestEmbeddingManager, 
        TestVectorStore,
        TestRAGEngine,
        TestUtils,
        TestIntegration
    ]
    
    all_passed = True
    
    for test_class in test_classes:
        print(f"\n=== Testing {test_class.__name__} ===")
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [method for method in dir(test_instance) 
                       if method.startswith('test_')]
        
        for method_name in test_methods:
            try:
                method = getattr(test_instance, method_name)
                method()
                print(f"✅ {method_name} - PASSED")
            except Exception as e:
                print(f"❌ {method_name} - FAILED: {str(e)}")
                all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print(f"{'='*50}")
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)
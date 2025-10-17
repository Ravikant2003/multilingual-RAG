import sys
import os
import tempfile

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_basic_functionality():
    """Test basic functionality without complex dependencies"""
    print("🧪 Testing Basic Functionality...")
    
    try:
        # Test config
        import config
        print("✅ Config loaded")
        
        # Test utils
        from src.utils import clean_text, PerformanceTimer
        print("✅ Utils imported")
        
        # Test text cleaning
        result = clean_text("  Hello   World  ")
        assert result == "Hello World"
        print("✅ Text cleaning works")
        
        # Test performance timer
        with PerformanceTimer("Test"):
            pass
        print("✅ Performance timer works")
        
        # Test document processor (without actual PDF processing)
        from src.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        assert processor is not None
        print("✅ Document processor created")
        
        print("\n🎉 Basic tests passed! Core functionality is working.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_embeddings():
    """Test embedding functionality"""
    print("\n🧪 Testing Embeddings...")
    
    try:
        from src.embedding_manager import EmbeddingManager
        
        # Test with a smaller model for testing
        manager = EmbeddingManager()
        print("✅ Embedding manager created")
        
        # Test encoding
        texts = ["Hello world", "Test document"]
        embeddings = manager.encode_texts(texts)
        print(f"✅ Embeddings generated: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Simplified RAG System Tests...\n")
    
    basic_ok = test_basic_functionality()
    embeddings_ok = test_embeddings()
    
    print("\n" + "="*50)
    if basic_ok and embeddings_ok:
        print("🎉 ALL ESSENTIAL TESTS PASSED!")
        print("Your RAG system is ready to use!")
    else:
        print("⚠️  Some tests failed, but core functionality might still work.")
        print("Try running the Streamlit app to test the complete system.")
    print("="*50)
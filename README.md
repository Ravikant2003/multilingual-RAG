# Multilingual PDF RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system designed to process multilingual PDFs, extract information, and provide intelligent question-answering capabilities across multiple languages including English, Hindi, Bengali, Chinese, and Urdu.


The Youtube working demonstration: https://youtu.be/snctDz5fyAA 

## 🚀 Features

### Core Capabilities
- **Multilingual Processing**: Supports English, Hindi, Bengali, Chinese, and Urdu
- **Dual PDF Handling**: Processes both scanned (OCR) and digital PDFs
- **Advanced RAG Pipeline**: Implements cutting-edge retrieval techniques
- **Streamlit Web Interface**: User-friendly web application
- **Scalable Architecture**: Designed to handle up to 1TB of data

### Advanced RAG Features
- ✅ Chat Memory Functionality
- ✅ Query Decomposition  
- ✅ Optimized Chunking Algorithms
- ✅ Hybrid Search (Semantic + Keyword)
- ✅ High-Performance Vector Database
- ✅ Intelligent Model Selection
- ✅ Reranking Algorithms
- ✅ Metadata Filtering

## 📋 System Architecture

### Flow Diagram

```
User Query (Streamlit UI)
     ↓
Query Preprocessing
     ↓
Query Decomposition
     ↓
Hybrid Search → Semantic + Keyword Search
     ↓
Vector Database (ChromaDB)
     ↓
Reranking & Filtering
     ↓
Context Enhancement
     ↓
LLM Generation (Gemini-2.5-Flash-Lite)
     ↓
Response with Memory
     ↓
Streamlit UI Display
```

### Component Architecture
```
PDF Input → Document Processor → Embedding Manager → Vector Store
                                      ↓
User Query → RAG Engine → Response Generator
                                      ↓
Chat Memory ← Response → Streamlit Interface
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Tesseract OCR
- 4GB+ RAM recommended

### Setup
```bash
# Clone repository
git clone <repository-url>
cd multilingual-rag

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR (Ubuntu)
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-hin tesseract-ocr-ben tesseract-ocr-chi-sim tesseract-ocr-urd

# Install Tesseract OCR (Windows)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

## 🚀 Quick Start

### Running the Application
```bash
# Start the Streamlit application
streamlit run app.py

# Access the web interface
# http://localhost:8501
```

### Basic Usage through Streamlit

1. **Launch the application**: Run `streamlit run app.py`
2. **Upload PDFs**: Use the sidebar file uploader to add your documents
3. **Process Documents**: Click "Process Documents" to extract and index content
4. **Ask Questions**: Use the chat interface to ask questions about your documents
5. **View Results**: See answers with source citations and confidence scores

## 📁 Project Structure

```
multilingual-rag/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── config.py                   # Configuration settings
├── README.md                   # This file
├── src/
│   ├── __init__.py
│   ├── document_processor.py   # PDF & OCR processing
│   ├── embedding_manager.py    # Embedding model handling
│   ├── vector_store.py         # Vector database operations
│   ├── rag_engine.py           # Core RAG logic
│   ├── gemini_client.py        # Gemini model integration
│   └── utils.py                # Utility functions
├── data #vectordv
```

## 🔧 Configuration

### Key Configuration Options
```python
# config.py
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "gemini-2.5-flash-lite"
VECTOR_DB = "chromadb"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
```

### Environment Variables
```bash
export GOOGLE_API_KEY="your-google-api-key"
```

## 💡 Usage Examples

### Through Streamlit Interface

1. **Document Upload**:
   - Drag and drop PDF files in the sidebar
   - Support for multiple files simultaneously
   - Progress indicators for processing

2. **Interactive Chat**:
   - Real-time question answering
   - Conversation history
   - Source document references
   - Confidence scoring

3. **Document Management**:
   - View processed documents
   - Clear conversation history
   - Reset vector store

### Programmatic Usage
```python
from src.document_processor import DocumentProcessor
from src.rag_engine import RAGEngine

# Process documents
processor = DocumentProcessor()
documents = processor.process_pdf("document.pdf", language="auto")

# Initialize RAG engine
rag = RAGEngine()
rag.vector_store.add_documents(documents)

# Ask questions
response = rag.ask("What are the key findings?")
print(response.answer)
print(response.sources)
```

## 🎯 Advanced Features

### Hybrid Search
- **Semantic Search**: Dense vector embeddings
- **Keyword Search**: BM25 algorithm
- **Combined Ranking**: Weighted score fusion

### Query Decomposition
```python
# Complex query breakdown
query = "Compare the economic policies of 2020 and 2021"
# Decomposed to:
# - "economic policies 2020"
# - "economic policies 2021" 
# - "comparison framework"
```

### Reranking
- Cross-encoder models for relevance scoring
- Metadata-based filtering
- Temporal relevance weighting

## 🔍 Model Selection

### Embedding Models
- **Primary**: `paraphrase-multilingual-MiniLM-L12-v2` (420MB)
- **Fallback**: `multilingual-e5-small` (smaller alternative)

### LLM Selection
- **Primary**: Gemini-2.5-Flash-Lite

### OCR Engine
- **Tesseract** with language packs
- Support for Hindi, Bengali, Chinese, Urdu

## 🚀 Scaling to 1TB

### Architecture Decisions
- **ChromaDB**: Horizontal scaling capability
- **Batch Processing**: Parallel document ingestion
- **Streaming**: Memory-efficient large file handling
- **Distributed Processing**: Future-ready architecture



## ⚠️ Challenges Faced

### 1. OCR Integration Limitations
- **Basic PyTesseract**: Limited accuracy for complex multilingual documents
- **Language Detection**: Challenges in accurately detecting mixed-language content
- **Format Preservation**: Difficulty in maintaining original document structure

### 2. LLM Constraints
- **Model Size Limitations**: Explored smaller models but faced quality trade-offs
- **Multilingual Capabilities**: Smaller models struggled with non-English languages
- **Context Understanding**: Limited reasoning capabilities in compact models

### 3. Language Processing
- **Accuracy Issues**: Current answer retrieval accuracy needs improvement
- **Context Relevance**: Challenges in ensuring answers are properly grounded in context
- **Language Switching**: Handling documents with multiple languages mixed

### 4. Streamlit Integration
- **State Management**: Handling conversation memory across sessions
- **File Processing**: Efficient handling of large PDF uploads
- **UI/UX**: Creating intuitive interface for complex RAG operations

## 🔮 Future Enhancements

### Planned Improvements
- [ ] Advanced OCR solutions (Google Vision, Azure OCR)
- [ ] Integration of smaller, efficient LLMs with better multilingual support
- [ ] Enhanced language detection and processing
- [ ] Improved context-aware answer generation
- [ ] Better chunking strategies for multilingual content
- [ ] Advanced Streamlit features like document preview and annotation

## 🚧 Important Note

**Development Timeline**: This project was completed within approximately 8-10 hours instead of the allocated 72 hours due to prior commitments. The compressed timeline impacted:

- Depth of OCR integration and accuracy
- Exploration of smaller LLM alternatives
- Comprehensive testing across all language combinations


Despite these constraints, the system demonstrates a solid foundation for multilingual RAG with all core requirements implemented and a functional Streamlit web interface.



## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.



---

**Built under time constraints - demonstrates core RAG capabilities with multilingual support and Streamlit web interface**

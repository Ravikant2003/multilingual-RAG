import streamlit as st
import os
import tempfile
from src.document_processor import DocumentProcessor
from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStore
from src.rag_engine import RAGEngine
import time
from typing import List, Dict

# Page configuration
st.set_page_config(
    page_title="Multilingual RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e6f3ff;
        border-left: 4px solid #1f77b4;
    }
    .assistant-message {
        background-color: #f0f8ff;
        border-left: 4px solid #ff7f0e;
    }
    .file-uploader {
        border: 2px dashed #ccc;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
    }
    .stStatus {
        border: 1px solid #e6e6e6;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class MultilingualRAGApp:
    def __init__(self):
        self.init_session_state()
        self.setup_components()
    
    def init_session_state(self):
        """Initialize session state variables"""
        default_state = {
            'messages': [],
            'vector_store': None,
            'rag_engine': None,
            'documents_processed': False,
            'processing_status': "",
            'uploaded_files': [],
            'current_document_name': None,
            'chat_input': "",
            'processing_error': None
        }
        
        for key, value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def clear_current_document(self):
        """Clear the current document and reset chat"""
        try:
            if st.session_state.vector_store:
                st.session_state.vector_store.clear_collection()
            
            st.session_state.messages = []
            st.session_state.documents_processed = False
            st.session_state.current_document_name = None
            st.session_state.uploaded_files = []
            st.session_state.processing_error = None
            
            st.success("✅ Current document cleared! You can upload a new document.")
            st.rerun()
        except Exception as e:
            st.error(f"Error clearing document: {str(e)}")

    def setup_components(self):
        """Initialize RAG components"""
        try:
            if st.session_state.rag_engine is None:
                with st.spinner("Loading AI models..."):
                    embedding_manager = EmbeddingManager()
                    vector_store = VectorStore(embedding_manager)
                    st.session_state.vector_store = vector_store
                    st.session_state.rag_engine = RAGEngine(vector_store, embedding_manager)
                    st.session_state.processing_error = None
        except Exception as e:
            error_msg = f"Error initializing components: {str(e)}"
            st.session_state.processing_error = error_msg
            st.error(error_msg)
    
    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        with st.sidebar:
            st.title("⚙️ Configuration")
            
            # System Status Overview
            st.subheader("📊 System Status")
            if st.session_state.processing_error:
                st.error("❌ System Error")
                st.code(st.session_state.processing_error[:200] + "..." if len(st.session_state.processing_error) > 200 else st.session_state.processing_error)
            elif st.session_state.documents_processed and st.session_state.current_document_name:
                st.success(f"**Ready:** {st.session_state.current_document_name}")
                doc_count = st.session_state.vector_store.get_document_count() if st.session_state.vector_store else 0
                st.metric("Document Chunks", doc_count)
                st.metric("Chat Messages", len(st.session_state.messages))
            else:
                st.warning("**Status:** Waiting for Document")
            
            st.divider()
            
            # Document Management
            st.subheader("📄 Document Management")
            st.info("**Current Mode:** Single Document")
            st.warning("Uploading a new document automatically clears the previous one.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Clear Document", use_container_width=True, 
                           disabled=not st.session_state.documents_processed):
                    self.clear_current_document()
            with col2:
                if st.button("🗣️ Clear Chat", use_container_width=True,
                           disabled=len(st.session_state.messages) == 0):
                    st.session_state.messages = []
                    st.success("Chat history cleared!")
                    st.rerun()
            
            st.divider()
            
            # AI Model Settings
            st.subheader("🤖 AI Model Settings")
            
            with st.expander("Model Configuration", expanded=False):
                st.markdown("**Language Model**")
                st.info("Using: Google Gemini")
                
                # Note: These sliders are currently decorative
                # To make them functional, you'd need to update config and restart components
                llm_temp = st.slider(
                    "Creativity", 
                    0.0, 1.0, 0.7, 0.1,
                    help="Higher values make responses more creative, lower values more focused"
                )
                
                max_tokens = st.slider(
                    "Response Length", 
                    100, 2000, 1024, 100,
                    help="Maximum length of generated responses"
                )
            
            st.divider()
            
            # RAG Settings
            st.subheader("🔍 Search Settings")
            
            with st.expander("Retrieval Configuration", expanded=False):
                # Note: These sliders are currently decorative
                chunk_size = st.slider(
                    "Chunk Size", 
                    100, 1000, 300, 50,
                    help="Size of text chunks for processing"
                )
                
                top_k = st.slider(
                    "Retrieval Count", 
                    1, 10, 3,
                    help="Number of document chunks to retrieve"
                )
                
                st.markdown("**Search Strategy**")
                semantic_weight = st.slider(
                    "Semantic Weight", 
                    0.0, 1.0, 0.7, 0.1,
                    help="Weight for semantic similarity search"
                )
                
                keyword_weight = st.slider(
                    "Keyword Weight", 
                    0.0, 1.0, 0.3, 0.1,
                    help="Weight for keyword matching search"
                )
                
                # Show combined weight
                total_weight = semantic_weight + keyword_weight
                if abs(total_weight - 1.0) > 0.01:
                    st.warning(f"Total weight: {total_weight:.1f} (recommended: 1.0)")
                else:
                    st.success(f"Total weight: {total_weight:.1f}")
            
            st.divider()
            
            # Language Support
            st.subheader("🌍 Language Support")
            with st.expander("Supported Languages", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.success("✅ Urdu")
                    st.success("✅ Bengali")
                    st.success("✅ Hindi")
                with col2:
                    st.success("✅ Chinese")
                    st.success("✅ English")
                    st.info("+ 100+ languages")
                
                st.caption("Multilingual embedding model supports 50+ languages")
            
            st.divider()
            
            # Debug & Tools
            st.subheader("🔧 Tools & Debug")
            
            if st.button("🔍 Debug Document Content", use_container_width=True,
                        disabled=not st.session_state.documents_processed):
                self.debug_document_content()
                
            if st.button("🔄 Refresh Models", use_container_width=True):
                st.session_state.rag_engine = None
                st.session_state.vector_store = None
                st.session_state.processing_error = None
                st.rerun()
            
            # Model Information
            with st.expander("Technical Info", expanded=False):
                if st.session_state.rag_engine:
                    try:
                        model_info = st.session_state.rag_engine.get_model_info()
                        st.write("**Current Models:**")
                        st.write(f"• LLM: {model_info.get('llm', 'Google Gemini Pro')}")
                        st.write(f"• Embeddings: {model_info.get('embedding_model', 'Multilingual MiniLM')}")
                        st.write(f"• Vector DB: {model_info.get('vector_db', 'ChromaDB')}")
                    except Exception as e:
                        st.error(f"Error getting model info: {e}")
                else:
                    st.info("Models not loaded yet")
            
            st.divider()
            
            # Quick Actions
            st.subheader("⚡ Quick Actions")
            
            quick_questions = [
                "What is this document about?",
                "Summarize the main points",
                "What languages are used in this document?",
                "How many pages does this have?",
                "What are the key topics?"
            ]
            
            for question in quick_questions:
                if st.button(f"❓ {question}", use_container_width=True, key=f"quick_{question}"):
                    if st.session_state.documents_processed:
                        st.session_state.chat_input = question
                        st.rerun()
                    else:
                        st.warning("Please upload a document first")
    
    def debug_document_content(self):
        """Debug method to see what's in the vector store"""
        if not st.session_state.vector_store:
            st.error("Vector store not initialized")
            return
            
        try:
            results = st.session_state.vector_store.collection.get(limit=5)
            if not results or 'documents' not in results or not results['documents']:
                st.warning("No documents found in vector store")
                return
                
            st.subheader("🔍 Document Content Debug")
            st.write(f"Total documents in collection: {len(results['documents'])}")
            
            for i, doc in enumerate(results['documents'][:3]):
                st.write(f"**Document {i+1}:**")
                st.text_area(f"Content {i+1}", doc, height=150, key=f"debug_doc_{i}")
                
                if 'metadatas' in results and results['metadatas'] and i < len(results['metadatas'][0]):
                    st.json(results['metadatas'][0][i])
            
            # Content quality analysis
            st.subheader("📊 Content Quality Analysis")
            for i, doc in enumerate(results['documents'][:3]):
                content = doc
                words = content.split()
                unique_words = set(words)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"Doc {i+1} Words", len(words))
                with col2:
                    st.metric(f"Doc {i+1} Unique", len(unique_words))
                with col3:
                    if len(content) > 0:
                        alpha_ratio = sum(1 for char in content if char.isalpha()) / len(content)
                        st.metric(f"Doc {i+1} Alpha Ratio", f"{alpha_ratio:.2f}")
                        if alpha_ratio < 0.3:
                            st.error("⚠️ Low quality - possible OCR issues")
                            
        except Exception as e:
            st.error(f"Debug error: {e}")
    
    def render_file_upload(self):
        """Render file upload section"""
        st.markdown('<div class="main-header">📚 Multilingual RAG System</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📤 Upload PDF Document")
            
            # Show current document if any
            if st.session_state.current_document_name:
                st.info(f"📄 Current Document: **{st.session_state.current_document_name}**")
                st.warning("⚠️ Uploading a new document will clear the current one and all chat history.")
            
            st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose a PDF file (will replace current document)",
                type="pdf",
                accept_multiple_files=False,
                help="Upload a single PDF document. Previous document will be automatically cleared."
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            if uploaded_file:
                self.process_uploaded_files([uploaded_file])
        
        with col2:
            st.subheader("ℹ️ Instructions")
            st.markdown("""
            1. **Upload PDF** - Single document at a time
            2. **Auto-clear** - Previous document is automatically removed
            3. **Start chatting** - Ask questions about the current document
            
            **Supported languages:**
            - Urdu 
            - Chinese   
            - Bengali 
            - English 
            - Hindi 
            
            **Current mode:** Single document
            
            **💡 Tip:** Start with simple questions like:
            - "What is this document about?"
            - "What languages are used?"
            - "Summarize the main points"
            """)
    
    def process_uploaded_files(self, uploaded_files):
        """Process uploaded PDF files and clear previous documents"""
        if not uploaded_files:
            return
        
        try:
            with st.status("🔄 Processing documents...", expanded=True) as status:
                # CLEAR PREVIOUS DATA before processing new files
                if st.session_state.vector_store:
                    status.update(label="🗑️ Clearing previous documents...")
                    st.session_state.vector_store.clear_collection()
                    st.session_state.messages = []
                    st.session_state.documents_processed = False
                    st.session_state.processing_error = None
                
                # Create temporary directory for uploaded files
                with tempfile.TemporaryDirectory() as temp_dir:
                    file_paths = []
                    
                    # Save uploaded files
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        file_paths.append(file_path)
                    
                    # Initialize document processor
                    doc_processor = DocumentProcessor()
                    
                    # Process documents
                    status.update(label="📄 Extracting text from PDFs...")
                    documents = doc_processor.process_documents(file_paths)
                    
                    if documents:
                        status.update(label="🔍 Creating embeddings...")
                        st.session_state.vector_store.add_documents(documents)
                        
                        status.update(label="✅ Setting up RAG engine...")
                        st.session_state.documents_processed = True
                        st.session_state.current_document_name = uploaded_files[0].name
                        
                        # Show processing statistics (with error handling)
                        try:
                            if hasattr(doc_processor, 'get_processing_stats'):
                                stats = doc_processor.get_processing_stats(documents)
                                st.success(f"✅ Successfully processed: {uploaded_files[0].name}")
                                st.info(f"📊 Processing stats: {stats['total_chunks']} chunks, {stats['total_characters']} characters")
                                
                                if 'languages' in stats and stats['languages']:
                                    lang_info = ", ".join([f"{lang}({count})" for lang, count in stats['languages'].items()])
                                    st.info(f"🌍 Languages detected: {lang_info}")
                            else:
                                st.success(f"✅ Successfully processed: {uploaded_files[0].name}")
                                st.info(f"📊 Processing stats: {len(documents)} chunks extracted")
                                
                        except Exception as e:
                            st.success(f"✅ Successfully processed: {uploaded_files[0].name}")
                            st.info(f"📊 Processing stats: {len(documents)} chunks extracted")
                        
                        st.info("📝 Previous documents and chat history have been cleared.")
                    else:
                        st.error("❌ No text could be extracted from the uploaded documents.")
                        st.info("💡 Try uploading a different PDF or check if the file contains readable text.")
                            
        except Exception as e:
            error_msg = f"❌ Error processing documents: {str(e)}"
            st.session_state.processing_error = error_msg
            st.error(error_msg)

    def render_chat_interface(self):
        """Render the main chat interface"""
        if not st.session_state.documents_processed:
            st.warning("⚠️ Please upload and process a PDF document first to start chatting.")
            return
        
        st.subheader("💬 Chat with Your Documents")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"**Source {i+1}:** {source[:200]}...")
        
        # Chat input with quick action support
        chat_placeholder = "Ask a question about your document..."
        if st.session_state.chat_input:
            chat_placeholder = st.session_state.chat_input
        
        if prompt := st.chat_input(chat_placeholder):
            # Clear the quick action after use
            st.session_state.chat_input = ""
            
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching documents..."):
                    try:
                        if not st.session_state.rag_engine:
                            st.error("RAG engine not initialized. Please refresh the page.")
                            return
                            
                        response, sources = st.session_state.rag_engine.generate_response(
                            prompt, 
                            chat_history=st.session_state.messages[:-1]
                        )
                        
                        st.markdown(response)
                        
                        # Display sources
                        if sources:
                            with st.expander("📚 View Sources"):
                                for i, source in enumerate(sources):
                                    st.markdown(f"**Source {i+1}:** {source[:200]}...")
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": sources
                        })
                        
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error while generating a response: {str(e)}"
                        st.markdown(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })
    
    def render_system_status(self):
        """Render system status information"""
        with st.expander("🔍 System Status & Information", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                doc_count = st.session_state.vector_store.get_document_count() if st.session_state.vector_store else 0
                st.metric("Document Chunks", doc_count)
            
            with col2:
                st.metric("Chat Messages", len(st.session_state.messages))
            
            with col3:
                if st.session_state.documents_processed and st.session_state.current_document_name:
                    status = "Ready"
                else:
                    status = "Waiting"
                st.metric("System Status", status)
            
            # Current document info
            if st.session_state.current_document_name:
                st.subheader("📄 Current Document")
                st.write(f"**File:** {st.session_state.current_document_name}")
                st.write(f"**Chunks:** {doc_count} text segments")
                
                if st.button("🗑️ Clear Current Document", key="status_clear"):
                    self.clear_current_document()
            
            # Model information
            st.subheader("🤖 Model Information")
            if st.session_state.rag_engine:
                try:
                    model_info = st.session_state.rag_engine.get_model_info()
                    for key, value in model_info.items():
                        if isinstance(value, dict):
                            st.write(f"**{key}:**")
                            for k, v in value.items():
                                st.write(f"  - {k}: {v}")
                        else:
                            st.write(f"**{key}:** {value}")
                except Exception as e:
                    st.error(f"Error getting model info: {e}")
            else:
                st.info("RAG engine not initialized")
            
            # Performance tips
            st.subheader("💡 Performance Tips")
            st.info("""
            - Start with simple questions about the document
            - Use the quick action buttons for common queries
            - If responses are poor, check document content with debug tool
            - Try rephrasing questions if you get 'no information found'
            - For poor OCR quality, try uploading a clearer PDF version
            """)
    
    def run(self):
        """Main application runner"""
        # Show error banner if there's a processing error
        if st.session_state.processing_error:
            st.error(f"🚨 System Error: {st.session_state.processing_error}")
        
        self.render_sidebar()
        
        tab1, tab2, tab3 = st.tabs(["📁 Upload Documents", "💬 Chat", "⚙️ System Status"])
        
        with tab1:
            self.render_file_upload()
        
        with tab2:
            self.render_chat_interface()
        
        with tab3:
            self.render_system_status()

def main():
    # App header
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 0.5rem; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0;'>Multilingual RAG System</h1>
        <p style='color: white; margin: 0.5rem 0 0 0;'>Process PDFs in Urdu, Chinese, Bengali & English with Google Gemini</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize and run the app
    app = MultilingualRAGApp()
    app.run()

if __name__ == "__main__":
    main()
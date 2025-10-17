import os
from typing import Dict, List

class Config:
    """Configuration class for Multilingual RAG System"""
    
    # ==================== PATHS & DIRECTORIES ====================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # ==================== DOCUMENT PROCESSING ====================
    # Supported languages for OCR
    SUPPORTED_LANGUAGES = {
        'urdu': 'urd',      # Urdu
        'chinese': 'chi_sim+chi_tra',  # Simplified + Traditional Chinese
        'bengali': 'ben',   # Bengali
        'english': 'eng',   # English
        'hindi': 'hin'      # Hindi
    }
    
    # Default OCR language (auto-detect)
    DEFAULT_OCR_LANGUAGES = "urd+chi_sim+chi_tra+ben+eng+hin"
    
    # Chunking settings
    CHUNK_SIZE = 300
    CHUNK_OVERLAP = 50
    MAX_CHUNK_LENGTH = 500
    
    # PDF processing settings
    MAX_FILE_SIZE_MB = 100
    SUPPORTED_PDF_TYPES = ['application/pdf']
    
    # ==================== EMBEDDING MODELS ====================
    # Small multilingual embedding model - ACTUAL MODEL WE'RE USING
    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_MODEL_DIMENSION = 384  # Dimension for the chosen model
    EMBEDDING_DEVICE = "cpu"  # Use "cuda" if GPU available
    
    # Alternative smaller models (fallback options)
    ALTERNATIVE_EMBEDDING_MODELS = [
        "intfloat/multilingual-e5-small",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]
    
    # ==================== GEMINI CONFIG ====================
    USE_GEMINI = True
    GEMINI_MODEL = "gemini-2.5-flash-lite"
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # If using Gemini, we don't need local LLM settings
    if USE_GEMINI:
        LLM_MODEL_NAME = "gemini-2.5-flash-lite"
        LLM_MAX_TOKENS = 1024
        LLM_TEMPERATURE = 0.3

    PROMPT_TEMPLATES = {
    "query_decomposition": """
    Break down the following question into simpler sub-questions:
    
    Question: {question}
    
    Sub-questions:
    1.
    """,
    
    "summarization": """
    Summarize the following text concisely:
    
    Text: {text}
    
    Summary:
    """,
    
    "qa_prompt": """
    Use the context below to answer the question. Be concise and factual.
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """,
    
    "multilingual_qa": """
    <start_of_turn>user
    You are a helpful multilingual assistant. Answer the question based ONLY on the provided context.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    
    CHAT HISTORY (for reference):
    {chat_history}
    
    Instructions:
    - Answer using only information from the context
    - If context doesn't contain answer, say "I cannot find this information in the document."
    - Be clear and concise
    - Support multiple languages including Urdu and Bengali
    
    Please provide your answer:<end_of_turn>
    <start_of_turn>model
    """
}




    # ==================== VECTOR DATABASE ====================
    VECTOR_DB_TYPE = "chromadb"  # Options: chromadb, faiss
    VECTOR_DB_PATH = os.path.join(DATA_DIR, "vector_store")
    COLLECTION_NAME = "multilingual_documents"
    
    # Search settings
    TOP_K_RESULTS = 3
    SIMILARITY_THRESHOLD = 0.6
    
    # ==================== RAG SETTINGS ====================
    # Hybrid search weights
    SEMANTIC_WEIGHT = 0.7
    KEYWORD_WEIGHT = 0.3
    
    # Reranking settings
    ENABLE_RERANKING = True
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_TOP_K = 5
    
    # Memory settings
    MAX_CHAT_HISTORY = 10
    ENABLE_CHAT_MEMORY = True
    
    # ==================== PERFORMANCE SETTINGS ====================
    # Batch processing
    BATCH_SIZE = 32
    MAX_CONCURRENT_PROCESSES = 2
    
    # Caching
    ENABLE_CACHING = True
    CACHE_EXPIRY_HOURS = 24
    
    # ==================== STREAMLIT UI SETTINGS ====================
    UI_SETTINGS = {
        "page_title": "Multilingual RAG System",
        "page_icon": "📚",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }
    
    # ==================== LANGUAGE DETECTION ====================
    LANGUAGE_DETECTION_MODEL = "papluca/xlm-roberta-base-language-detection"
    CONFIDENCE_THRESHOLD = 0.7
    
    # ==================== ERROR HANDLING & LOGGING ====================
    LOG_LEVEL = "INFO"
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    
    # ==================== GOOGLE CLOUD VISION CONFIG ====================
    USE_GOOGLE_VISION = True
    GOOGLE_VISION_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    GOOGLE_VISION_MAX_RESULTS = 10

    # Supported languages for Vision API (it auto-detects)
    VISION_API_FEATURES = ["DOCUMENT_TEXT_DETECTION"]  # Use DOCUMENT_TEXT_DETECTION for dense text
    # ==================== SECURITY & LIMITS ====================
    MAX_DOCUMENTS_PER_USER = 100
    MAX_QUERY_LENGTH = 1000
    RATE_LIMIT_REQUESTS = 100  # per hour
    
    # ==================== ENVIRONMENT CONFIG ====================
    @classmethod
    def get_environment_config(cls) -> Dict:
        """Get environment-specific configuration"""
        env = os.getenv("RAG_ENVIRONMENT", "development")
        
        configs = {
            "development": {
                "debug": True,
                "log_level": "DEBUG",
                "cache_enabled": False,
            },
            "production": {
                "debug": False,
                "log_level": "INFO",
                "cache_enabled": True,
            }
        }
        
        return configs.get(env, configs["development"])
    
    # ==================== VALIDATION METHODS ====================
    @classmethod
    def validate_file_type(cls, file_type: str) -> bool:
        """Validate if file type is supported"""
        return file_type in cls.SUPPORTED_PDF_TYPES
    
    @classmethod
    def validate_file_size(cls, file_size: int) -> bool:
        """Validate if file size is within limits"""
        max_size_bytes = cls.MAX_FILE_SIZE_MB * 1024 * 1024
        return file_size <= max_size_bytes
    
    @classmethod
    def get_supported_languages_list(cls) -> List[str]:
        """Get list of supported language names"""
        return list(cls.SUPPORTED_LANGUAGES.keys())
    
    @classmethod
    def get_ocr_language_codes(cls) -> str:
        """Get OCR language codes string"""
        return "+".join(cls.SUPPORTED_LANGUAGES.values())


# Global configuration instance
config = Config()

# Environment variables (can override above settings)
if os.getenv("EMBEDDING_MODEL"):
    config.EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL")

if os.getenv("LLM_MODEL"):
    config.LLM_MODEL_NAME = os.getenv("LLM_MODEL")

if os.getenv("VECTOR_DB_PATH"):
    config.VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")
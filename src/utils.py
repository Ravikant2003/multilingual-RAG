import logging
import os
import re
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
import json
import tempfile
from datetime import datetime
import streamlit as st
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceTimer:
    """Utility class for performance monitoring"""
    
    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting {self.operation_name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        logger.info(f"Completed {self.operation_name} in {elapsed:.2f} seconds")
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

def setup_logging(log_level: str = None):
    """Setup logging configuration"""
    if log_level is None:
        log_level = config.LOG_LEVEL
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(config.BASE_DIR, 'rag_system.log'))
        ]
    )

def validate_pdf_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate PDF file for processing
    """
    try:
        # Check file exists
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if not config.validate_file_size(file_size):
            return False, f"File size exceeds limit of {config.MAX_FILE_SIZE_MB}MB"
        
        # Check file extension
        if not file_path.lower().endswith('.pdf'):
            return False, "File is not a PDF"
        
        # Basic PDF validation by trying to open it
        import fitz
        try:
            doc = fitz.open(file_path)
            doc.close()
        except Exception as e:
            return False, f"Invalid PDF file: {str(e)}"
        
        return True, "Valid PDF file"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove unwanted characters but preserve multilingual characters
    text = re.sub(r'[^\w\s\u0600-\u06FF\u4e00-\u9fff\u0980-\u09FF.,!?;:()\-]', ' ', text)
    
    # Remove multiple consecutive spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def detect_language_simple(text: str) -> str:
    """
    Simple language detection based on character ranges
    """
    if not text:
        return "unknown"
    
    sample = text[:500]  # Use first 500 characters for detection
    
    # Character range checks
    urdu_chars = re.findall(r'[\u0600-\u06FF]', sample)
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', sample)
    bengali_chars = re.findall(r'[\u0980-\u09FF]', sample)
    hindi_chars = re.findall(r'[\u0900-\u097F]', sample)
    
    # Count characters for each language
    counts = {
        'urdu': len(urdu_chars),
        'chinese': len(chinese_chars),
        'bengali': len(bengali_chars),
        'hindi': len(hindi_chars),
        'english': len(re.findall(r'[a-zA-Z]', sample))
    }
    
    # Find language with most characteristic characters
    detected_lang = max(counts, key=counts.get)
    
    # Only return if we have significant character count
    if counts[detected_lang] > 10:
        return detected_lang
    
    return "unknown"

def calculate_md5(file_path: str) -> str:
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating MD5 for {file_path}: {str(e)}")
        return ""

def format_timestamp(timestamp: float = None) -> str:
    """Format timestamp to readable string"""
    if timestamp is None:
        timestamp = time.time()
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

def safe_json_dumps(data: Any) -> str:
    """Safely convert data to JSON string"""
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"JSON serialization error: {str(e)}")
        return str(data)

def safe_json_loads(json_str: str) -> Any:
    """Safely parse JSON string"""
    try:
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"JSON parsing error: {str(e)}")
        return {}

def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """Split list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except Exception as e:
        logger.error(f"Error getting file size: {str(e)}")
        return 0.0

def create_temp_file(content: bytes, suffix: str = ".pdf") -> str:
    """Create a temporary file with content"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(content)
            return temp_file.name
    except Exception as e:
        logger.error(f"Error creating temp file: {str(e)}")
        raise

def cleanup_temp_files(file_paths: List[str]):
    """Clean up temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Error cleaning up temp file {file_path}: {str(e)}")

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def estimate_processing_time(file_size_mb: float, document_type: str = "mixed") -> float:
    """
    Estimate processing time based on file size and type
    """
    base_time_per_mb = 2.0  # seconds per MB base
    
    if document_type == "scanned":
        base_time_per_mb *= 3  # OCR takes longer
    elif document_type == "digital":
        base_time_per_mb *= 1.5
    
    estimated_time = file_size_mb * base_time_per_mb
    return min(estimated_time, 300)  # Cap at 5 minutes

def validate_email(email: str) -> bool:
    """Basic email validation"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to remove unsafe characters"""
    # Remove directory traversal attempts
    filename = os.path.basename(filename)
    
    # Remove unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255 - len(ext)] + ext
    
    return filename

def get_system_info() -> Dict[str, Any]:
    """Get system information"""
    import platform
    import psutil
    
    try:
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "system_ram_gb": round(psutil.virtual_memory().total / (1024**3), 1),
            "available_ram_gb": round(psutil.virtual_memory().available / (1024**3), 1),
            "disk_usage": {
                "total_gb": round(psutil.disk_usage('/').total / (1024**3), 1),
                "free_gb": round(psutil.disk_usage('/').free / (1024**3), 1)
            }
        }
        
        # Add GPU info if available
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu"] = {
                    "available": True,
                    "name": torch.cuda.get_device_name(0),
                    "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
                }
            else:
                info["gpu"] = {"available": False}
        except ImportError:
            info["gpu"] = {"available": False, "error": "torch not available"}
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return {"error": str(e)}

def format_file_size(bytes_size: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"

def retry_operation(operation, max_retries: int = None, delay: float = None, 
                   exceptions: tuple = (Exception,)):
    """
    Retry decorator for operations that might fail
    """
    if max_retries is None:
        max_retries = config.MAX_RETRIES
    if delay is None:
        delay = config.RETRY_DELAY
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")
                        raise last_exception
            return None
        return wrapper
    
    if callable(operation):
        return decorator(operation)
    return decorator

@retry_operation
def download_file(url: str, save_path: str) -> bool:
    """Download file from URL with retry logic"""
    try:
        import requests
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        logger.info(f"Successfully downloaded file to {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download file from {url}: {str(e)}")
        return False

def create_progress_tracker(total_steps: int, description: str = "Processing"):
    """Create a simple progress tracker"""
    class ProgressTracker:
        def __init__(self, total_steps, description):
            self.total_steps = total_steps
            self.description = description
            self.current_step = 0
            self.start_time = time.time()
        
        def update(self, step_description: str = None):
            self.current_step += 1
            progress = (self.current_step / self.total_steps) * 100
            elapsed = time.time() - self.start_time
            
            if step_description:
                logger.info(f"{self.description}: {step_description} ({progress:.1f}%)")
            else:
                logger.info(f"{self.description}: Step {self.current_step}/{self.total_steps} ({progress:.1f}%)")
            
            return progress
        
        def complete(self):
            total_time = time.time() - self.start_time
            logger.info(f"{self.description} completed in {format_duration(total_time)}")
    
    return ProgressTracker(total_steps, description)

def setup_streamlit_session_state():
    """Initialize Streamlit session state with default values"""
    default_state = {
        'messages': [],
        'documents_processed': False,
        'uploaded_files': [],
        'processing_status': "",
        'chat_history': [],
        'vector_store_initialized': False,
        'rag_engine_initialized': False
    }
    
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

def validate_streamlit_file(uploaded_file) -> Tuple[bool, str]:
    """Validate Streamlit uploaded file"""
    try:
        # Check file type
        if uploaded_file.type != "application/pdf":
            return False, "Please upload a PDF file"
        
        # Check file size
        if len(uploaded_file.getvalue()) > config.MAX_FILE_SIZE_MB * 1024 * 1024:
            return False, f"File size exceeds {config.MAX_FILE_SIZE_MB}MB limit"
        
        # Check filename
        if not uploaded_file.name.lower().endswith('.pdf'):
            return False, "File must have .pdf extension"
        
        return True, "Valid file"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def get_language_name(language_code: str) -> str:
    """Get full language name from code"""
    language_names = {
        'urdu': 'Urdu',
        'chinese': 'Chinese',
        'bengali': 'Bengali',
        'english': 'English',
        'hindi': 'Hindi',
        'unknown': 'Unknown'
    }
    return language_names.get(language_code, language_code)

def create_language_flag(language: str) -> str:
    """Create emoji flag for language"""
    flags = {
        'urdu': '🇵🇰',
        'chinese': '🇨🇳',
        'bengali': '🇧🇩',
        'english': '🇺🇸',
        'hindi': '🇮🇳'
    }
    return flags.get(language, '🌐')

# Test functions
def test_utils():
    """Test utility functions"""
    try:
        # Test text cleaning
        test_text = "  Hello   World!  \n\nThis is a test.  "
        cleaned = clean_text(test_text)
        print(f"Text cleaning: '{test_text}' -> '{cleaned}'")
        
        # Test language detection
        languages_to_test = [
            "Hello world",  # English
            "你好世界",  # Chinese
            "হ্যালো বিশ্ব",  # Bengali
        ]
        
        for text in languages_to_test:
            lang = detect_language_simple(text)
            print(f"Language detection: '{text}' -> {lang}")
        
        # Test performance timer
        with PerformanceTimer("Test Operation") as timer:
            time.sleep(0.1)
        print(f"Operation took {timer.get_elapsed_time():.2f} seconds")
        
        # Test file size formatting
        sizes = [100, 1024, 1048576, 1073741824]
        for size in sizes:
            formatted = format_file_size(size)
            print(f"File size {size} -> {formatted}")
        
        return True
        
    except Exception as e:
        print(f"Utils test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_utils()
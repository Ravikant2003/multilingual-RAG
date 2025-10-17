import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import logging
from typing import List, Dict, Optional, Tuple
import langdetect
from langdetect import DetectorFactory
import tempfile
from config import config

# Set seed for consistent language detection
DetectorFactory.seed = 0

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles PDF processing including OCR for multilingual documents"""
    
    def __init__(self):
        self.supported_languages = config.SUPPORTED_LANGUAGES
        self.ocr_languages = config.get_ocr_language_codes()
        
    def process_documents(self, file_paths: List[str]) -> List[Dict]:
        """
        Process multiple PDF documents and extract text
        """
        all_documents = []
        
        for file_path in file_paths:
            try:
                logger.info(f"Processing document: {file_path}")
                documents = self.process_single_document(file_path)
                if documents:
                    all_documents.extend(documents)
                    logger.info(f"Successfully processed {len(documents)} chunks from {file_path}")
                else:
                    logger.warning(f"No text extracted from {file_path}")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
                
        return all_documents
    
    def process_single_document(self, file_path: str) -> List[Dict]:
        """
        Process a single PDF document
        """
        try:
            # Check if file exists and is valid
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Open PDF document
            doc = fitz.open(file_path)
            documents = []
            
            # Determine if PDF is digital or scanned
            is_digital = self._is_digital_pdf(doc)
            logger.info(f"PDF {file_path} is {'digital' if is_digital else 'scanned'}")
            
            # Process each page
            for page_num in range(len(doc)):
                try:
                    page_docs = self._process_page(doc, page_num, is_digital, file_path)
                    documents.extend(page_docs)
                except Exception as e:
                    logger.error(f"Error processing page {page_num} of {file_path}: {str(e)}")
                    continue
            
            doc.close()
            return documents
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return []
    
    def _is_digital_pdf(self, doc) -> bool:
        """
        Determine if PDF is digital (text-based) or scanned (image-based)
        """
        try:
            # Check first few pages for text content
            sample_pages = min(3, len(doc))
            text_length = 0
            
            for page_num in range(sample_pages):
                page = doc[page_num]
                text = page.get_text()
                text_length += len(text.strip())
            
            # If average text per page is substantial, it's likely digital
            avg_text_per_page = text_length / sample_pages
            return avg_text_per_page > 100  # Threshold for digital PDF
            
        except Exception:
            # If detection fails, assume scanned
            return False
    
    def _process_page(self, doc, page_num: int, is_digital: bool, file_path: str) -> List[Dict]:
        """
        Process a single page of PDF
        """
        page = doc[page_num]
        page_documents = []
        
        if is_digital:
            # Extract text directly from digital PDF
            text = page.get_text()
            if text.strip():
                chunks = self._chunk_text(text, file_path, page_num)
                page_documents.extend(chunks)
        else:
            # Use OCR for scanned PDF
            text = self._extract_text_with_ocr(page, file_path)
            if text.strip():
                chunks = self._chunk_text(text, file_path, page_num)
                page_documents.extend(chunks)
        
        return page_documents
    
    def _extract_text_with_ocr(self, page, file_path: str) -> str:
        """
        Extract text from scanned PDF page using OCR
        """
        try:
            # Get page as image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution for better OCR
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            # Configure OCR for multilingual support
            ocr_config = f'--oem 3 --psm 6 -l {self.ocr_languages}'
            
            # Perform OCR
            text = pytesseract.image_to_string(image, config=ocr_config)
            
            logger.info(f"OCR extracted {len(text)} characters from page")
            return text
            
        except Exception as e:
            logger.error(f"OCR failed for {file_path}: {str(e)}")
            return ""
    
    def _chunk_text(self, text: str, file_path: str, page_num: int) -> List[Dict]:
        """
        Split text into meaningful chunks with clean metadata
        """
        # Clean text
        text = self._clean_text(text)
        if not text.strip():
            return []
        
        # Extract just filename for cleaner metadata
        file_name = os.path.basename(file_path)
        
        # Detect language
        language = self._detect_language(text)
        
        # Split into chunks
        chunks = self._split_into_chunks(text)
        
        documents = []
        for chunk_num, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) > 30:  # Minimum chunk length
                document = {
                    'content': chunk_text,
                    'metadata': {
                        'source': file_name,  # Store just filename, not full path
                        'page': page_num + 1,
                        'chunk': chunk_num,
                        'language': language,
                        'file_name': file_name,
                        'chunk_size': len(chunk_text)
                    }
                }
                documents.append(document)
        
        return documents
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove unwanted characters but preserve multilingual characters
        # Keep letters, numbers, punctuation, and whitespace
        import re
        text = re.sub(r'[^\w\s\u0600-\u06FF\u4e00-\u9fff\u0980-\u09FF.,!?;:()\-]', ' ', text)
        
        return text.strip()
    
    def _detect_language(self, text: str) -> str:
        """
        Detect language of text
        """
        try:
            # Sample first 500 characters for detection (langdetect works better with shorter text)
            sample_text = text[:500]
            
            if not sample_text.strip():
                return "unknown"
            
            # Language mapping for langdetect
            lang_map = {
                'ur': 'urdu',
                'zh-cn': 'chinese', 'zh-tw': 'chinese',
                'bn': 'bengali',
                'en': 'english',
                'hi': 'hindi'
            }
            
            detected_lang = langdetect.detect(sample_text)
            return lang_map.get(detected_lang, 'unknown')
            
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}")
            return "unknown"
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks using multiple strategies
        """
        chunks = []
        
        # Strategy 1: Split by sentences (for languages that use similar punctuation)
        sentence_chunks = self._split_by_sentences(text)
        
        # Strategy 2: Fixed-length chunks with overlap
        fixed_chunks = self._split_fixed_length(text)
        
        # Use sentence chunks if they produce reasonable sizes, otherwise use fixed chunks
        if sentence_chunks and max(len(chunk) for chunk in sentence_chunks) <= config.MAX_CHUNK_LENGTH:
            chunks = sentence_chunks
        else:
            chunks = fixed_chunks
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """
        Split text by sentences (basic implementation for multiple languages)
        """
        import re
        
        # Multi-language sentence boundaries
        sentence_endings = r'[.!?।۔！？]+\s+'
        sentences = re.split(sentence_endings, text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence exceeds chunk size, save current chunk and start new one
            if len(current_chunk) + len(sentence) <= config.CHUNK_SIZE:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_fixed_length(self, text: str) -> List[str]:
        """
        Split text into fixed-length chunks with overlap
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = start + config.CHUNK_SIZE
            
            # If this isn't the last chunk, try to end at a sentence boundary
            if end < text_length:
                # Look for sentence endings in the overlap region
                overlap_region = text[max(start, end - config.CHUNK_OVERLAP):end]
                
                # Find last sentence ending in overlap region
                sentence_endings = ['.', '!', '?', '।', '۔', '！', '？']
                last_end_pos = -1
                
                for ending in sentence_endings:
                    pos = overlap_region.rfind(ending)
                    if pos > last_end_pos:
                        last_end_pos = pos
                
                if last_end_pos != -1:
                    end = start + (len(overlap_region) - last_end_pos - 1) + config.CHUNK_OVERLAP
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position for next chunk
            start = end - config.CHUNK_OVERLAP
            
            # Prevent infinite loop
            if start >= text_length:
                break
        
        return chunks
    
    def get_processing_stats(self, documents: List[Dict]) -> Dict:
        """
        Get statistics about processed documents
        """
        if not documents:
            return {}
        
        total_chunks = len(documents)
        total_chars = sum(len(doc['content']) for doc in documents)
        languages = {}
        
        for doc in documents:
            lang = doc['metadata'].get('language', 'unknown')
            languages[lang] = languages.get(lang, 0) + 1
        
        return {
            'total_chunks': total_chunks,
            'total_characters': total_chars,
            'average_chunk_size': total_chars / total_chunks if total_chunks > 0 else 0,
            'languages': languages,
            'sources': len(set(doc['metadata']['source'] for doc in documents))
        }


# Utility function for quick testing
def test_document_processor():
    """Test the document processor with a sample PDF"""
    processor = DocumentProcessor()
    
    # Create a temporary test directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # You would add a test PDF here for actual testing
        test_files = []  # Add paths to test PDFs
        
        if test_files:
            documents = processor.process_documents(test_files)
            stats = processor.get_processing_stats(documents)
            print("Processing Statistics:", stats)
            return documents
        else:
            print("No test files available")
            return []


if __name__ == "__main__":
    # Test the processor
    test_document_processor()
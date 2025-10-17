import google.generativeai as genai
import os
import logging
from typing import List, Dict
from config import config
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for Google Gemini API"""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")  # Get from environment
        self.model_name = config.GEMINI_MODEL
        self.model = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Gemini client"""
        try:
            if not self.api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables. Please create a .env file with your API key.")
            
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info("✅ Gemini client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {str(e)}")
            raise
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using Gemini"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=config.LLM_MAX_TOKENS,
                    temperature=config.LLM_TEMPERATURE
                )
            )
            
            if response.text:
                return response.text
            else:
                logger.warning("Gemini returned empty response")
                return "I couldn't generate a response. Please try again."
                
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            return f"Error: {str(e)}"
    
    def chat_with_context(self, context: str, question: str, chat_history: List[Dict] = None) -> str:
        """Chat with context using Gemini"""
        try:
            # Build conversation history
            history_text = ""
            if chat_history and config.ENABLE_CHAT_MEMORY:
                for msg in chat_history[-config.MAX_CHAT_HISTORY:]:
                    role = "User" if msg['role'] == 'user' else 'Assistant'
                    history_text += f"{role}: {msg['content']}\n"
            
            prompt = f"""You are a helpful multilingual assistant. Analyze the provided context and answer the question.

CONTEXT FROM DOCUMENT:
{context}

USER'S QUESTION:
{question}

CONVERSATION HISTORY:
{history_text}

INSTRUCTIONS:
1. FIRST, analyze if the context contains information related to the question
2. If there's ANY relevant information (even partial), provide the best answer you can
3. If the context is unclear or contains garbled text, try to interpret it
4. Only say "I cannot find this information" if there's absolutely NO relevant content
5. For general questions about the document, use the available context
6. Support multiple languages including Urdu and Bengali

ANALYSIS AND ANSWER:"""
            
            return self.generate_response(prompt)
            
        except Exception as e:
            logger.error(f"Chat with context failed: {str(e)}")
            return "I encountered an error while processing your question."
    
    def summarize_text(self, text: str, max_length: int = 300) -> str:
        """Summarize text using Gemini"""
        try:
            prompt = f"""Please summarize the following text in a clear and concise manner. Focus on the main points and key information.

TEXT:
{text[:4000]}  # Limit text length

SUMMARY:"""
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_length,
                    temperature=0.2
                )
            )
            
            return response.text if response.text else "Unable to generate summary."
            
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return "Unable to generate summary."
    
    def check_api_health(self) -> bool:
        """Check if Gemini API is working"""
        try:
            test_response = self.model.generate_content("Say 'Hello' in Urdu and Bengali")
            return test_response.text is not None
        except Exception as e:
            logger.error(f"API health check failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict:
        """Get information about the Gemini model"""
        return {
            "model": self.model_name,
            "provider": "Google Gemini",
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": config.LLM_TEMPERATURE,
            "multilingual": True,
            "supported_languages": ["Urdu", "Bengali", "English", "Hindi", "Chinese", "Arabic", "and 100+ more"]
        }


# Utility function to test Gemini connection
def test_gemini_connection():
    """Test the Gemini API connection"""
    try:
        client = GeminiClient()
        
        # Test basic functionality
        test_prompt = "Say 'Hello' in Urdu and Bengali"
        response = client.generate_response(test_prompt)
        
        print("✅ Gemini connection successful!")
        print(f"Test response: {response}")
        
        # Test API health
        if client.check_api_health():
            print("✅ API health check passed")
        else:
            print("❌ API health check failed")
        
        return True
        
    except Exception as e:
        print(f"❌ Gemini connection failed: {e}")
        return False


if __name__ == "__main__":
    print("🧪 Testing Gemini Client...")
    test_gemini_connection()
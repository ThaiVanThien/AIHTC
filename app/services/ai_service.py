import openai
import google.generativeai as genai
import logging
import os
from typing import Dict, List, Optional, Union, Any
from app.core.ai_config import AIProvider, ai_settings

logger = logging.getLogger(__name__)

# Debug của API key Gemini
gemini_api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
logger.info(f"GOOGLE_API_KEY from env: {bool(gemini_api_key)} (length: {len(gemini_api_key) if gemini_api_key else 0})")

fallback_key = os.environ.get("GEMINI_KEY", "").strip()
logger.info(f"GEMINI_KEY from env: {bool(fallback_key)} (length: {len(fallback_key) if fallback_key else 0})")

# Cấu hình toàn cục Gemini nếu có API key
if gemini_api_key:
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    logger.info("Configuring Gemini globally")
    genai.configure(api_key=gemini_api_key)
elif fallback_key:
    os.environ["GOOGLE_API_KEY"] = fallback_key
    logger.info("Configuring Gemini globally with fallback key")
    genai.configure(api_key=fallback_key)
else:
    logger.warning("No Gemini API key found in environment")

class AIService:
    """Service xử lý AI, hỗ trợ cả OpenAI và Gemini"""
    
    def __init__(self):
        self.openai_config = ai_settings.openai
        self.gemini_config = ai_settings.gemini
        
        # Cấu hình OpenAI
        openai.api_key = self.openai_config.api_key
        
        # Lưu API key Gemini
        self.gemini_api_key = self.gemini_config.api_key.strip() if self.gemini_config.api_key else ""
        if self.gemini_api_key:
            logger.info(f"Using Gemini API key from config (length: {len(self.gemini_api_key)})")
            
        # Log warning nếu không có API key
        if not self.openai_config.api_key:
            logger.warning("OpenAI API key not configured")
        if not self.gemini_api_key:
            logger.warning("Gemini API key not configured")
        
        logger.info("AI Service initialized")
    
    async def chat_with_openai(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Gửi yêu cầu đến OpenAI"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await openai.ChatCompletion.acreate(
                model=self.openai_config.model,
                messages=messages,
                temperature=self.openai_config.temperature,
                max_tokens=self.openai_config.max_tokens,
                top_p=self.openai_config.top_p,
                frequency_penalty=self.openai_config.frequency_penalty,
                presence_penalty=self.openai_config.presence_penalty,
                timeout=self.openai_config.timeout
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
    
    async def chat_with_gemini(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Gửi yêu cầu đến Gemini"""
        try:
            # Hardcode API key cho test
            api_key = "AIzaSyC_y1Ur707NadFqjuaDmHKAEHZvFrt6sm4"
            logger.info("Using hardcoded Gemini API key for testing")
            
            # Cấu hình lại cho chắc
            genai.configure(api_key=api_key)
            
            # Prepare model parameters
            generation_config = {
                "temperature": self.gemini_config.temperature,
                "max_output_tokens": self.gemini_config.max_output_tokens,
                "top_p": self.gemini_config.top_p,
                "top_k": self.gemini_config.top_k,
            }
            
            # Create model
            logger.info(f"Creating Gemini model: {self.gemini_config.model}")
            model = genai.GenerativeModel(
                model_name=self.gemini_config.model,
                generation_config=generation_config,
            )
            
            # Combine system prompt and user prompt if needed
            if system_prompt:
                prompt = f"{system_prompt}\n\n{prompt}"
            
            logger.info("Generating content with Gemini")
            # Use a synchronous method first to test
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            logger.error(f"API key available: {bool(self.gemini_api_key)}")
            # Fallback to OpenAI if Gemini fails
            logger.info("Falling back to OpenAI")
            return await self.chat_with_openai(prompt, system_prompt)
    
    async def generate_response(self, 
                         prompt: str, 
                         system_prompt: Optional[str] = None,
                         provider: Optional[AIProvider] = None) -> str:
        """
        Tạo phản hồi từ mô hình AI
        
        Args:
            prompt: Câu hỏi hoặc yêu cầu của người dùng
            system_prompt: Hướng dẫn cho AI (không bắt buộc)
            provider: Nhà cung cấp AI (mặc định là cấu hình toàn cục)
            
        Returns:
            Phản hồi từ mô hình AI
        """
        # Xác định provider để sử dụng
        provider = provider or ai_settings.default_provider
        
        try:
            if provider == AIProvider.OPENAI:
                return await self.chat_with_openai(prompt, system_prompt)
            elif provider == AIProvider.GEMINI:
                return await self.chat_with_gemini(prompt, system_prompt)
            else:
                raise ValueError(f"Provider không được hỗ trợ: {provider}")
        except Exception as e:
            logger.error(f"Lỗi khi tạo phản hồi AI: {str(e)}")
            raise
    
    async def process_chat(self, 
                    messages: List[Dict[str, str]], 
                    provider: Optional[AIProvider] = None) -> str:
        """
        Xử lý danh sách các tin nhắn chat
        
        Args:
            messages: Danh sách tin nhắn trong cuộc trò chuyện
            provider: Nhà cung cấp AI (mặc định là cấu hình toàn cục)
            
        Returns:
            Phản hồi từ mô hình AI
        """
        # Xử lý tin nhắn đầu vào thành prompt
        system_prompt = None
        last_message = None
        
        for message in messages:
            if message.get("role") == "system":
                system_prompt = message.get("content", "")
            elif message.get("role") == "user":
                last_message = message.get("content", "")
        
        if not last_message:
            raise ValueError("Không tìm thấy tin nhắn của người dùng")
            
        return await self.generate_response(last_message, system_prompt, provider)

# Khởi tạo service
ai_service = AIService() 
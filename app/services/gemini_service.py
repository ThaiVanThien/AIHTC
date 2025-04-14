import os
import logging
import requests
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from app.services.nlp_service import BaseNLPService
from app.core.config import settings

logger = logging.getLogger(__name__)

class GeminiService(BaseNLPService):
    """
    Dịch vụ NLP sử dụng Google Gemini API
    """
    def __init__(self):
        super().__init__()
        self.api_key = settings.GOOGLE_API_KEY
        self.model_name = "gemini-1.5-flash"  # Mô hình mặc định
        self.api_base_url = "https://generativelanguage.googleapis.com/v1beta"
        self.is_model_loaded = self.check_api_key()
        
    def check_api_key(self) -> bool:
        """
        Kiểm tra API key có hợp lệ không
        """
        if not self.api_key:
            logger.warning("Google API key không được cấu hình")
            return False
        return True
        
    def load_models(self) -> bool:
        """
        Kiểm tra kết nối với Google Gemini API
        """
        try:
            if not self.api_key:
                logger.warning("Google API key không được cấu hình")
                return False
                
            # Thử gọi API để kiểm tra kết nối với một request đơn giản
            url = f"{self.api_base_url}/models/{self.model_name}?key={self.api_key}"
            
            response = requests.get(url)
            
            if response.status_code == 200:
                self.is_model_loaded = True
                logger.info("Kết nối thành công với Google Gemini API")
                return True
            else:
                error_message = response.json().get("error", {}).get("message", "Unknown error")
                logger.error(f"Lỗi kết nối với Google Gemini API: {error_message}")
                self.is_model_loaded = False
                return False
                
        except Exception as e:
            logger.error(f"Lỗi khi kết nối Google Gemini API: {str(e)}")
            self.is_model_loaded = False
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """
        Lấy trạng thái hiện tại của dịch vụ Gemini
        """
        return {
            "service_type": "gemini",
            "model_loaded": self.is_model_loaded,
            "model_name": self.model_name,
            "api_configured": bool(self.api_key)
        }
        
    def answer_question(self, question: str, context: str = None) -> Dict[str, Any]:
        """
        Trả lời câu hỏi sử dụng Google Gemini API
        
        Args:
            question: Câu hỏi
            context: Ngữ cảnh (optional)
            
        Returns:
            Dict chứa câu trả lời và thông tin liên quan
        """
        if not self.is_model_loaded:
            logger.warning("Google Gemini API chưa được kết nối")
            if not self.load_models():
                return {
                    "answer": "Lỗi: API chưa được cấu hình đúng",
                    "success": False,
                    "error": "Google Gemini API chưa được kết nối"
                }
        
        try:
            url = f"{self.api_base_url}/models/{self.model_name}:generateContent?key={self.api_key}"
            
            # Xây dựng prompt với ngữ cảnh nếu có
            content = ""
            if context:
                content = f"Sử dụng thông tin sau đây để trả lời câu hỏi: {context}\n\nCâu hỏi: {question}"
            else:
                content = question
                
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": content
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 500,
                    "topP": 0.8,
                    "topK": 40
                }
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Kiểm tra nếu không có candidates
                if not result.get("candidates", []):
                    return {
                        "answer": "Gemini không thể tạo ra câu trả lời",
                        "success": False,
                        "error": "No candidates returned"
                    }
                    
                answer = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                
                return {
                    "answer": answer,
                    "model": self.model_name,
                    "success": True,
                    "confidence": None,  # Gemini không cung cấp điểm tin cậy
                    "context": context
                }
            else:
                error_message = response.json().get("error", {}).get("message", "Unknown error")
                logger.error(f"Lỗi từ Google Gemini API: {error_message}")
                return {
                    "answer": f"Lỗi từ Google Gemini API: {error_message}",
                    "success": False,
                    "error": error_message
                }
                
        except Exception as e:
            logger.error(f"Lỗi khi gọi Google Gemini API: {str(e)}")
            return {
                "answer": f"Lỗi kết nối: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def set_model(self, model_name: str) -> bool:
        """
        Thay đổi mô hình Gemini đang sử dụng
        
        Args:
            model_name: Tên mô hình Gemini muốn sử dụng (gemini-pro, gemini-ultra, ...)
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        valid_models = [
            "gemini-pro", 
            "gemini-ultra", 
            "gemini-pro-vision", 
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.0-pro"
        ]
        
        # Hỗ trợ tên mô hình linh hoạt
        normalized_model = model_name.lower()
        if "gemini" not in normalized_model:
            normalized_model = f"gemini-{normalized_model}"
            
        # Kiểm tra mô hình có hợp lệ không
        if normalized_model not in valid_models:
            logger.warning(f"Mô hình {model_name} không được hỗ trợ. Hỗ trợ các mô hình: {', '.join(valid_models)}")
            logger.warning(f"Đang sử dụng mô hình mặc định: {self.model_name}")
            return False
            
        # Kiểm tra kết nối đến mô hình
        try:
            url = f"{self.api_base_url}/models/{normalized_model}?key={self.api_key}"
            response = requests.get(url)
            
            if response.status_code != 200:
                error_message = response.json().get("error", {}).get("message", "Unknown error")
                logger.warning(f"Không thể kết nối đến mô hình {normalized_model}: {error_message}")
                logger.warning(f"Tiếp tục sử dụng mô hình hiện tại: {self.model_name}")
                return False
        except Exception as e:
            logger.warning(f"Lỗi khi kiểm tra mô hình {normalized_model}: {str(e)}")
            logger.warning(f"Tiếp tục sử dụng mô hình hiện tại: {self.model_name}")
            return False
            
        # Nếu mọi thứ OK, cập nhật mô hình
        self.model_name = normalized_model
        logger.info(f"Đã chuyển sang sử dụng mô hình {normalized_model}")
        return True
        
    async def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 500) -> Dict[str, Any]:
        """
        Xử lý chat nhiều lượt với Gemini API
        
        Args:
            messages: Danh sách tin nhắn trong cuộc trò chuyện
            temperature: Độ ngẫu nhiên (0.0-1.0)
            max_tokens: Số lượng token tối đa trong phản hồi
            
        Returns:
            Dict chứa câu trả lời và thông tin liên quan
        """
        if not self.is_model_loaded:
            logger.warning("Google Gemini API chưa được kết nối")
            if not self.load_models():
                return {
                    "answer": "Lỗi: API chưa được cấu hình đúng",
                    "success": False,
                    "error": "Google Gemini API chưa được kết nối"
                }
        
        try:
            url = f"{self.api_base_url}/models/{self.model_name}:generateContent?key={self.api_key}"
            
            # Chuẩn bị nội dung từ lịch sử chat
            # Gemini không hỗ trợ trực tiếp định dạng OpenAI nên cần chuyển đổi
            system_prompt = None
            conversation = []
            
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "system":
                    system_prompt = content
                else:
                    # Thêm role vào trước nội dung cho rõ ràng
                    formatted_content = f"{role.upper()}: {content}"
                    conversation.append(formatted_content)
            
            # Kết hợp system prompt (nếu có) và lịch sử cuộc trò chuyện
            prompt = ""
            if system_prompt:
                prompt = f"SYSTEM: {system_prompt}\n\n"
                
            prompt += "\n".join(conversation)
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "topP": 0.8,
                    "topK": 40
                }
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            logger.info(f"Sending chat request to Gemini API: {prompt[:100]}...")
            response = requests.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Kiểm tra nếu không có candidates
                if not result.get("candidates", []):
                    return {
                        "answer": "Gemini không thể tạo ra câu trả lời cho cuộc trò chuyện này",
                        "success": False,
                        "error": "No candidates returned"
                    }
                    
                answer = result["candidates"][0]["content"]["parts"][0]["text"].strip()
                
                return {
                    "answer": answer,
                    "model": self.model_name,
                    "success": True
                }
            else:
                error_message = response.json().get("error", {}).get("message", "Unknown error")
                logger.error(f"Lỗi từ Google Gemini API: {error_message}")
                return {
                    "answer": f"Lỗi từ Google Gemini API: {error_message}",
                    "success": False,
                    "error": error_message
                }
                
        except Exception as e:
            logger.error(f"Lỗi khi gọi Google Gemini API cho chat: {str(e)}")
            return {
                "answer": f"Lỗi kết nối: {str(e)}",
                "success": False,
                "error": str(e)
            }

# Khởi tạo dịch vụ Gemini
gemini_service = GeminiService() 
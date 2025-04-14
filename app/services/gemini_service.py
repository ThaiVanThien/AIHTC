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
        self.model_name = "gemini-pro"  # Mô hình mặc định
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
        valid_models = ["gemini-pro", "gemini-ultra", "gemini-pro-vision"]
        
        if model_name not in valid_models:
            logger.warning(f"Mô hình {model_name} không được hỗ trợ")
            return False
            
        self.model_name = model_name
        logger.info(f"Đã chuyển sang sử dụng mô hình {model_name}")
        return True

# Khởi tạo dịch vụ Gemini
gemini_service = GeminiService() 
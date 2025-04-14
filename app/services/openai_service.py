import os
import logging
import requests
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from app.services.nlp_service import BaseNLPService
from app.core.config import settings

logger = logging.getLogger(__name__)

class OpenAIService(BaseNLPService):
    """
    Dịch vụ NLP sử dụng OpenAI API
    """
    def __init__(self):
        super().__init__()
        self.api_key = settings.OPENAI_API_KEY
        self.model_name = "gpt-3.5-turbo"  # Mô hình mặc định
        self.api_base_url = "https://api.openai.com/v1"
        self.is_model_loaded = self.check_api_key()
        
    def check_api_key(self) -> bool:
        """
        Kiểm tra API key có hợp lệ không
        """
        if not self.api_key:
            logger.warning("OpenAI API key không được cấu hình")
            return False
        return True
        
    def load_models(self) -> bool:
        """
        Kiểm tra kết nối với OpenAI API
        """
        try:
            if not self.api_key:
                logger.warning("OpenAI API key không được cấu hình")
                return False
                
            # Thử gọi API để kiểm tra kết nối
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Chỉ lấy danh sách models để kiểm tra kết nối
            response = requests.get(
                f"{self.api_base_url}/models",
                headers=headers
            )
            
            if response.status_code == 200:
                self.is_model_loaded = True
                logger.info("Kết nối thành công với OpenAI API")
                return True
            else:
                error_message = response.json().get("error", {}).get("message", "Unknown error")
                logger.error(f"Lỗi kết nối với OpenAI API: {error_message}")
                self.is_model_loaded = False
                return False
                
        except Exception as e:
            logger.error(f"Lỗi khi kết nối OpenAI API: {str(e)}")
            self.is_model_loaded = False
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """
        Lấy trạng thái hiện tại của dịch vụ OpenAI
        """
        return {
            "service_type": "openai",
            "model_loaded": self.is_model_loaded,
            "model_name": self.model_name,
            "api_configured": bool(self.api_key)
        }
        
    def answer_question(self, question: str, context: str = None) -> Dict[str, Any]:
        """
        Trả lời câu hỏi sử dụng OpenAI API
        
        Args:
            question: Câu hỏi
            context: Ngữ cảnh (optional)
            
        Returns:
            Dict chứa câu trả lời và thông tin liên quan
        """
        if not self.is_model_loaded:
            logger.warning("OpenAI API chưa được kết nối")
            if not self.load_models():
                return {
                    "answer": "Lỗi: API chưa được cấu hình đúng",
                    "success": False,
                    "error": "OpenAI API chưa được kết nối"
                }
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            messages = []
            if context:
                # Thêm ngữ cảnh vào prompt nếu có
                messages.append({
                    "role": "system", 
                    "content": f"Sử dụng thông tin sau đây để trả lời câu hỏi của người dùng: {context}"
                })
            else:
                messages.append({
                    "role": "system", 
                    "content": "Bạn là trợ lý AI hữu ích. Trả lời câu hỏi một cách chính xác và ngắn gọn."
                })
                
            messages.append({"role": "user", "content": question})
            
            data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"].strip()
                
                return {
                    "answer": answer,
                    "model": self.model_name,
                    "success": True,
                    "confidence": None,  # OpenAI không cung cấp điểm tin cậy
                    "context": context
                }
            else:
                error_message = response.json().get("error", {}).get("message", "Unknown error")
                logger.error(f"Lỗi từ OpenAI API: {error_message}")
                return {
                    "answer": f"Lỗi từ OpenAI API: {error_message}",
                    "success": False,
                    "error": error_message
                }
                
        except Exception as e:
            logger.error(f"Lỗi khi gọi OpenAI API: {str(e)}")
            return {
                "answer": f"Lỗi kết nối: {str(e)}",
                "success": False,
                "error": str(e)
            }
    
    def set_model(self, model_name: str) -> bool:
        """
        Thay đổi mô hình OpenAI đang sử dụng
        
        Args:
            model_name: Tên mô hình OpenAI muốn sử dụng (gpt-3.5-turbo, gpt-4, ...)
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        valid_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo-16k"]
        
        if model_name not in valid_models:
            logger.warning(f"Mô hình {model_name} không được hỗ trợ")
            return False
            
        self.model_name = model_name
        logger.info(f"Đã chuyển sang sử dụng mô hình {model_name}")
        return True

# Khởi tạo dịch vụ OpenAI
openai_service = OpenAIService() 
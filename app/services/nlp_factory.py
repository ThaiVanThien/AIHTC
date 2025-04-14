import logging
from typing import Dict, Any, List, Optional

from app.services.nlp_service import BaseNLPService
from app.services.openai_service import openai_service
from app.services.gemini_service import gemini_service
from app.services.vimrc_service import vimrc_service

logger = logging.getLogger(__name__)

class NLPFactory:
    """
    Factory để quản lý các dịch vụ NLP khác nhau
    """
    def __init__(self):
        self.services = {
            "openai": openai_service,
            "gemini": gemini_service,
            "vimrc": vimrc_service
        }
        self.default_service = "vimrc"
        
    def get_service(self, service_type: str = None) -> BaseNLPService:
        """
        Lấy service tương ứng với loại được yêu cầu
        
        Args:
            service_type: Loại service (openai, gemini, vimrc)
            
        Returns:
            Dịch vụ NLP tương ứng
        """
        if not service_type:
            service_type = self.default_service
            
        service = self.services.get(service_type.lower())
        if not service:
            logger.warning(f"Không tìm thấy dịch vụ {service_type}, sử dụng {self.default_service}")
            return self.services[self.default_service]
            
        return service
        
    def get_all_services_status(self) -> Dict[str, Any]:
        """
        Lấy trạng thái của tất cả các service
        
        Returns:
            Dict chứa trạng thái của từng service
        """
        result = {}
        for name, service in self.services.items():
            result[name] = service.get_status()
            
        return result
    
    def answer_with_all_services(self, question: str, context: str = None) -> Dict[str, Any]:
        """
        Trả lời câu hỏi sử dụng tất cả các dịch vụ có sẵn
        
        Args:
            question: Câu hỏi
            context: Ngữ cảnh (nếu có)
            
        Returns:
            Dict chứa câu trả lời từ tất cả các dịch vụ
        """
        results = {}
        
        for name, service in self.services.items():
            try:
                # Chỉ gọi các service đã sẵn sàng
                if service.is_model_loaded:
                    results[name] = service.answer_question(question, context)
                else:
                    results[name] = {
                        "answer": f"Dịch vụ {name} chưa sẵn sàng",
                        "success": False,
                        "error": "Service not loaded"
                    }
            except Exception as e:
                logger.error(f"Lỗi khi gọi dịch vụ {name}: {str(e)}")
                results[name] = {
                    "answer": f"Lỗi khi gọi dịch vụ {name}: {str(e)}",
                    "success": False,
                    "error": str(e)
                }
                
        return results
    
    def set_default_service(self, service_type: str) -> bool:
        """
        Thay đổi dịch vụ mặc định
        
        Args:
            service_type: Loại dịch vụ (openai, gemini, vimrc)
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        if service_type.lower() in self.services:
            self.default_service = service_type.lower()
            logger.info(f"Đã chuyển dịch vụ mặc định sang {service_type}")
            return True
        else:
            logger.warning(f"Không tìm thấy dịch vụ {service_type}")
            return False

# Khởi tạo factory
nlp_factory = NLPFactory() 
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union

from app.services.vimrc_service import vimrc_service
from app.services.openai_service import openai_service
from app.services.gemini_service import gemini_service
from app.services.document_store import document_store
from app.core.ai_config import AIProvider, ai_settings

# Thiết lập logging
logger = logging.getLogger(__name__)

class IntentAnalyzer:
    """
    Phân tích ý định truy vấn kết hợp VI-MRC và LLM (Gemini/OpenAI)
    Sử dụng VI-MRC cho truy vấn cần ngữ cảnh cụ thể
    Sử dụng LLM cho việc phân loại và phân tích
    """
    
    def __init__(self, default_provider: str = None):
        self.default_provider = default_provider or ai_settings.default_provider.value
        logger.info(f"Khởi tạo IntentAnalyzer với provider mặc định: {self.default_provider}")
        
        # Cấu hình ngưỡng độ tin cậy
        self.confidence_threshold = 0.7
    
    async def analyze_with_ai_api(self, query: str, provider: str = None, model: str = None) -> Dict[str, Any]:
        """
        Phân tích ý định sử dụng API LLM (Gemini hoặc OpenAI)
        Dùng này khi cần phân loại chung hoặc không có ngữ cảnh cụ thể
        """
        start_time = time.time()
        
        # Xác định provider nếu không được cung cấp
        provider = provider or self.default_provider
        
        # Tạo prompt
        prompt = self._create_intent_prompt(query)
        
        try:
            # Xử lý theo provider
            if provider.lower() == "gemini":
                # Đặt model nếu có chỉ định
                if model:
                    gemini_service.set_model(model)
                
                # Gọi Gemini API
                response = await gemini_service.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,  # Nhiệt độ thấp để đảm bảo kết quả nhất quán 
                    max_tokens=500
                )
                
                # Xử lý kết quả JSON từ văn bản
                return self._parse_ai_json_response(response["answer"], provider="gemini")
                
            elif provider.lower() == "openai":
                # Đặt model nếu có chỉ định
                if model:
                    openai_service.set_model(model)
                
                # Gọi OpenAI API
                response = await openai_service.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=500
                )
                
                # Xử lý kết quả JSON từ văn bản
                return self._parse_ai_json_response(response["answer"], provider="openai")
                
            else:
                # Mặc định dùng Gemini
                logger.warning(f"Provider không hợp lệ: {provider}, sử dụng Gemini thay thế")
                
                if model:
                    gemini_service.set_model(model)
                
                response = await gemini_service.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=500
                )
                
                return self._parse_ai_json_response(response["answer"], provider="gemini")
                
        except Exception as e:
            logger.error(f"Lỗi khi phân tích ý định với {provider}: {str(e)}")
            
            # Thử dùng provider khác nếu thất bại
            try:
                fallback_provider = "openai" if provider.lower() == "gemini" else "gemini"
                logger.info(f"Sử dụng {fallback_provider} làm fallback cho phân tích ý định")
                
                if fallback_provider == "gemini":
                    response = await gemini_service.chat(
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=500
                    )
                    return self._parse_ai_json_response(response["answer"], provider="gemini")
                else:
                    response = await openai_service.chat(
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=500
                    )
                    return self._parse_ai_json_response(response["answer"], provider="openai")
                    
            except Exception as e2:
                logger.error(f"Lỗi khi sử dụng fallback {fallback_provider}: {str(e2)}")
                return {
                    "intent_type": "UNKNOWN",
                    "confidence_score": 0,
                    "error": f"Không thể phân tích ý định: {str(e)}"
                }
                
    async def analyze_with_vimrc(self, query: str) -> Dict[str, Any]:
        """
        Phân tích ý định sử dụng VI-MRC khi có ngữ cảnh phù hợp
        """
        start_time = time.time()
        
        try:
            # Tìm tài liệu liên quan
            relevant_docs = document_store.search(query, top_k=2)
            
            # Nếu không tìm thấy tài liệu bằng tìm kiếm ngữ nghĩa, thử tìm bằng từ khóa
            if not relevant_docs:
                keywords = document_store.extract_keywords(query)
                if keywords:
                    relevant_docs = document_store.keyword_search(keywords, top_k=2)
            
            # Nếu tìm thấy tài liệu liên quan, sử dụng VI-MRC
            if relevant_docs:
                context = relevant_docs[0].content
                
                # Tạo prompt phân tích ý định với context
                intent_prompt = f"""
                Dựa trên thông tin sau: 
                {context}
                
                Hãy phân tích ý định của câu hỏi: "{query}"
                
                Phân loại thành một trong hai loại:
                1. INFORMATION_RETRIEVAL: Người dùng muốn tìm kiếm dữ liệu cụ thể.
                2. QUESTION_ANSWERING: Người dùng đang đặt câu hỏi cần giải thích hoặc phân tích.
                
                Trả về JSON với định dạng:
                {{
                  "intent_type": "INFORMATION_RETRIEVAL hoặc QUESTION_ANSWERING",
                  "confidence_score": số từ 0.0 đến 1.0,
                  "parameters": {{
                    "search_keywords": ["từ khóa1", "từ khóa2"],
                    "entities": ["thực thể1", "thực thể2"],
                    "question_type": "loại câu hỏi (nếu có)"
                  }}
                }}
                """
                
                # Sử dụng VI-MRC để trả lời
                response = vimrc_service.answer_question(intent_prompt, context)
                
                if response["success"] and response["answer"].strip():
                    # Thử parse JSON từ câu trả lời
                    result = self._parse_ai_json_response(response["answer"], provider="vimrc")
                    
                    # Có kết quả hợp lệ với độ tin cậy tốt
                    if result.get("intent_type") and result.get("confidence_score", 0) > self.confidence_threshold:
                        logger.info(f"Phân tích ý định bằng VI-MRC thành công cho: {query}")
                        return result
        
        except Exception as e:
            logger.error(f"Lỗi khi phân tích ý định với VI-MRC: {str(e)}")
        
        # Nếu VI-MRC không thành công, trả về None để luồng chính chuyển sang LLM
        return None
    
    async def analyze(self, query: str, provider: str = None, model: str = None) -> Dict[str, Any]:
        """
        Phương thức chính để phân tích ý định
        Kết hợp cả vi-mrc và LLM để có kết quả tối ưu
        """
        logger.info(f"Bắt đầu phân tích ý định cho truy vấn: {query}")
        
        # Bước 1: Thử dùng VI-MRC nếu có
        if hasattr(vimrc_service, 'answer_question'):
            vimrc_result = await self.analyze_with_vimrc(query)
            
            # Nếu VI-MRC phân tích thành công, trả về kết quả
            if vimrc_result and vimrc_result.get("intent_type") != "UNKNOWN":
                return vimrc_result
        
        # Bước 2: Sử dụng LLM nếu VI-MRC không thành công hoặc không có VI-MRC
        return await self.analyze_with_ai_api(query, provider, model)
    
    def _create_intent_prompt(self, query: str) -> str:
        """
        Tạo prompt phân tích ý định dựa trên integration_plan
        """
        return f"""
        SYSTEM:
        Bạn là hệ thống phân tích truy vấn tiếng Việt. Nhiệm vụ của bạn là xác định loại câu, thời gian và hàm cần truy vấn.
        
        Phân loại truy vấn thành một trong hai loại sau:
        1. INFORMATION_RETRIEVAL (Truy Xuất Thông Tin): Người dùng muốn tìm kiếm dữ liệu cụ thể.
        2. QUESTION_ANSWERING (Hỏi Đáp): Người dùng đang đặt câu hỏi cần giải thích hoặc phân tích.
        
        Dựa trên phân tích của bạn, hãy trả về kết quả dưới dạng JSON với định dạng sau:
        {{
          "intent_type": "INFORMATION_RETRIEVAL hoặc QUESTION_ANSWERING",
          "confidence_score": số từ 0.0 đến 1.0,
          "time_info": {{
            "from_date": "dd/MM/yyyy", // Ngày bắt đầu nếu có
            "to_date": "dd/MM/yyyy",   // Ngày kết thúc nếu có
            "time_type": "specific/range/none", // Loại thời gian được chỉ định
            "quarter": "Q1/Q2/Q3/Q4", // Quý nếu có
            "year": "YYYY" // Năm của quý
          }},
          "function_info": {{
            "name": "tên hàm cần gọi",
            "parameters": {{
              // Các tham số cần thiết cho hàm
              "param1": "giá trị1",
              "param2": "giá trị2"
            }}
          }},
          "parameters": {{
            // Cho INFORMATION_RETRIEVAL
            "search_keywords": ["từ khóa1", "từ khóa2"],
            "filters": ["điều kiện1", "điều kiện2"],
            // Cho QUESTION_ANSWERING
            "question_type": "loại câu hỏi (what/why/how...)",
            "entities": ["thực thể1", "thực thể2"],
            "context": "ngữ cảnh nếu có"
          }}
        }}
        
        Hãy phân tích dựa trên các đặc điểm:
        - Cấu trúc ngữ pháp của câu
        - Từ nghi vấn hoặc từ chỉ lệnh
        - Các thực thể được đề cập
        - Yêu cầu ẩn hoặc rõ ràng của người dùng
        
        Dưới đây là một số ví dụ để hướng dẫn phân loại của bạn:

        Ví dụ 1:
        Truy vấn: "Tìm kiếm những laptop dưới 15 triệu có card đồ họa NVIDIA"
        Kết quả:
        {{
          "intent_type": "INFORMATION_RETRIEVAL",
          "confidence_score": 0.96,
          "time_info": {{
            "from_date": null,
            "to_date": null,
            "time_type": "none",
            "quarter": null,
            "year": null
          }},
          "parameters": {{
            "search_keywords": ["laptop", "card đồ họa", "NVIDIA"],
            "filters": ["giá < 15000000"],
            "question_type": null,
            "entities": ["laptop", "NVIDIA"],
            "context": null
          }}
        }}

        Ví dụ 2:
        Truy vấn: "Tại sao MacBook Pro lại có hiệu suất pin tốt hơn so với các laptop Windows?"
        Kết quả:
        {{
          "intent_type": "QUESTION_ANSWERING",
          "confidence_score": 0.92,
          "time_info": {{
            "from_date": null,
            "to_date": null,
            "time_type": "none",
            "quarter": null,
            "year": null
          }},
          "parameters": {{ 
            "search_keywords": null,
            "filters": null,
            "question_type": "why",
            "entities": ["MacBook Pro", "laptop Windows", "hiệu suất pin"],
            "context": "so sánh hiệu suất giữa hai loại thiết bị"
          }}
        }}
        
        CHÚ Ý QUAN TRỌNG: Vui lòng trả về JSON thuần túy, KHÔNG được thêm markdown hoặc code block. Chỉ trả về một đối tượng JSON hợp lệ.
        
        USER:
        {query}
        """
    
    def _parse_ai_json_response(self, response_text: str, provider: str) -> Dict[str, Any]:
        """
        Xử lý và parse JSON từ kết quả text của LLM hoặc VI-MRC
        Xử lý các trường hợp code block, format lỗi...
        """
        logger.debug(f"Phân tích kết quả từ {provider}: {response_text[:100]}...")
        
        try:
            # Làm sạch phản hồi
            # Xử lý trường hợp json trong code block
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                if json_end == -1:  # Không tìm thấy dấu đóng
                    cleaned_text = response_text[json_start:].strip()
                else:
                    cleaned_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                if json_end == -1:
                    cleaned_text = response_text[json_start:].strip()
                else:
                    cleaned_text = response_text[json_start:json_end].strip()
            else:
                cleaned_text = response_text.strip()
            
            # Parse JSON
            try:
                result = json.loads(cleaned_text)
                # Đảm bảo có các trường bắt buộc
                if "intent_type" not in result:
                    result["intent_type"] = "UNKNOWN"
                if "confidence_score" not in result:
                    result["confidence_score"] = 0.5
                if "parameters" not in result:
                    result["parameters"] = {}
                    
                logger.info(f"Phân tích JSON thành công từ {provider}")
                return result
            except json.JSONDecodeError:
                # Thử tìm { đầu tiên và } cuối cùng để trích xuất JSON
                start_index = cleaned_text.find('{')
                end_index = cleaned_text.rfind('}') + 1
                if start_index != -1 and end_index > start_index:
                    try:
                        json_text = cleaned_text[start_index:end_index]
                        result = json.loads(json_text)
                        logger.info(f"Phân tích JSON thành công sau khi cắt tỉa từ {provider}")
                        return result
                    except:
                        pass
                        
                logger.error(f"Không thể parse JSON từ {provider}: {cleaned_text[:100]}...")
                return {
                    "intent_type": "UNKNOWN",
                    "confidence_score": 0,
                    "error": "Không thể parse JSON từ phản hồi"
                }
        except Exception as e:
            logger.error(f"Lỗi khi xử lý phản hồi từ {provider}: {str(e)}")
            return {
                "intent_type": "UNKNOWN", 
                "confidence_score": 0,
                "error": str(e)
            }

# Khởi tạo analyzer sẵn cho import từ module khác
intent_analyzer = IntentAnalyzer() 
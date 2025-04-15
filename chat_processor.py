import logging
from typing import Dict, List, Any, Optional

from app.services.vimrc_service import vimrc_service
from app.services.openai_service import openai_service
from app.services.gemini_service import gemini_service
from app.services.document_store import document_store
from intent_analyzer import intent_analyzer

logger = logging.getLogger(__name__)

class ChatProcessor:
    """
    Xử lý và phản hồi các truy vấn chat sử dụng intent_analyzer để phân tích ý định
    và chọn phương pháp xử lý phù hợp (VI-MRC hoặc LLM).
    """
    
    def __init__(self):
        """Khởi tạo ChatProcessor"""
        self.intent_analyzer = intent_analyzer  # Sử dụng instance được khởi tạo sẵn
        logger.info("ChatProcessor đã được khởi tạo")
        
    async def process_message(self, message: str, provider: str = None, model: str = None, temperature: float = 0.7) -> Dict[str, Any]:
        """
        Xử lý tin nhắn từ người dùng:
        1. Phân tích ý định sử dụng intent_analyzer
        2. Dựa vào ý định để quyết định cách xử lý (VI-MRC hoặc LLM)
        3. Trả về kết quả phù hợp
        """
        logger.info(f"Đang xử lý tin nhắn: '{message[:50]}...' với provider={provider}, model={model}")
        
        # Bước 1: Phân tích ý định truy vấn
        intent_result = await self.intent_analyzer.analyze(message, provider=provider, model=model)
        
        if not intent_result or intent_result.get("intent_type") == "UNKNOWN":
            logger.warning("Không thể xác định ý định của truy vấn")
            # Mặc định sử dụng LLM khi không xác định được ý định
            return await self._process_with_llm(message, "UNKNOWN", {}, provider, model, temperature)
        
        intent_type = intent_result.get("intent_type")
        parameters = intent_result.get("parameters", {})
        confidence = intent_result.get("confidence_score", 0)
        
        logger.info(f"Phân tích ý định: {intent_type} với độ tin cậy {confidence}")
        
        # Bước 2: Xử lý dựa trên ý định
        if intent_type == "QUESTION_ANSWERING":
            # Truy vấn hỏi đáp - thử dùng VI-MRC trước
            return await self._process_question(message, parameters, provider, model, temperature)
        elif intent_type == "INFORMATION_RETRIEVAL":
            # Truy vấn tìm kiếm thông tin
            return await self._process_search(message, parameters, provider, model, temperature)
        else:
            # Ý định không rõ ràng hoặc không được hỗ trợ
            return await self._process_with_llm(message, intent_type, parameters, provider, model, temperature)
    
    async def _process_question(self, question: str, parameters: Dict, provider: str, model: str, temperature: float) -> Dict[str, Any]:
        """
        Xử lý câu hỏi (QUESTION_ANSWERING)
        Sử dụng VI-MRC nếu có tài liệu liên quan, nếu không dùng LLM
        """
        # Bước 1: Tìm tài liệu liên quan
        entities = parameters.get("entities", [])
        search_terms = []
        
        # Sử dụng entities từ phân tích ý định để tìm kiếm
        if entities:
            search_terms = entities
        else:
            # Hoặc trích xuất từ khóa từ câu hỏi
            keywords = document_store.extract_keywords(question)
            if keywords:
                search_terms = keywords
        
        # Tìm tài liệu liên quan
        relevant_docs = []
        if search_terms:
            relevant_docs = document_store.search(" ".join(search_terms), top_k=2)
            if not relevant_docs:
                relevant_docs = document_store.keyword_search(search_terms, top_k=2)
        
        # Nếu không tìm thấy bằng entities, thử tìm bằng toàn bộ câu hỏi
        if not relevant_docs:
            relevant_docs = document_store.search(question, top_k=2)
        
        # Bước 2: Sử dụng VI-MRC nếu có tài liệu liên quan
        if relevant_docs:
            context = relevant_docs[0].content
            
            # Thử dùng VI-MRC
            try:
                vimrc_response = vimrc_service.answer_question(question, context)
                
                if vimrc_response["success"] and vimrc_response["answer"].strip():
                    # VI-MRC trả lời thành công
                    return {
                        "answer": vimrc_response["answer"],
                        "source": "vimrc",
                        "provider": "vimrc",
                        "model": vimrc_service.model_name,
                        "confidence": vimrc_response.get("confidence", 0.8),
                        "has_context": True
                    }
            except Exception as e:
                logger.error(f"Lỗi khi sử dụng VI-MRC: {str(e)}")
        
        # Bước 3: Sử dụng LLM nếu VI-MRC không thành công
        return await self._process_with_llm(question, "QUESTION_ANSWERING", parameters, provider, model, temperature, 
                                           context=relevant_docs[0].content if relevant_docs else None)
    
    async def _process_search(self, query: str, parameters: Dict, provider: str, model: str, temperature: float) -> Dict[str, Any]:
        """
        Xử lý truy vấn tìm kiếm (INFORMATION_RETRIEVAL)
        """
        # Lấy các từ khóa và bộ lọc từ parameters
        search_keywords = parameters.get("search_keywords", [])
        filters = parameters.get("filters", [])
        
        # Nếu không có từ khóa, trích xuất từ câu truy vấn
        if not search_keywords:
            search_keywords = document_store.extract_keywords(query)
        
        # Tìm kiếm dữ liệu
        search_results = document_store.search(" ".join(search_keywords), top_k=5)
        
        # Nếu có kết quả, tạo phản hồi
        if search_results:
            # Chuẩn bị context dựa trên kết quả tìm kiếm
            context = "\n\n".join([doc.content for doc in search_results[:3]])
            
            # Sử dụng LLM để tổng hợp kết quả và tạo phản hồi
            return await self._process_with_llm(
                query, 
                "INFORMATION_RETRIEVAL", 
                parameters, 
                provider, 
                model, 
                temperature,
                context=context
            )
        else:
            # Không tìm thấy kết quả, dùng LLM trả lời trực tiếp
            return await self._process_with_llm(query, "INFORMATION_RETRIEVAL", parameters, provider, model, temperature)
    
    async def _process_with_llm(self, message: str, intent_type: str, parameters: Dict, 
                               provider: str, model: str, temperature: float, context: str = None) -> Dict[str, Any]:
        """
        Xử lý tin nhắn bằng LLM (Gemini hoặc OpenAI)
        Thêm cấu trúc và ngữ cảnh phù hợp với loại ý định
        """
        # Xác định provider (mặc định dùng Gemini nếu không có)
        provider = provider or "gemini"
        
        # Chuẩn bị prompt với context nếu có
        prompt = message
        if context:
            # Tùy chỉnh prompt dựa trên loại ý định
            if intent_type == "QUESTION_ANSWERING":
                prompt = f"""
                Dựa trên thông tin sau:
                
                {context}
                
                Hãy trả lời câu hỏi sau: {message}
                """
            elif intent_type == "INFORMATION_RETRIEVAL":
                prompt = f"""
                Dựa trên thông tin tìm kiếm sau:
                
                {context}
                
                Hãy trả lời hoặc tóm tắt thông tin phù hợp với yêu cầu: {message}
                """
            else:
                prompt = f"""
                Thông tin tham khảo:
                
                {context}
                
                Yêu cầu: {message}
                """
        
        # Gọi API dựa trên provider
        try:
            if provider.lower() == "gemini":
                # Đặt model nếu có chỉ định
                if model:
                    gemini_service.set_model(model)
                    
                response = await gemini_service.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1000
                )
                
                return {
                    "answer": response["answer"],
                    "source": "llm",
                    "provider": "gemini",
                    "model": gemini_service.model_name,
                    "has_context": bool(context)
                }
                
            else:  # OpenAI
                # Đặt model nếu có chỉ định
                if model:
                    openai_service.set_model(model)
                    
                response = await openai_service.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=1000
                )
                
                return {
                    "answer": response["answer"],
                    "source": "llm",
                    "provider": "openai",
                    "model": openai_service.model_name,
                    "has_context": bool(context)
                }
                
        except Exception as e:
            logger.error(f"Lỗi khi sử dụng {provider}: {str(e)}")
            
            # Thử dùng provider khác nếu thất bại
            try:
                fallback_provider = "openai" if provider.lower() == "gemini" else "gemini"
                logger.info(f"Sử dụng {fallback_provider} làm fallback")
                
                if fallback_provider == "gemini":
                    response = await gemini_service.chat(
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=1000
                    )
                    
                    return {
                        "answer": response["answer"],
                        "source": "llm",
                        "provider": "gemini",
                        "model": gemini_service.model_name,
                        "has_context": bool(context)
                    }
                else:
                    response = await openai_service.chat(
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=1000
                    )
                    
                    return {
                        "answer": response["answer"],
                        "source": "llm",
                        "provider": "openai",
                        "model": openai_service.model_name,
                        "has_context": bool(context)
                    }
            except Exception as e2:
                logger.error(f"Cả hai provider đều thất bại: {str(e2)}")
                return {
                    "answer": f"Không thể xử lý yêu cầu do lỗi hệ thống: {str(e)}",
                    "source": "error",
                    "provider": provider,
                    "model": "none",
                    "has_context": bool(context)
                }

# Khởi tạo processor sẵn sàng để import
chat_processor = ChatProcessor() 
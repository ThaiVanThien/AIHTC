from fastapi import APIRouter, HTTPException, BackgroundTasks, Form, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from pathlib import Path
import shutil

from app.core.config import settings
from app.services.nlp_factory import nlp_factory

router = APIRouter(
    responses={404: {"description": "Not found"}},
)

@router.get("/status", response_model=Dict[str, Any], summary="Trạng thái tất cả dịch vụ NLP")
async def get_nlp_status():
    """
    Lấy trạng thái hiện tại của tất cả các dịch vụ NLP:
    - ViMRC: Mô hình trả lời câu hỏi tiếng Việt
    - OpenAI: Kết nối đến API của OpenAI
    - Gemini: Kết nối đến API của Google
    """
    return nlp_factory.get_all_services_status()

@router.post("/compare", response_model=Dict[str, Any], summary="So sánh câu trả lời từ tất cả dịch vụ")
async def compare_answers(question: str, context: str):
    """
    Trả lời câu hỏi sử dụng tất cả các dịch vụ NLP có sẵn (ViMRC, OpenAI, Gemini) và trả về kết quả so sánh
    
    - **question**: Câu hỏi cần trả lời
    - **context**: Ngữ cảnh/đoạn văn chứa câu trả lời
    
    Kết quả sẽ bao gồm câu trả lời từ mỗi dịch vụ để dễ dàng so sánh.
    """
    try:
        results = nlp_factory.answer_with_all_services(question, context)
        return {
            "question": question,
            "context": context,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý câu hỏi: {str(e)}")

@router.post("/answer", response_model=Dict[str, Any], summary="Trả lời câu hỏi (điều hướng)")
async def answer_question(question: str, context: str, service: str = None):
    """
    Điều hướng yêu cầu trả lời câu hỏi đến dịch vụ thích hợp
    
    - **question**: Câu hỏi cần trả lời
    - **context**: Ngữ cảnh/đoạn văn chứa câu trả lời
    - **service**: Loại dịch vụ NLP (openai, gemini, vimrc). Để trống để sử dụng dịch vụ mặc định.
    
    Endpoint này sẽ điều hướng yêu cầu đến dịch vụ thích hợp. Bạn có thể sử dụng trực tiếp:
    - /vimrc/answer: Cho mô hình vi-mrc
    - /cloud/answer?provider=openai: Cho OpenAI
    - /cloud/answer?provider=gemini: Cho Gemini
    """
    try:
        service_obj = nlp_factory.get_service(service)
        result = service_obj.answer_question(question, context)
        
        # Thêm thông tin về loại service đã sử dụng
        result["service_used"] = service or nlp_factory.default_service
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý câu hỏi: {str(e)}") 
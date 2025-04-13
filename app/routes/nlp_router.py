from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from app.services.nlp_service import nlp_service

# Thiết lập logging
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/nlp",
    tags=["NLP Services"],
    responses={404: {"description": "Not found"}},
)

# Model cho API trả lời câu hỏi (QA)
class QuestionRequest(BaseModel):
    question: str
    context: str

class QuestionResponse(BaseModel):
    answer: str
    confidence: float
    start: Optional[int] = None  
    end: Optional[int] = None
    error: Optional[str] = None

# API endpoint cho việc trả lời câu hỏi
@router.post("/qa", response_model=QuestionResponse, summary="Trả lời câu hỏi dựa trên ngữ cảnh")
async def answer_question(request: QuestionRequest):
    """
    Trả lời câu hỏi dựa trên ngữ cảnh được cung cấp.
    
    - **question**: Câu hỏi bằng tiếng Việt
    - **context**: Đoạn văn bản chứa thông tin để trả lời câu hỏi
    
    Trả về:
    - **answer**: Câu trả lời được trích xuất từ ngữ cảnh
    - **confidence**: Độ tin cậy của câu trả lời (0-1)
    - **start**: Vị trí bắt đầu của câu trả lời trong ngữ cảnh (nếu có)
    - **end**: Vị trí kết thúc của câu trả lời trong ngữ cảnh (nếu có)
    """
    logger.info(f"Nhận yêu cầu QA: Question='{request.question}', Context length={len(request.context)}")
    
    # Đảm bảo mô hình đã được tải
    if not nlp_service.is_loaded:
        if not nlp_service.load_models():
            raise HTTPException(status_code=503, detail="Không thể tải mô hình QA")
    
    # Thực hiện phân tích và trả lời câu hỏi
    result = nlp_service.answer_question(request.question, request.context)
    
    # Kiểm tra lỗi
    if "error" in result and result["error"]:
        logger.error(f"Lỗi khi xử lý câu hỏi: {result['error']}")
        raise HTTPException(status_code=500, detail=result["error"])
    
    logger.info(f"Câu trả lời: '{result['answer']}' (confidence: {result['confidence']:.2f})")
    return result

# API endpoint để kiểm tra trạng thái của service
@router.get("/status", summary="Kiểm tra trạng thái của NLP service")
async def get_status():
    """
    Kiểm tra trạng thái của NLP service và các mô hình đã tải.
    """
    return nlp_service.get_status() 
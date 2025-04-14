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

@router.post("/set-service", response_model=Dict[str, Any], summary="Đặt dịch vụ NLP mặc định")
async def set_default_service(service: str):
    """
    Đặt dịch vụ NLP mặc định được sử dụng khi không có chỉ định cụ thể
    
    - **service**: Loại dịch vụ (openai, gemini, vimrc)
    
    Dịch vụ mặc định sẽ được sử dụng khi gọi API /answer mà không chỉ định tham số service.
    """
    if nlp_factory.set_default_service(service):
        return {"success": True, "message": f"Đã đặt {service} làm dịch vụ mặc định"}
    else:
        raise HTTPException(status_code=400, detail=f"Dịch vụ không hợp lệ: {service}")

@router.post("/clear-cache", response_model=Dict[str, Any], summary="Xóa cache Hugging Face")
async def clear_huggingface_cache():
    """
    Xóa tất cả cache của Hugging Face để giải phóng không gian đĩa
    
    Cache Hugging Face thường chiếm nhiều dung lượng sau khi tải các mô hình lớn.
    Sử dụng endpoint này để xóa cache và giải phóng dung lượng đĩa.
    """
    try:
        # Xóa cache của Hugging Face từ đường dẫn cấu hình
        cache_dir = Path(settings.HUGGINGFACE_CACHE_DIR).expanduser().resolve()
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            return {
                "success": True,
                "message": f"Đã xóa cache Hugging Face tại {cache_dir}"
            }
        else:
            return {
                "success": True,
                "message": f"Thư mục cache {cache_dir} không tồn tại, không cần xóa"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa cache: {str(e)}")

@router.get("/config", response_model=Dict[str, Any], summary="Xem cấu hình NLP")
def get_nlp_config():
    """
    Xem cấu hình hiện tại của các dịch vụ NLP, bao gồm:
    - Đường dẫn thư mục chứa mô hình
    - Đường dẫn thư mục dữ liệu huấn luyện
    - Tên mô hình mặc định
    - Dịch vụ mặc định
    """
    return {
        "models_dir": str(settings.MODELS_DIR),
        "training_data_dir": str(settings.TRAINING_DATA_DIR),
        "default_model_name": settings.DEFAULT_MODEL_NAME,
        "default_service": nlp_factory.default_service,
        "huggingface_cache_dir": str(Path(settings.HUGGINGFACE_CACHE_DIR).expanduser())
    }

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
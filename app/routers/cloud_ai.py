from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
from enum import Enum

from app.services.nlp_factory import nlp_factory
from app.services.openai_service import openai_service
from app.services.gemini_service import gemini_service
from app.core.config import settings

router = APIRouter(
    prefix="/cloud",
    tags=["cloud-ai"],
    responses={404: {"description": "Not found"}},
)

class AIProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"

class Message(BaseModel):
    role: str = Field(..., description="Vai trò của người gửi tin nhắn (user, system, assistant)", example="user")
    content: str = Field(..., description="Nội dung tin nhắn", example="Chào bạn, hôm nay thời tiết thế nào?")

class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., description="Danh sách các tin nhắn trong cuộc trò chuyện")
    provider: Optional[AIProvider] = Field(None, description="Nhà cung cấp AI để sử dụng", example="openai")
    model: Optional[str] = Field(None, description="Tên mô hình cụ thể", example="gpt-3.5-turbo")
    temperature: Optional[float] = Field(None, description="Nhiệt độ ảnh hưởng đến tính ngẫu nhiên", example=0.7)
    max_tokens: Optional[int] = Field(None, description="Số lượng token tối đa trong phản hồi", example=500)

class ChatResponse(BaseModel):
    content: str = Field(..., description="Nội dung phản hồi từ AI")
    provider: AIProvider = Field(..., description="Nhà cung cấp AI đã sử dụng")
    model: str = Field(..., description="Mô hình đã sử dụng")

@router.get("/status", response_model=Dict[str, Any], summary="Trạng thái dịch vụ Cloud AI")
async def get_cloud_status():
    """
    Lấy trạng thái hiện tại của các dịch vụ Cloud AI:
    - OpenAI: Kết nối đến API của OpenAI
    - Gemini: Kết nối đến API của Google
    """
    return {
        "openai": openai_service.get_status(),
        "gemini": gemini_service.get_status()
    }

@router.post("/answer", response_model=Dict[str, Any], summary="Trả lời câu hỏi với Cloud AI")
async def answer_question(question: str, context: str = None, provider: AIProvider = AIProvider.OPENAI):
    """
    Trả lời câu hỏi sử dụng dịch vụ Cloud AI (OpenAI hoặc Gemini)
    
    - **question**: Câu hỏi cần trả lời
    - **context**: Ngữ cảnh/đoạn văn chứa câu trả lời (tùy chọn)
    - **provider**: Nhà cung cấp AI (openai hoặc gemini)
    
    Dịch vụ Cloud AI sẽ sử dụng mô hình ngôn ngữ lớn để trả lời câu hỏi, có thể cần hoặc không cần ngữ cảnh.
    """
    try:
        if provider == AIProvider.OPENAI:
            service = openai_service
        else:
            service = gemini_service
            
        result = service.answer_question(question, context)
        
        # Thêm thông tin về provider đã sử dụng
        result["provider"] = provider
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý câu hỏi: {str(e)}")

@router.post("/compare", response_model=Dict[str, Any], summary="So sánh câu trả lời từ OpenAI và Gemini")
async def compare_answers(question: str, context: str = None):
    """
    Trả lời câu hỏi sử dụng cả OpenAI và Gemini và trả về kết quả so sánh
    
    - **question**: Câu hỏi cần trả lời
    - **context**: Ngữ cảnh/đoạn văn chứa câu trả lời (tùy chọn)
    
    Kết quả sẽ bao gồm câu trả lời từ cả hai dịch vụ để dễ dàng so sánh.
    """
    try:
        openai_result = openai_service.answer_question(question, context)
        gemini_result = gemini_service.answer_question(question, context)
        
        return {
            "question": question,
            "context": context,
            "results": {
                "openai": openai_result,
                "gemini": gemini_result
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý câu hỏi: {str(e)}")

@router.post("/chat", response_model=ChatResponse, summary="Chat với mô hình AI")
async def chat(request: ChatRequest):
    """
    Gửi tin nhắn chat đến mô hình AI và nhận phản hồi
    
    - **messages**: Danh sách các tin nhắn
    - **provider**: (Tùy chọn) Nhà cung cấp AI để sử dụng (openai/gemini)
    - **model**: (Tùy chọn) Tên mô hình cụ thể
    - **temperature**: (Tùy chọn) Nhiệt độ ảnh hưởng đến tính ngẫu nhiên
    - **max_tokens**: (Tùy chọn) Số lượng token tối đa trong phản hồi
    
    Endpoint này sử dụng API chat của OpenAI hoặc Gemini để xử lý cuộc trò chuyện nhiều lượt.
    """
    try:
        provider = request.provider or AIProvider.OPENAI
        model = request.model
        
        # Chuyển đổi messages thành định dạng phù hợp
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Xử lý chat dựa vào provider
        if provider == AIProvider.OPENAI:
            service = openai_service
            
            # Đặt model nếu có chỉ định
            if model:
                openai_service.set_model(model)
                
            # Gọi API OpenAI
            response = await openai_service.chat(
                messages=messages,
                temperature=request.temperature or 0.7,
                max_tokens=request.max_tokens or 500
            )
            
            return ChatResponse(
                content=response["answer"],
                provider=AIProvider.OPENAI,
                model=openai_service.model_name
            )
        else:
            service = gemini_service
            
            # Đặt model nếu có chỉ định
            if model:
                gemini_service.set_model(model)
                
            # Gọi API Gemini
            response = await gemini_service.chat(
                messages=messages,
                temperature=request.temperature or 0.7,
                max_tokens=request.max_tokens or 500
            )
            
            return ChatResponse(
                content=response["answer"],
                provider=AIProvider.GEMINI,
                model=gemini_service.model_name
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý yêu cầu chat: {str(e)}")

@router.get("/models", response_model=Dict[str, Any], summary="Danh sách mô hình AI có sẵn")
async def get_available_models():
    """
    Lấy danh sách tất cả mô hình AI có sẵn từ OpenAI và Gemini
    
    Thông tin trả về bao gồm:
    - Danh sách mô hình OpenAI
    - Danh sách mô hình Gemini
    """
    openai_models = [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-3.5-turbo-16k"
    ]
    
    gemini_models = [
        "gemini-pro",
        "gemini-ultra",
        "gemini-pro-vision"
    ]
    
    return {
        "openai": {
            "models": openai_models,
            "default": openai_service.model_name
        },
        "gemini": {
            "models": gemini_models,
            "default": gemini_service.model_name
        }
    }

@router.post("/set-model", response_model=Dict[str, Any], summary="Đặt mô hình mặc định")
async def set_default_model(provider: AIProvider, model: str):
    """
    Đặt mô hình mặc định cho dịch vụ AI
    
    - **provider**: Nhà cung cấp AI (openai hoặc gemini)
    - **model**: Tên mô hình cần đặt làm mặc định
    
    Mô hình mặc định sẽ được sử dụng khi không có chỉ định cụ thể.
    """
    try:
        if provider == AIProvider.OPENAI:
            success = openai_service.set_model(model)
            if success:
                return {
                    "success": True,
                    "message": f"Đã đặt {model} làm mô hình mặc định cho OpenAI"
                }
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Mô hình {model} không được hỗ trợ bởi OpenAI"
                )
        else:
            success = gemini_service.set_model(model)
            if success:
                return {
                    "success": True,
                    "message": f"Đã đặt {model} làm mô hình mặc định cho Gemini"
                }
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Mô hình {model} không được hỗ trợ bởi Gemini"
                )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Lỗi khi đặt mô hình mặc định: {str(e)}"
        ) 
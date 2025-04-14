from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
from app.core.ai_config import AIProvider

class Message(BaseModel):
    role: str = Field(..., description="Vai trò của người gửi tin nhắn (user/system/assistant)")
    content: str = Field(..., description="Nội dung tin nhắn")
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "Xin chào, cho tôi biết về lịch sử thành phố Huế"
            }
        }

class ChatRequest(BaseModel):
    messages: List[Message] = Field(
        ..., 
        description="Danh sách tin nhắn trong cuộc trò chuyện",
        example=[
            {"role": "system", "content": "Bạn là trợ lý AI thông minh, hữu ích, lịch sự và thân thiện. Hãy trả lời chính xác, ngắn gọn và hữu ích."},
            {"role": "user", "content": "Xin chào, cho tôi biết về lịch sử thành phố Huế"}
        ]
    )
    provider: Optional[AIProvider] = Field(
        AIProvider.GEMINI, 
        description="Nhà cung cấp AI (openai/gemini)"
    )
    model: Optional[str] = Field(
        "gemini-1.5-flash", 
        description="Tên mô hình cụ thể muốn sử dụng",
        example="gemini-1.5-flash"
    )
    temperature: Optional[float] = Field(
        0.7, 
        description="Nhiệt độ (0.0-1.0) ảnh hưởng đến tính ngẫu nhiên",
        example=0.7
    )
    max_tokens: Optional[int] = Field(
        500, 
        description="Số lượng token tối đa trong phản hồi",
        example=500
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "system", "content": "Bạn là trợ lý AI thông minh, hữu ích, lịch sự và thân thiện. Hãy trả lời chính xác, ngắn gọn và hữu ích."},
                    {"role": "user", "content": "Xin chào, cho tôi biết về lịch sử thành phố Huế"}
                ],
                "provider": "gemini",
                "model": "gemini-1.5-flash",
                "temperature": 0.7,
                "max_tokens": 500
            }
        }

class ChatResponse(BaseModel):
    content: str = Field(..., description="Nội dung phản hồi từ AI")
    provider: str = Field(..., description="Nhà cung cấp AI đã sử dụng")
    model: str = Field(..., description="Mô hình AI đã sử dụng")
    
class ModelInfo(BaseModel):
    name: str = Field(..., description="Tên mô hình")
    provider: AIProvider = Field(..., description="Nhà cung cấp AI")
    description: str = Field(..., description="Mô tả về mô hình")

class ModelsResponse(BaseModel):
    models: List[ModelInfo] = Field(..., description="Danh sách các mô hình AI có sẵn")

class SmartQARequest(BaseModel):
    question: str = Field(..., description="Câu hỏi cần trả lời")
    provider: Optional[AIProvider] = Field(
        AIProvider.GEMINI, 
        description="Nhà cung cấp AI sử dụng khi cần LLM (openai/gemini)"
    )
    model: Optional[str] = Field(
        "gemini-1.5-flash", 
        description="Tên mô hình cụ thể muốn sử dụng khi câu hỏi được chuyển sang LLM",
        example="gemini-1.5-flash"
    )
    temperature: Optional[float] = Field(
        0.7, 
        description="Nhiệt độ (0.0-1.0) ảnh hưởng đến tính ngẫu nhiên của LLM",
        example=0.7
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Doanh thu quý 1 năm 2023 là bao nhiêu?",
                "provider": "gemini",
                "model": "gemini-1.5-flash",
                "temperature": 0.7
            }
        }

class SmartQAResponse(BaseModel):
    answer: str = Field(..., description="Câu trả lời")
    source: str = Field(..., description="Nguồn cung cấp câu trả lời (vimrc/llm)")
    provider: str = Field(..., description="Nhà cung cấp AI đã sử dụng")
    model: str = Field(..., description="Mô hình AI đã sử dụng") 
    confidence: Optional[float] = Field(None, description="Độ tin cậy của câu trả lời (nếu có)")
    has_context: bool = Field(False, description="Có tìm thấy ngữ cảnh phù hợp hay không")
    processing_time: float = Field(..., description="Thời gian xử lý (giây)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Doanh thu quý 1 năm 2023 là 500 tỷ đồng",
                "source": "vimrc",
                "provider": "vimrc", 
                "model": "vi-mrc-large",
                "confidence": 0.92,
                "has_context": True,
                "processing_time": 0.75
            }
        } 
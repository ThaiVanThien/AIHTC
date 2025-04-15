from typing import Dict, List, Optional, Any
import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ai_service import AIService
from intent_router_test import IntentRouter

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="E-commerce Chatbot API",
    description="API xử lý chat và truy vấn sản phẩm",
    version="1.0.0"
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo dịch vụ AI
ai_service = AIService()
intent_router = IntentRouter()

# Định nghĩa các model
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = "gemini"  # mặc định sử dụng Gemini

class ChatResponse(BaseModel):
    response: str
    intent: Optional[str] = None
    data: Optional[dict] = None

@app.get("/")
async def root():
    return {"message": "Dịch vụ chat đang hoạt động"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Lấy tin nhắn cuối cùng từ người dùng
        user_message = request.messages[-1].content
        
        # Phân tích intent
        intent_result = intent_router.analyze_query(user_message)
        
        # Xử lý dựa trên intent
        if intent_result.get("is_product_search"):
            # Xử lý tìm kiếm sản phẩm
            products = intent_router.find_products(user_message)
            return ChatResponse(
                response="Đây là kết quả tìm kiếm sản phẩm của bạn",
                intent="product_search",
                data={"products": products}
            )
        elif intent_result.get("is_product_price"):
            # Xử lý tìm giá sản phẩm
            price_info = intent_router.get_product_price(user_message)
            return ChatResponse(
                response=f"Giá sản phẩm là {price_info['price']}",
                intent="product_price",
                data=price_info
            )
        elif intent_result.get("is_product_comparison"):
            # Xử lý so sánh sản phẩm
            comparison = intent_router.compare_products(user_message)
            return ChatResponse(
                response="Đây là kết quả so sánh sản phẩm",
                intent="product_comparison",
                data=comparison
            )
        else:
            # Trả lời thông thường bằng AI
            response = ai_service.process_chat(request.messages, request.model)
            return ChatResponse(
                response=response,
                intent="general_chat"
            )
            
    except Exception as e:
        logger.error(f"Lỗi xử lý chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "Chat AI API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) 
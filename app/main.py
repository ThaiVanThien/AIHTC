from fastapi import FastAPI, APIRouter, Request, status, HTTPException, Depends
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import logging
import os
import sys
import signal
import subprocess
import socket
import time
import uuid
from datetime import datetime
from typing import Callable, Dict, Union, List, Optional

# Load .env file
from dotenv import load_dotenv
load_dotenv(verbose=True)

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.routers import nlp, vimrc, cloud_ai, chat, product, chat_cho, product_category, product_routes
from app.middleware.rate_limiter import RateLimitMiddleware, rate_limiter

# Thiết lập logging
logger = setup_logging()
logger.info("Khởi động ứng dụng...")

# Lưu lại port và PID
APP_PORT = 8002
APP_PID = os.getpid()

# Rate limiting settings
RATE_LIMIT_DURATION = 60  # seconds
RATE_LIMIT_REQUESTS = 100  # requests per duration

# Xử lý exception toàn cục
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return await http_exception_handler(request, exc)
    
    # Ghi log lỗi nhưng không hiển thị chi tiết cho client
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "message": "Đã xảy ra lỗi nội bộ. Vui lòng thử lại sau.",
            "timestamp": datetime.now().isoformat()
        }
    )

# Tạo ứng dụng FastAPI với cấu hình nâng cao
app = FastAPI(
    title="Vietnamese NLP API",
    description="""
    <h2>API Xử lý Ngôn ngữ Tự nhiên Tiếng Việt</h2>
    <p>API hỗ trợ nhiều dịch vụ xử lý ngôn ngữ tự nhiên tiếng Việt</p>
    <ul>
        <li>✅ ViMRC - Trả lời câu hỏi tiếng Việt</li>
        <li>✅ OpenAI - Tích hợp API của OpenAI</li>
        <li>✅ Gemini - Tích hợp API của Google</li>
        <li>✅ So sánh kết quả từ nhiều mô hình</li>
    </ul>
    """,
    version="2.0.0",
    swagger_ui_parameters={
        "docExpansion": "list",
        "defaultModelsExpandDepth": 2,
        "deepLinking": True,
        "displayRequestDuration": True,
        "filter": True,
        "syntaxHighlight.theme": "monokai",
        "persistAuthorization": True,
        "tryItOutEnabled": True
    },
    # Thông tin bổ sung
    openapi_tags=[
        {
            "name": "nlp",
            "description": "Các tính năng xử lý ngôn ngữ tự nhiên chung"
        },
        {
            "name": "vi-mrc",
            "description": "Mô hình trả lời câu hỏi tiếng Việt (Vi-MRC)"
        },
        {
            "name": "cloud-ai",
            "description": "Dịch vụ AI trên cloud (OpenAI, Gemini)"
        },
        {
            "name": "admin",
            "description": "Các tính năng quản trị hệ thống"
        }
    ],
    # Cấu hình liên hệ và giấy phép
    contact={
        "name": "Hợp tác xã công nghệ thông tin Huế (HuetechCoop)",
        "url": "https://huetechcoop.com/",
        "email": "support@example.com",
    },
    license_info={
        "name": "HueTechCoop",
        "url": "https://huetechcoop.com/",
    },
)

# Thêm handler xử lý exception toàn cục
app.add_exception_handler(Exception, global_exception_handler)

# Thêm middleware
app.add_middleware(RateLimitMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký các router trực tiếp mà không dùng tiền tố /api/v1
# Đầu tiên import tất cả để tránh lỗi import cycle
from app.routers import nlp, vimrc, cloud_ai, chat, product, chat_cho, product_category, product_routes

# Đăng ký các router theo thứ tự
#app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(chat_cho.router, prefix="/chat", tags=["chat"])
app.include_router(vimrc.router, prefix="/vimrc", tags=["vi-mrc"])
app.include_router(cloud_ai.router, prefix="/cloud", tags=["cloud-ai"])
app.include_router(nlp.router, tags=["nlp"])
app.include_router(product.router, prefix="/api", tags=["product"])
app.include_router(product_category.router, prefix="/product-category", tags=["product-category"])
app.include_router(product_routes.router, tags=["products"])

# Thêm cấu hình để phục vụ các static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")


def is_port_in_use(port: int) -> bool:
    """Kiểm tra xem cổng có đang được sử dụng hay không"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


# Endpoint thông tin hệ thống và kiểm tra sức khỏe
@app.get("/system", tags=["admin"], summary="Thông tin và trạng thái hệ thống", 
        description="Trả về thông tin về hệ thống và trạng thái của server.")
def system_status():
    """
    Lấy thông tin về hệ thống và trạng thái hoạt động.
    
    Returns:
        dict: Thông tin về hệ thống và trạng thái hoạt động.
    """
    return {
        "status": "healthy",
        "version": "1.1.0",
        "server_time": datetime.now().isoformat(),
        "platform": sys.platform,
        "pid": APP_PID,
        "port": APP_PORT,
        "python_version": sys.version,
    }


@app.get("/", tags=["root"], summary="Trang chủ", response_class=HTMLResponse, 
        description="Chuyển hướng đến giao diện chat.")
async def root(request: Request):
    """
    Trang chủ chuyển hướng đến giao diện chat.
    """
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/chat")


@app.on_event("startup")
async def startup_event():
    """
    Khởi tạo ứng dụng và tạo các thư mục cần thiết khi khởi động
    """
    import os
    from app.core.config import settings
    
    # Tạo thư mục templates nếu chưa tồn tại
    os.makedirs(os.path.join(settings.BASE_DIR, "app", "templates"), exist_ok=True)
    
    # Log thông tin khởi động
    logger.info(f"Starting up application version: {settings.APP_VERSION}")
    
    # Ghi nhận cổng đang được sử dụng
    if is_port_in_use(APP_PORT) and APP_PORT != 0:
        logger.warning(f"Cổng {APP_PORT} đang được sử dụng bởi một tiến trình khác")
    
    logger.info("Ứng dụng đã khởi động thành công")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Sự kiện khi ứng dụng tắt
    """
    logger.info("Ứng dụng đang tắt...")
    logger.info("Tất cả kết nối đã được đóng")
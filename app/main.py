from fastapi import FastAPI, APIRouter, Request, status, HTTPException, Depends
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse
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
from app.routers import nlp, vimrc, cloud_ai, chat
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

# Middleware đo hiệu suất
class PerformanceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Thêm request_id vào header request để theo dõi
        request.state.request_id = request_id
        
        # Log thông tin request
        logger.info(f"Request {request_id} started: {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            
            # Đo thời gian xử lý
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            
            # Log kết quả
            logger.info(f"Request {request_id} completed: {response.status_code} in {process_time:.4f}s")
            
            return response
        except Exception as e:
            # Log lỗi
            process_time = time.time() - start_time
            logger.error(f"Request {request_id} failed: {str(e)} in {process_time:.4f}s")
            raise

# Xử lý exception toàn cục
async def global_exception_handler(request: Request, exc: Exception):
    error_id = str(uuid.uuid4())
    logger.error(f"Unhandled error {error_id}: {str(exc)}", exc_info=True)
    
    if isinstance(exc, HTTPException):
        return await http_exception_handler(request, exc)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_id": error_id,
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
app.add_middleware(PerformanceMiddleware)
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

# Tạo router API gốc
api_router = APIRouter(prefix="/api/v1")
# Đăng ký router con
api_router.include_router(nlp.router, tags=["nlp"])
api_router.include_router(vimrc.router, tags=["vi-mrc"])
api_router.include_router(cloud_ai.router, tags=["cloud-ai"])
api_router.include_router(chat.router, tags=["chat"])

# Đăng ký router API gốc vào ứng dụng
app.include_router(api_router)

# Thêm cấu hình để phục vụ các static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")


def is_port_in_use(port: int) -> bool:
    """Kiểm tra xem cổng có đang được sử dụng hay không"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def free_port(port: int) -> None:
    """Giải phóng cổng đang bị chiếm"""
    try:
        if sys.platform == 'win32':
            # Tìm PID của tiến trình đang giữ cổng (Windows)
            result = subprocess.run(
                f'netstat -ano | findstr :{port}', 
                shell=True, 
                capture_output=True, 
                text=True
            )
            if result.stdout:
                # Tìm PID từ output của netstat
                for line in result.stdout.strip().split('\n'):
                    if 'LISTENING' in line or 'ESTABLISHED' in line:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            pid = parts[-1]
                            if pid and pid.isdigit() and int(pid) != APP_PID:
                                # Kết thúc tiến trình với PID này (nếu không phải tiến trình hiện tại)
                                subprocess.run(f'taskkill /PID {pid} /F', shell=True)
                                logger.info(f"Đã giải phóng cổng {port} từ PID {pid}")
                
                # Nếu cổng vẫn bị chiếm khi đang ở trạng thái TIME_WAIT, cần chờ một chút
                if 'TIME_WAIT' in result.stdout:
                    logger.info(f"Cổng {port} đang ở trạng thái TIME_WAIT, đợi hệ thống giải phóng...")
                    # Đợi tối đa 5 giây để TIME_WAIT hết hạn
                    for _ in range(5):
                        time.sleep(1)
                        # Kiểm tra lại xem cổng đã được giải phóng chưa
                        if not is_port_in_use(port):
                            logger.info(f"Cổng {port} đã được giải phóng")
                            break
        else:
            # Linux/MacOS
            result = subprocess.run(
                f"lsof -i :{port} -t", 
                shell=True, 
                capture_output=True, 
                text=True
            )
            if result.stdout:
                for pid in result.stdout.strip().split('\n'):
                    if pid and pid.isdigit() and int(pid) != APP_PID:
                        # Gửi tín hiệu SIGTERM đến tiến trình
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                            logger.info(f"Đã giải phóng cổng {port} từ PID {pid}")
                        except ProcessLookupError:
                            pass
    except Exception as e:
        logger.error(f"Lỗi khi giải phóng cổng {port}: {str(e)}")


# Endpoint lấy thông tin hệ thống
@app.get("/system/info", tags=["admin"], summary="Thông tin hệ thống", 
        description="Trả về thông tin về hệ thống và trạng thái của server.")
def system_info():
    """
    Lấy thông tin về hệ thống.
    
    Returns:
        dict: Thông tin về hệ thống bao gồm phiên bản, thời gian hoạt động, v.v.
    """
    return {
        "version": "1.1.0",
        "server_time": datetime.now().isoformat(),
        "platform": sys.platform,
        "pid": APP_PID,
        "port": APP_PORT,
        "python_version": sys.version,
    }


# Endpoint health check
@app.get("/health", tags=["admin"], summary="Kiểm tra sức khỏe", 
        description="Kiểm tra xem API có hoạt động bình thường không.")
def health_check():
    """
    Kiểm tra sức khỏe của API.
    
    Returns:
        dict: Thông tin về trạng thái của API.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/", tags=["root"], summary="Trang chủ API", 
        description="Hiển thị thông tin chào mừng và hướng dẫn sử dụng API.")
def root(request: Request):
    """
    Trang chủ của API.
    
    Returns:
        dict: Thông tin chào mừng và liên kết đến tài liệu.
    """
    return {
        "message": "Chào mừng đến với AI Hệ thống Chat Demo",
        "status": "active",
        "version": "1.1.0",
        "request_id": getattr(request.state, "request_id", "unknown"),
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        },
        "endpoints": {
            "items": "/api/v1/items",
            "health": "/health",
            "system_info": "/system/info"
        }
    }


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
    
    # Kiểm tra và giải phóng cổng nếu nó đang bị chiếm
    if is_port_in_use(APP_PORT):
        logger.warning(f"Cổng {APP_PORT} đang bị chiếm, đang thử giải phóng...")
        free_port(APP_PORT)
    
    logger.info("Ứng dụng đã khởi động thành công")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Sự kiện khi ứng dụng tắt
    """
    logger.info("Ứng dụng đang tắt...")
    
    # Thử giải phóng cổng
    logger.info(f"Đang giải phóng cổng {APP_PORT}...")
    free_port(APP_PORT)
    
    logger.info("Tất cả kết nối đã được đóng")
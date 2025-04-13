from fastapi import FastAPI, APIRouter, Request, status, HTTPException, Depends
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

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.routers import items
from app.routers import nlp

# Thiết lập logging
logger = setup_logging()
logger.info("Khởi động ứng dụng...")

# Lưu lại port và PID
APP_PORT = 8002
APP_PID = os.getpid()

# Rate limiting settings
RATE_LIMIT_DURATION = 60  # seconds
RATE_LIMIT_REQUESTS = 100  # requests per duration

# Class cho rate limiting
class RateLimiter:
    def __init__(self, duration: int = RATE_LIMIT_DURATION, requests: int = RATE_LIMIT_REQUESTS):
        self.requests_per_window = requests
        self.window_duration = duration
        self.clients: Dict[str, List[float]] = {}
        
    def is_rate_limited(self, client_id: str) -> bool:
        now = time.time()
        if client_id not in self.clients:
            self.clients[client_id] = [now]
            return False
            
        # Lọc ra các request trong khoảng thời gian hiện tại
        client_requests = [req_time for req_time in self.clients[client_id] 
                          if now - req_time < self.window_duration]
        
        # Cập nhật danh sách request của client
        self.clients[client_id] = client_requests
        
        # Thêm request hiện tại
        self.clients[client_id].append(now)
        
        # Kiểm tra giới hạn
        return len(self.clients[client_id]) > self.requests_per_window

# Tạo instance của rate limiter
rate_limiter = RateLimiter()

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

# Middleware giới hạn tốc độ truy cập
class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Xác định client dựa trên IP hoặc header authorization
        client_id = request.client.host
        if "authorization" in request.headers:
            client_id = request.headers["authorization"]
            
        # Kiểm tra giới hạn
        if rate_limiter.is_rate_limited(client_id):
            logger.warning(f"Rate limit exceeded for client {client_id}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Quá nhiều yêu cầu. Vui lòng thử lại sau."}
            )
            
        return await call_next(request)

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
    title="AI Chat API",
    description="""
    <h2>API Chat thông minh</h2>
    <p>API hỗ trợ chat và xử lý ngôn ngữ tự nhiên tiếng Việt</p>
    <ul>
        <li>✅ Chat thông minh</li>
        <li>✅ Xử lý tiếng Việt</li>
        <li>✅ Trả lời câu hỏi</li>
    </ul>
    """,
    version="1.0.0",
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
            "name": "root",
            "description": "Endpoint chính của API"
        },
        {
            "name": "items",
            "description": "Quản lý các items trong hệ thống"
        },
        {
            "name": "nlp",
            "description": "Các tính năng xử lý ngôn ngữ tự nhiên"
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
api_router.include_router(items.router, prefix="/items", tags=["items"])
api_router.include_router(nlp.router, prefix="/nlp", tags=["nlp"])

# Đăng ký router API gốc vào ứng dụng
app.include_router(api_router)


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
    Sự kiện khi ứng dụng khởi động
    """
    logger.info(f"Ứng dụng đang chạy với PID: {APP_PID}")
    
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
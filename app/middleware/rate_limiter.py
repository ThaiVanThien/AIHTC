from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, status
from starlette.responses import JSONResponse
import time
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, duration: int = 60, requests: int = 100):
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
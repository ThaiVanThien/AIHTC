import os
from dotenv import load_dotenv
import logging

# Tải biến môi trường từ file .env
load_dotenv()

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Kiểm tra API keys
if not OPENAI_API_KEY:
    logger.warning("Thiếu OPENAI_API_KEY")

if not GEMINI_API_KEY:
    logger.warning("Thiếu GEMINI_API_KEY")

# Cấu hình ứng dụng
APP_NAME = "Hệ thống Trợ lý Thương mại Điện tử"
DEFAULT_AI_PROVIDER = "openai"  # hoặc "gemini"

# Cấu hình model
OPENAI_MODEL = "gpt-3.5-turbo"
GEMINI_MODEL = "gemini-pro"

# Cấu hình ứng dụng FastAPI
ALLOWED_ORIGINS = ["*"]  # Trong môi trường sản xuất, hãy chỉ định chính xác các origins 
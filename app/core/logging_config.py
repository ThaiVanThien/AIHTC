import logging
import os
import sys
from logging.handlers import RotatingFileHandler

def setup_logging(log_level=logging.INFO):
    """
    Thiết lập logging cho ứng dụng
    """
    # Tạo thư mục logs nếu chưa tồn tại
    os.makedirs("logs", exist_ok=True)
    
    # Định dạng log
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler cho console - hỗ trợ tiếng Việt
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Handler cho file với encoding utf-8
    file_handler = RotatingFileHandler(
        "logs/app.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Cấu hình root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Giảm bớt log từ các thư viện khác
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    # Đặt log level cho transformer
    # Chú ý: các thư viện như transformers in ra rất nhiều log
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
    
    return root_logger 
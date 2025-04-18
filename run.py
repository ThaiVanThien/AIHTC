import uvicorn
import socket
import os
import sys
import logging
from pathlib import Path

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Cổng mặc định
APP_PORT = 8002

def is_port_in_use(port: int) -> bool:
    """Kiểm tra xem cổng có đang được sử dụng hay không"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def find_available_port(start_port: int, max_attempts: int = 10) -> int:
    """Tìm cổng khả dụng từ start_port"""
    port = start_port
    for _ in range(max_attempts):
        if not is_port_in_use(port):
            return port
        port += 1
    # Nếu không tìm thấy cổng nào khả dụng, trả về 0 để hệ thống tự động chọn
    return 0

def setup_environment():
    """Thiết lập môi trường"""
    # Lấy đường dẫn thư mục gốc của dự án
    current_file = Path(__file__).resolve()
    project_root = current_file.parent
    
    # Thêm thư mục gốc vào PYTHONPATH
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        logger.info(f"Đã thêm {project_root} vào PYTHONPATH")
    
    # In thông tin
    logger.info(f"Project root: {project_root}")

if __name__ == "__main__":
    # Thiết lập môi trường
    setup_environment()
    
    # Kiểm tra cổng đã sử dụng chưa
    if is_port_in_use(APP_PORT):
        logger.warning(f"Cổng {APP_PORT} đã được sử dụng. Tìm cổng khác...")
        new_port = find_available_port(APP_PORT + 1)
        if new_port == 0:
            logger.error("Không tìm thấy cổng khả dụng. Hãy đóng các ứng dụng đang chạy và thử lại.")
            sys.exit(1)
        
        logger.info(f"Đã tìm thấy cổng khả dụng: {new_port}")
        APP_PORT = new_port
    
    try:
        logger.info(f"Khởi động ứng dụng trên cổng {APP_PORT}...")
        uvicorn.run("app.main:app", host="127.0.0.1", port=APP_PORT, reload=False, workers=1)
    except Exception as e:
        logger.error(f"Lỗi khi khởi động ứng dụng: {e}")
        sys.exit(1) 
import os
import sys
import logging
import socket
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

def setup_debug_environment():
    """
    Cấu hình môi trường để debug ứng dụng FastAPI
    """
    # Lấy đường dẫn thư mục gốc của dự án
    current_file = Path(__file__).resolve()
    project_root = current_file.parent
    
    # Thêm thư mục gốc vào PYTHONPATH
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        logger.info(f"Đã thêm {project_root} vào PYTHONPATH")
    
    # Kiểm tra cấu trúc thư mục
    app_dir = project_root / "app"
    if not app_dir.exists():
        logger.warning(f"Không tìm thấy thư mục app tại {app_dir}")
    else:
        logger.info(f"Tìm thấy thư mục app tại {app_dir}")
    
    # In thông tin debug
    logger.info(f"Project root: {project_root}")
    logger.info(f"Python path: {sys.path}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    return project_root

def run_debug():
    """
    Chạy ứng dụng FastAPI trong chế độ debug
    """
    global APP_PORT
    project_root = setup_debug_environment()
    
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
        # Import uvicorn sau khi cấu hình PYTHONPATH
        import uvicorn
        
        try:
            from app.main import app
            
            # Hiển thị thông tin về các router đã đăng ký
            logger.info("Các router đã đăng ký:")
            for route in app.routes:
                logger.info(f"  {route.path}")
        except ImportError:
            logger.warning("Không thể import app.main:app. Có thể không cần kiểm tra router.")
        
        # Chạy ứng dụng với uvicorn
        logger.info(f"Khởi động ứng dụng với uvicorn trên cổng {APP_PORT}...")
        uvicorn.run("app.main:app", host="127.0.0.1", port=APP_PORT, reload=True)
        
    except ImportError as e:
        logger.error(f"Lỗi import: {e}")
        logger.error("Vui lòng chạy từ thư mục gốc của dự án hoặc kiểm tra cài đặt các thư viện")
    except Exception as e:
        logger.error(f"Lỗi khi chạy ứng dụng: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_debug() 
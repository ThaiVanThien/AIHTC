import os
import subprocess
import sys
import logging

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def kill_port(port):
    """Dừng tất cả các tiến trình đang sử dụng một cổng nhất định"""
    try:
        # Tìm PID của tiến trình đang sử dụng cổng
        result = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode()
        lines = result.strip().split('\n')
        
        if not lines:
            logger.info(f"Không có tiến trình nào đang sử dụng cổng {port}")
            return False
        
        for line in lines:
            if 'LISTENING' in line:
                parts = line.strip().split()
                pid = parts[-1]
                
                # Kiểm tra nếu không phải số
                if not pid.isdigit():
                    continue
                
                logger.info(f"Tiến trình (PID: {pid}) đang sử dụng cổng {port}, đang dừng...")
                
                # Dừng tiến trình
                try:
                    subprocess.call(f"taskkill /F /PID {pid}", shell=True)
                    logger.info(f"Đã dừng tiến trình {pid}")
                    return True
                except Exception as e:
                    logger.error(f"Lỗi khi dừng tiến trình {pid}: {e}")
        
        return False
    except subprocess.CalledProcessError:
        logger.info(f"Không có tiến trình nào đang sử dụng cổng {port}")
        return False
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra cổng {port}: {e}")
        return False

def stop_all_servers():
    """Dừng tất cả các máy chủ uvicorn đang chạy"""
    try:
        # Tìm tất cả các tiến trình uvicorn
        result = subprocess.check_output("tasklist | findstr uvicorn", shell=True).decode()
        lines = result.strip().split('\n')
        
        if not lines:
            logger.info("Không tìm thấy tiến trình uvicorn nào đang chạy")
            return False
        
        for line in lines:
            parts = line.strip().split()
            pid = parts[1]
            
            # Kiểm tra nếu không phải số
            if not pid.isdigit():
                continue
            
            logger.info(f"Tìm thấy tiến trình uvicorn (PID: {pid}), đang dừng...")
            
            # Dừng tiến trình
            try:
                subprocess.call(f"taskkill /F /PID {pid}", shell=True)
                logger.info(f"Đã dừng tiến trình {pid}")
            except Exception as e:
                logger.error(f"Lỗi khi dừng tiến trình {pid}: {e}")
        
        return True
    except subprocess.CalledProcessError:
        logger.info("Không tìm thấy tiến trình uvicorn nào đang chạy")
        return False
    except Exception as e:
        logger.error(f"Lỗi khi tìm tiến trình uvicorn: {e}")
        return False

def kill_python_servers():
    """Dừng tất cả các tiến trình Python đang chạy liên quan đến uvicorn"""
    try:
        # Tìm tất cả các tiến trình python đang chạy uvicorn
        result = subprocess.check_output("wmic process where \"commandline like '%uvicorn%'\" get processid", shell=True).decode()
        lines = result.strip().split('\n')
        
        # Bỏ qua dòng tiêu đề
        if len(lines) > 1:
            lines = lines[1:]
        
        if not lines or all(not line.strip() for line in lines):
            logger.info("Không tìm thấy tiến trình Python nào đang chạy uvicorn")
            return False
        
        for line in lines:
            pid = line.strip()
            
            # Kiểm tra nếu không phải số
            if not pid.isdigit():
                continue
            
            logger.info(f"Tìm thấy tiến trình Python đang chạy uvicorn (PID: {pid}), đang dừng...")
            
            # Dừng tiến trình
            try:
                subprocess.call(f"taskkill /F /PID {pid}", shell=True)
                logger.info(f"Đã dừng tiến trình {pid}")
            except Exception as e:
                logger.error(f"Lỗi khi dừng tiến trình {pid}: {e}")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi tìm tiến trình Python: {e}")
        return False
    except Exception as e:
        logger.error(f"Lỗi: {e}")
        return False

if __name__ == "__main__":
    logger.info("Đang tìm và dừng các máy chủ đang chạy...")
    
    # Dừng tiến trình đang sử dụng cổng 8002
    port_killed = kill_port(8002)
    
    # Dừng tất cả các máy chủ uvicorn
    uvicorn_killed = stop_all_servers()
    
    # Dừng tất cả các tiến trình Python đang chạy uvicorn
    python_killed = kill_python_servers()
    
    if port_killed or uvicorn_killed or python_killed:
        logger.info("Đã dừng các máy chủ thành công")
    else:
        logger.info("Không tìm thấy máy chủ nào đang chạy")
    
    input("Nhấn Enter để thoát...") 
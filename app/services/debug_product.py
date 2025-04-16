import os
import sys
import logging
import asyncio
from pathlib import Path

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def setup_debug_environment():
    """
    Cấu hình môi trường để debug product service
    """
    # Lấy đường dẫn thư mục gốc của dự án
    current_file = Path(__file__).resolve()
    services_dir = current_file.parent
    app_dir = services_dir.parent
    project_root = app_dir.parent
    
    # Thêm thư mục gốc vào PYTHONPATH
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        logger.info(f"Đã thêm {project_root} vào PYTHONPATH")
    
    # In thông tin debug
    logger.info(f"Project root: {project_root}")
    logger.info(f"Python path: {sys.path}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    return project_root

async def test_product_service():
    """
    Kiểm tra product service
    """
    try:
        # Import product_service sau khi đã cấu hình PYTHONPATH
        from app.services.product_service import product_service
        
        # Test các phương thức của product_service
        logger.info("=== Kiểm tra phương thức is_product_query ===")
        test_queries = [
            "Giá gạo ST25 bao nhiêu?",
            "Tôi muốn mua gạo nếp",
            "Bạn có thể giới thiệu loại gạo nào tốt không?"
        ]
        
        for query in test_queries:
            result = await product_service.is_product_query(query)
            logger.info(f"Câu hỏi: '{query}' -> Là câu hỏi về sản phẩm: {result}")
        
        logger.info("\n=== Kiểm tra phương thức extract_product_name ===")
        for query in test_queries:
            result = await product_service.extract_product_name(query)
            logger.info(f"Câu hỏi: '{query}' -> Tên sản phẩm: '{result}'")
        
        logger.info("\n=== Kiểm tra phương thức get_products ===")
        product_names = ["gạo", "gạo ST25", "gạo nếp"]
        for name in product_names:
            logger.info(f"Tìm kiếm sản phẩm: '{name}'")
            result = await product_service.get_products(name)
            total = len(result.get("data", []))
            logger.info(f"Tìm thấy {total} sản phẩm")
            
            # Hiển thị chi tiết sản phẩm đầu tiên nếu có
            if total > 0:
                first_product = result.get("data", [])[0]
                logger.info(f"Sản phẩm đầu tiên: {first_product.get('productName', 'Không có tên')}")
                logger.info(f"Giá: {first_product.get('price_display', 'Không có giá')}")
        
        logger.info("\n=== Kiểm tra phương thức process_product_query ===")
        test_product_queries = [
            "Gạo ST25 có giá bao nhiêu?",
            "So sánh giá các loại gạo",
            "Loại gạo nào rẻ nhất?"
        ]
        
        for query in test_product_queries:
            logger.info(f"Xử lý câu hỏi: '{query}'")
            result = await product_service.process_product_query(query)
            # Hiển thị kết quả ngắn gọn
            summary = result[:200] + "..." if len(result) > 200 else result
            logger.info(f"Kết quả: {summary}")
        
    except ImportError as e:
        logger.error(f"Lỗi import: {e}")
    except Exception as e:
        logger.error(f"Lỗi khi test product service: {e}", exc_info=True)

if __name__ == "__main__":
    # Thiết lập môi trường debug
    setup_debug_environment()
    
    # Chạy test bất đồng bộ
    asyncio.run(test_product_service()) 
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import Dict, Any, Optional, List
import logging
import os

# Cấu hình logging
logger = logging.getLogger(__name__)

# URL cơ sở của API
router = APIRouter()

# Đảm bảo thư mục templates tồn tại
os.makedirs("app/templates", exist_ok=True)

# Cấu hình templates
templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse, summary="Trang danh mục sản phẩm")
async def get_categories_page(request: Request):
    """
    Hiển thị trang danh mục sản phẩm cho người dùng
    
    Returns:
        Trang HTML với danh sách các danh mục sản phẩm
    """
    try:
        return templates.TemplateResponse("categories.html", {
            "request": request,
            "title": "Danh mục sản phẩm"
        })
    except Exception as e:
        logger.error(f"Lỗi khi tải template danh mục: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi tải giao diện danh mục: {str(e)}")

@router.get("/api/categories", response_model=Dict[str, Any], summary="API lấy danh sách danh mục")
async def get_categories_api(page: int = 0, page_size: int = 50):
    """
    API lấy danh sách tất cả danh mục sản phẩm
    
    - **page**: Số trang (bắt đầu từ 0)
    - **page_size**: Số lượng danh mục mỗi trang
    
    Returns:
        Danh sách danh mục dạng JSON
    """
    try:
        # Import module với kiểm tra lỗi
        try:
            from app.api.query_demo.product_api import get_categories
        except ImportError as e:
            logger.error(f"Không thể import module product_api: {str(e)}")
            raise HTTPException(status_code=500, detail="Không thể tải thông tin danh mục. Vui lòng thử lại sau.")
        
        # Lấy danh sách danh mục
        categories = await get_categories(page_size, page)
        
        if not categories.get("success", False) or not categories.get("data", []):
            # Trả về danh sách trống nếu không tìm thấy danh mục
            return {
                "success": True,
                "data": [],
                "total": 0,
                "message": "Không tìm thấy danh mục nào"
            }
        
        return categories
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi khi lấy danh sách danh mục: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Đã xảy ra lỗi khi lấy danh sách danh mục: {str(e)}")

@router.get("/api/products/category/{category_id}", response_model=Dict[str, Any], summary="API lấy sản phẩm theo danh mục")
async def get_products_by_category_api(category_id: int, page: int = 0, page_size: int = 20):
    """
    API lấy danh sách sản phẩm theo danh mục
    
    - **category_id**: ID của danh mục cần tìm sản phẩm
    - **page**: Số trang (bắt đầu từ 0)
    - **page_size**: Số lượng sản phẩm mỗi trang
    
    Returns:
        Danh sách sản phẩm thuộc danh mục dạng JSON
    """
    try:
        # Import module với kiểm tra lỗi
        try:
            from app.api.query_demo.product_api import get_categories, get_products_by_category
        except ImportError as e:
            logger.error(f"Không thể import module product_api: {str(e)}")
            raise HTTPException(status_code=500, detail="Không thể tải thông tin sản phẩm. Vui lòng thử lại sau.")
        
        # Lấy danh sách danh mục
        categories = await get_categories(page_size=50)
        
        # Kiểm tra xem danh mục có tồn tại không
        category_name = None
        for cat in categories.get("data", []):
            cat_id = cat.get("category_id", cat.get("id"))
            if cat_id == category_id:
                category_name = cat.get("name")
                break
        
        if not category_name:
            # Xử lý đặc biệt cho dữ liệu mẫu
            if category_id == 3:
                category_name = "Thủ công mỹ nghệ"
            elif category_id == 4:
                category_name = "Thổ cẩm"
            elif category_id == 2:
                category_name = "Gạo các loại"
            else:
                raise HTTPException(status_code=404, detail=f"Không tìm thấy danh mục với ID: {category_id}")
        
        # Lấy sản phẩm theo category_id
        logger.info(f"API tìm sản phẩm theo category_id: {category_id}")
        products_result = await get_products_by_category(category_id, page, page_size)
        
        if not products_result.get("success", False) or not products_result.get("data", []):
            # Trả về danh sách trống nếu không tìm thấy sản phẩm
            return {
                "success": True,
                "data": [],
                "total": 0,
                "category_id": category_id,
                "category_name": category_name,
                "message": f"Không tìm thấy sản phẩm nào thuộc danh mục '{category_name}'"
            }
        
        # Thêm thông tin về danh mục vào kết quả
        products_result["category_name"] = category_name
        
        return products_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lỗi khi tìm sản phẩm theo danh mục: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Đã xảy ra lỗi khi tìm sản phẩm theo danh mục: {str(e)}")

@router.get("/category/{category_id}", response_class=HTMLResponse, summary="Trang sản phẩm theo danh mục")
async def get_products_by_category_page(request: Request, category_id: int):
    """
    Hiển thị trang sản phẩm theo danh mục cho người dùng
    
    - **category_id**: ID của danh mục
    
    Returns:
        Trang HTML với danh sách sản phẩm của danh mục đã chọn
    """
    try:
        # Import module để lấy thông tin danh mục
        try:
            from app.api.query_demo.product_api import get_categories
        except ImportError as e:
            logger.error(f"Không thể import module product_api: {str(e)}")
            raise HTTPException(status_code=500, detail="Không thể tải thông tin danh mục. Vui lòng thử lại sau.")
        
        # Lấy danh sách danh mục
        categories = await get_categories(page_size=50)
        
        # Kiểm tra xem danh mục có tồn tại không
        category_name = None
        for cat in categories.get("data", []):
            cat_id = cat.get("category_id", cat.get("id"))
            if cat_id == category_id:
                category_name = cat.get("name")
                break
        
        if not category_name:
            # Xử lý đặc biệt cho dữ liệu mẫu
            if category_id == 3:
                category_name = "Thủ công mỹ nghệ"
            elif category_id == 4:
                category_name = "Thổ cẩm"
            elif category_id == 2:
                category_name = "Gạo các loại"
            else:
                raise HTTPException(status_code=404, detail=f"Không tìm thấy danh mục với ID: {category_id}")
                
        return templates.TemplateResponse("category_products.html", {
            "request": request,
            "title": f"Sản phẩm - {category_name}",
            "category_id": category_id,
            "category_name": category_name
        })
    except Exception as e:
        logger.error(f"Lỗi khi tải template sản phẩm theo danh mục: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi tải giao diện sản phẩm: {str(e)}") 
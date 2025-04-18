import httpx
import logging
import os
import urllib.parse
import json
from typing import List, Dict, Any, Optional
from fastapi import HTTPException

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URL cơ sở của API
BASE_URL = "https://chodongbao.com/api"
AUTH_TOKEN = "ChoDongBao_HueCIT"  # Token xác thực

# Dữ liệu mẫu cho gạo - sử dụng khi API thực không hoạt động
SAMPLE_RICE_DATA = [
    {
        "productId": "RICE001",
        "productName": "Gạo ST25",
        "price": 45000,
        "unit": "kg",
        "sellerName": "Nông Trại Xanh",
        "description": "Gạo ST25 đạt giải nhất gạo ngon nhất thế giới",
        "images": ["https://example.com/rice1.jpg"]
    },
    {
        "productId": "RICE002",
        "productName": "Gạo Nếp Cái Hoa Vàng",
        "price": 35000,
        "unit": "kg",
        "sellerName": "HTX Nông Nghiệp",
        "description": "Gạo nếp thơm ngon, dẻo",
        "images": ["https://example.com/rice2.jpg"]
    },
    {
        "productId": "RICE003",
        "productName": "Gạo Lứt", 
        "price": 28000,
        "unit": "kg",
        "sellerName": "Organic Farm",
        "description": "Gạo lứt hữu cơ, tốt cho sức khỏe",
        "images": ["https://example.com/rice3.jpg"]
    },
    {
        "productId": "RICE004",
        "productName": "Gạo Japonica", 
        "price": 52000,
        "unit": "kg",
        "sellerName": "Nhà Phân Phối ABC",
        "description": "Gạo Nhật, hạt tròn, dẻo",
        "images": ["https://example.com/rice4.jpg"]
    },
    {
        "productId": "RICE005",
        "productName": "Gạo Tấm", 
        "price": 18000,
        "unit": "kg",
        "sellerName": "Cửa Hàng Gạo Sạch",
        "description": "Gạo tấm dùng để nấu cháo, cơm tấm",
        "images": ["https://example.com/rice5.jpg"]
    }
]

# Dữ liệu mẫu cho danh mục
SAMPLE_CATEGORIES = [
    {
        "category_id": 1,
        "name": "Thực phẩm",
        "description": "Các sản phẩm thực phẩm",
        "parent_id": 0
    },
    {
        "category_id": 2,
        "name": "Gạo các loại",
        "description": "Các sản phẩm gạo",
        "parent_id": 1
    },
    {
        "category_id": 3,
        "name": "Thủ công mỹ nghệ",
        "description": "Sản phẩm thủ công",
        "parent_id": 0
    },
    {
        "category_id": 4,
        "name": "Thổ cẩm",
        "description": "Các sản phẩm thổ cẩm",
        "parent_id": 3
    },
    {
        "category_id": 5,
        "name": "Đặc sản vùng miền",
        "description": "Đặc sản vùng miền",
        "parent_id": 0
    }
]

# Cache dữ liệu
product_cache = {}

async def test_api_connection() -> Dict[str, Any]:
    """
    Kiểm tra kết nối đến API Chợ Đồng Bào
    
    Returns:
        Kết quả kiểm tra kết nối
    """
    test_result = {
        "success": False,
        "message": "Sử dụng dữ liệu mẫu thay vì API thực",
        "status_code": None,
        "response": {
            "count": len(SAMPLE_RICE_DATA),
            "first_item_example": SAMPLE_RICE_DATA[0] if SAMPLE_RICE_DATA else None
        }
    }
    
    # Không thực hiện kết nối thật nữa mà trả về thông báo dùng dữ liệu mẫu
    logger.info("Đang sử dụng dữ liệu mẫu thay vì gọi API thực")
    return test_result

async def get_products_by_name_real_api(name: str, page: int = 0, page_size: int = 20) -> List[Dict[str, Any]]:
    """
    Lấy danh sách sản phẩm từ chodongbao.com theo tên (API thực)
    
    Args:
        name: Tên sản phẩm cần tìm
        page: Số trang (bắt đầu từ 0)
        page_size: Số lượng sản phẩm mỗi trang
        
    Returns:
        Danh sách sản phẩm
    """
    # Mã hóa tên sản phẩm để tránh lỗi ký tự đặc biệt
    name_encoded = urllib.parse.quote(name)
    url = f"{BASE_URL}/ProductsByName/{page_size}?name={name_encoded}&page={page}"
    
    # Log thông tin gọi API để debug
    logger.info(f"Gọi API: {url}")
    logger.info(f"Headers: authenticatetoken={AUTH_TOKEN}")
    
    headers = {"authenticatetoken": AUTH_TOKEN}
    
    try:
        async with httpx.AsyncClient() as client:
            # Tăng timeout để tránh lỗi kết nối
            response = await client.get(url, headers=headers, timeout=30.0)
            
            # Log response
            logger.info(f"API response status: {response.status_code}")
            
            # Kiểm tra status code
            if response.status_code != 200:
                logger.error(f"Lỗi khi gọi API: {response.status_code} - {response.text}")
                return []
            
            # Parse JSON
            data = response.json()
            logger.info(f"Đã tìm thấy {len(data)} sản phẩm")
            return data
            
    except Exception as e:
        logger.error(f"Lỗi khi gọi API chodongbao: {str(e)}")
        return []

async def get_products_by_name(name: str, page: int = 0, page_size: int = 20) -> List[Dict[str, Any]]:
    """
    Lấy danh sách sản phẩm theo tên - sử dụng dữ liệu mẫu
    
    Args:
        name: Tên sản phẩm cần tìm
        page: Số trang (bắt đầu từ 0)
        page_size: Số lượng sản phẩm mỗi trang
        
    Returns:
        Danh sách sản phẩm
    """
    # Kiểm tra nếu đã có trong cache
    cache_key = f"{name}_{page}_{page_size}"
    if cache_key in product_cache:
        logger.info(f"Lấy dữ liệu từ cache cho: {name}")
        return product_cache[cache_key]
    
    # Lọc dữ liệu mẫu theo tên
    name_lower = name.lower()
    
    # Trường hợp đặc biệt cho "gạo"
    if "gạo" in name_lower:
        # Lọc thêm theo giá nếu có từ khóa giá
        if "dưới 100" in name_lower or "dưới 100k" in name_lower or "dưới 100 nghìn" in name_lower:
            filtered_data = [p for p in SAMPLE_RICE_DATA if p["price"] < 100000]
            logger.info(f"Đã lọc {len(filtered_data)} sản phẩm gạo dưới 100 nghìn")
            
            # Cập nhật thông tin giá
            for product in filtered_data:
                product["price_display"] = f"{product['price']:,}đ".replace(",", ".")
            
            # Lưu cache
            product_cache[cache_key] = filtered_data
            return filtered_data
        
        # Trả về toàn bộ dữ liệu gạo mẫu
        logger.info(f"Trả về {len(SAMPLE_RICE_DATA)} sản phẩm gạo mẫu")
        
        # Cập nhật thông tin giá
        for product in SAMPLE_RICE_DATA:
            product["price_display"] = f"{product['price']:,}đ".replace(",", ".")
            
        # Lưu cache
        product_cache[cache_key] = SAMPLE_RICE_DATA
        return SAMPLE_RICE_DATA
    
    # Các sản phẩm khác sẽ trả về danh sách rỗng
    logger.info(f"Không tìm thấy dữ liệu mẫu cho: {name}")
    return []

async def search_products(
    keyword: Optional[str] = None, 
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    page: int = 0,
    page_size: int = 20
) -> Dict[str, Any]:
    """
    Tìm kiếm sản phẩm với nhiều điều kiện - Sử dụng dữ liệu mẫu
    
    Args:
        keyword: Từ khóa tìm kiếm
        category: Danh mục sản phẩm
        min_price: Giá tối thiểu
        max_price: Giá tối đa
        page: Số trang
        page_size: Số lượng sản phẩm mỗi trang
        
    Returns:
        Kết quả tìm kiếm
    """
    # Tìm sản phẩm theo tên
    products = []
    if keyword:
        products = await get_products_by_name(keyword)
    
    # Lọc theo danh mục nếu có
    if category and products:
        products = [p for p in products if category.lower() in p.get('productName', '').lower()]
    
    # Lọc theo giá nếu có
    if min_price is not None:
        products = [p for p in products if p.get('price', 0) >= min_price]
    if max_price is not None:
        products = [p for p in products if p.get('price', 0) <= max_price]
    
    return {
        "products": products,
        "total": len(products),
        "page": page,
        "page_size": page_size,
        "keyword": keyword,
        "category": category,
        "min_price": min_price,
        "max_price": max_price
    }

def format_product_list(products: List[Dict[str, Any]]) -> str:
    """
    Định dạng danh sách sản phẩm để hiển thị
    
    Args:
        products: Danh sách sản phẩm
        
    Returns:
        Chuỗi kết quả đã định dạng
    """
    if not products:
        return "Không tìm thấy sản phẩm nào."
    
    result = f"Tìm thấy {len(products)} sản phẩm:\n\n"
    
    for i, product in enumerate(products, 1):
        name = product.get("productName", product.get("name", "Không có tên"))
        
        # Lấy giá từ trường price_display nếu có, nếu không thì tính từ price
        price_display = product.get("price_display")
        if not price_display:
            price = product.get("price", 0)
            price_display = f"{price:,}đ".replace(",", ".")
            
        unit = product.get("unit", "")
        seller = product.get("sellerName", "Không có thông tin")
        
        result += f"{i}. {name}\n"
        result += f"   Giá: {price_display}/{unit}\n"
        result += f"   Người bán: {seller}\n\n"
    
    return result 

async def get_categories_real_api(page_size: int = 20, page: int = 0) -> Dict[str, Any]:
    """
    Lấy danh sách danh mục sản phẩm từ API thực
    
    Args:
        page_size: Số lượng danh mục mỗi trang
        page: Số trang (bắt đầu từ 0)
        
    Returns:
        Danh sách danh mục sản phẩm
    """
    url = f"{BASE_URL}/Categories/{page_size}?page={page}"
    
    # Log thông tin gọi API để debug
    logger.info(f"Gọi API danh mục: {url}")
    logger.info(f"Headers: authenticatetoken={AUTH_TOKEN}")
    
    headers = {"authenticatetoken": AUTH_TOKEN}
    
    try:
        async with httpx.AsyncClient() as client:
            # Tăng timeout để tránh lỗi kết nối
            response = await client.get(url, headers=headers, timeout=30.0)
            
            # Log response
            logger.info(f"API response status: {response.status_code}")
            
            # Kiểm tra status code
            if response.status_code != 200:
                logger.error(f"Lỗi khi gọi API danh mục: {response.status_code} - {response.text}")
                return {"success": False, "data": [], "total": 0, "message": f"Lỗi API: {response.status_code}"}
            
            # Parse JSON
            data = response.json()
            
            # Chuẩn hóa kết quả
            return {
                "success": True,
                "data": data,
                "total": len(data),
                "message": "Lấy danh mục thành công"
            }
            
    except Exception as e:
        logger.error(f"Lỗi khi gọi API danh mục: {str(e)}")
        return {"success": False, "data": [], "total": 0, "message": f"Lỗi: {str(e)}"}

async def get_categories(page_size: int = 20, page: int = 0) -> Dict[str, Any]:
    """
    Lấy danh sách danh mục sản phẩm
    
    Args:
        page_size: Số lượng danh mục mỗi trang
        page: Số trang (bắt đầu từ 0)
        
    Returns:
        Danh sách danh mục sản phẩm
    """
    # Kiểm tra nếu đã có trong cache
    cache_key = f"categories_{page}_{page_size}"
    if cache_key in product_cache:
        logger.info(f"Lấy danh mục từ cache")
        return product_cache[cache_key]
    
    # Thử gọi API thực
    try:
        api_result = await get_categories_real_api(page_size, page)
        if api_result["success"] and api_result["data"]:
            # Lưu vào cache
            product_cache[cache_key] = api_result
            return api_result
    except Exception as e:
        logger.error(f"Lỗi khi gọi API danh mục thực: {str(e)}")
    
    # Nếu API thực thất bại, sử dụng dữ liệu mẫu
    logger.info(f"Sử dụng dữ liệu danh mục mẫu")
    
    # Phân trang dữ liệu mẫu
    start_idx = page * page_size
    end_idx = start_idx + page_size
    paginated_data = SAMPLE_CATEGORIES[start_idx:end_idx]
    
    result = {
        "success": True,
        "data": paginated_data,
        "total": len(SAMPLE_CATEGORIES),
        "message": "Lấy danh mục từ dữ liệu mẫu"
    }
    
    # Lưu vào cache
    product_cache[cache_key] = result
    return result

def format_categories(categories: Dict[str, Any]) -> str:
    """
    Định dạng danh sách danh mục để hiển thị
    
    Args:
        categories: Kết quả từ hàm get_categories
        
    Returns:
        Chuỗi kết quả đã định dạng
    """
    if not categories.get("success", False) or not categories.get("data", []):
        return "Không lấy được danh sách danh mục sản phẩm."
    
    category_list = categories.get("data", [])
    result = f"Có {len(category_list)} danh mục sản phẩm:\n\n"
    
    for i, category in enumerate(category_list, 1):
        category_id = category.get("category_id", category.get("id", ""))
        name = category.get("name", "Không có tên")
        description = category.get("description", "")
        parent_id = category.get("parent_id", 0)
        
        result += f"{i}. {name} (ID: {category_id})\n"
        if description:
            result += f"   Mô tả: {description}\n"
        if parent_id:
            result += f"   Danh mục cha: {parent_id}\n"
        result += "\n"
    
    return result 
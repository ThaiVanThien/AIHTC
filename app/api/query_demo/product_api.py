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

# Dữ liệu mẫu cho sản phẩm thủ công mỹ nghệ
SAMPLE_HANDCRAFT_DATA = [
    {
        "productId": "HC001",
        "productName": "Vải thổ cẩm A Lưới",
        "price": 500000,
        "unit": "mét",
        "sellerName": "Làng Nghề A Lưới",
        "description": "Vải thổ cẩm dệt thủ công từ người dân A Lưới, với họa tiết truyền thống đặc trưng",
        "images": ["https://example.com/handcraft1.jpg"],
        "category_id": 3
    },
    {
        "productId": "HC002",
        "productName": "Khăn thổ cẩm A So",
        "price": 350000,
        "unit": "cái",
        "sellerName": "Làng Nghề A So",
        "description": "Khăn thổ cẩm dệt thủ công từ chỉ 100% cotton",
        "images": ["https://example.com/handcraft2.jpg"],
        "category_id": 4
    },
    {
        "productId": "HC003",
        "productName": "Túi thổ cẩm đựng Laptop",
        "price": 320000,
        "unit": "cái",
        "sellerName": "AzaKooh",
        "description": "Túi đựng laptop thổ cẩm sang trọng, bảo vệ thiết bị của bạn với phong cách độc đáo",
        "images": ["https://example.com/handcraft3.jpg"],
        "category_id": 4
    },
    {
        "productId": "HC004",
        "productName": "Áo dài thổ cẩm AzaKooh",
        "price": 1200000,
        "unit": "bộ",
        "sellerName": "AzaKooh",
        "description": "Áo dài thổ cẩm phối vải dệt truyền thống, phù hợp mọi dịp lễ, Tết",
        "images": ["https://example.com/handcraft4.jpg"],
        "category_id": 4
    },
    {
        "productId": "HC005",
        "productName": "Lắc tay thổ cẩm",
        "price": 80000,
        "unit": "cái",
        "sellerName": "AzaKooh",
        "description": "Lắc tay thủ công với họa tiết thổ cẩm đặc trưng, thích hợp làm quà tặng",
        "images": ["https://example.com/handcraft5.jpg"],
        "category_id": 4
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

# Map category_id với danh sách sản phẩm mẫu
CATEGORY_PRODUCT_MAP = {
    # Thực phẩm
    1: [],
    # Gạo các loại 
    2: SAMPLE_RICE_DATA,
    # Thủ công mỹ nghệ
    3: SAMPLE_HANDCRAFT_DATA,
    # Thổ cẩm
    4: [p for p in SAMPLE_HANDCRAFT_DATA if p.get("category_id") == 4],
    # Đặc sản vùng miền
    5: []
}

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

async def get_products_by_category_real_api(category_id: int, page: int = 0, page_size: int = 20) -> Dict[str, Any]:
    """
    Lấy danh sách sản phẩm từ API dựa theo danh mục
    
    Args:
        category_id: ID của danh mục cần tìm sản phẩm
        page: Số trang (bắt đầu từ 0)
        page_size: Số lượng sản phẩm mỗi trang
        
    Returns:
        Danh sách sản phẩm thuộc danh mục
    """
    url = f"{BASE_URL}/ProductsByCategory/{category_id}?page={page}&page_size={page_size}"
    
    # Log thông tin gọi API để debug
    logger.info(f"Gọi API sản phẩm theo danh mục: {url}")
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
                logger.error(f"Lỗi khi gọi API sản phẩm theo danh mục: {response.status_code} - {response.text}")
                return {"success": False, "data": [], "total": 0, "message": f"Lỗi API: {response.status_code}"}
            
            # Parse JSON
            data = response.json()
            
            # Đảm bảo mỗi sản phẩm có category_id
            for product in data:
                if "category_id" not in product:
                    product["category_id"] = category_id
                    
                # Đảm bảo mỗi sản phẩm có trường price_display
                if "price" in product and "price_display" not in product:
                    price = product["price"]
                    product["price_display"] = f"{price:,}đ".replace(",", ".")
            
            # Chuẩn hóa kết quả
            return {
                "success": True,
                "data": data,
                "total": len(data),
                "category_id": category_id,
                "message": f"Lấy sản phẩm theo danh mục {category_id} thành công"
            }
            
    except Exception as e:
        logger.error(f"Lỗi khi gọi API sản phẩm theo danh mục: {str(e)}")
        return {"success": False, "data": [], "total": 0, "message": f"Lỗi: {str(e)}"}

async def get_products_by_category(category_id: int, page: int = 0, page_size: int = 20) -> Dict[str, Any]:
    """
    Lấy danh sách sản phẩm theo danh mục
    
    Args:
        category_id: ID của danh mục cần tìm sản phẩm
        page: Số trang (bắt đầu từ 0)
        page_size: Số lượng sản phẩm mỗi trang
        
    Returns:
        Danh sách sản phẩm thuộc danh mục
    """
    # Kiểm tra nếu đã có trong cache
    cache_key = f"products_category_{category_id}_{page}_{page_size}"
    if cache_key in product_cache:
        logger.info(f"Lấy sản phẩm từ cache cho danh mục: {category_id}")
        return product_cache[cache_key]
    
    # Thử gọi API thực
    try:
        api_result = await get_products_by_category_real_api(category_id, page, page_size)
        if api_result["success"] and api_result["data"]:
            # Lưu vào cache
            product_cache[cache_key] = api_result
            return api_result
    except Exception as e:
        logger.error(f"Lỗi khi gọi API sản phẩm theo danh mục thực: {str(e)}")
    
    # Nếu API thực thất bại, sử dụng dữ liệu mẫu
    logger.info(f"Sử dụng dữ liệu mẫu cho danh mục: {category_id}")
    
    # Lấy dữ liệu mẫu cho danh mục
    sample_data = []
    category_id_str = str(category_id)
    if category_id in CATEGORY_PRODUCT_MAP:
        sample_data = CATEGORY_PRODUCT_MAP[category_id]
    
    # Phân trang dữ liệu mẫu
    start_idx = page * page_size
    end_idx = start_idx + page_size
    paginated_data = sample_data[start_idx:end_idx]
    
    # Đảm bảo mỗi sản phẩm có trường price_display
    for product in paginated_data:
        if "price" in product and "price_display" not in product:
            price = product["price"]
            product["price_display"] = f"{price:,}đ".replace(",", ".")
    
    result = {
        "success": True,
        "data": paginated_data,
        "total": len(sample_data),
        "category_id": category_id,
        "message": f"Lấy sản phẩm theo danh mục {category_id} từ dữ liệu mẫu"
    }
    
    # Lưu vào cache
    product_cache[cache_key] = result
    return result 

async def get_products(self, product_name: str, page: int = 0, page_size: int = 20) -> Dict[str, Any]:
    # ... existing code ...
    for product in result["data"]:
        # Thêm thông tin chi tiết
        product["description"] = product.get("description", "Không có mô tả")
        product["images"] = product.get("images", [])
    return result 

async def get_product_details(self, product_id: str) -> Dict[str, Any]:
    try:
        url = f"{self.base_url}/{product_id}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            if response.status_code == 200:
                return response.json()
            else:
                return {"success": False, "message": "Không tìm thấy sản phẩm."}
    except Exception as e:
        logger.error(f"Lỗi khi lấy thông tin sản phẩm: {str(e)}")
        return {"success": False, "message": str(e)} 

async def process_product_query(self, query: str) -> str:
    # ... existing code ...
    if is_specific_product_query:
        product_id = extract_product_id_from_query(query)  # Hàm để trích xuất productId từ câu hỏi
        product_details = await self.get_product_details(product_id)
        if product_details.get("success"):
            return format_product_details(product_details)  # Hàm để định dạng thông tin sản phẩm
        else:
            return product_details.get("message", "Không tìm thấy thông tin sản phẩm.") 

def format_product_details(product: Dict[str, Any]) -> str:
    return f"""
    <h2>{product['productName']}</h2>
    <p>Giá: {product['price_display']}</p>
    <p>Đơn vị: {product['unit']}</p>
    <p>Người bán: {product['sellerName']}</p>
    <p>Mô tả: {product['description']}</p>
    <img src="{product['images'][0]}" alt="{product['productName']}">
    """ 
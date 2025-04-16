from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any, List
import httpx
import urllib.parse
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URL cơ sở của API
BASE_URL = "https://chodongbao.com/api"
AUTH_TOKEN = "ChoDongBao_HueCIT"  # Token xác thực

router = APIRouter()

async def get_products_by_name(name: str, page: int = 0, page_size: int = 20) -> List[Dict[str, Any]]:
    """
    Lấy danh sách sản phẩm từ API Chợ Đồng Bào theo tên
    
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
            
            # Định dạng giá
            for product in data:
                if "price" in product:
                    product["price_display"] = f"{product['price']:,}đ".replace(",", ".")
            
            return data
            
    except Exception as e:
        logger.error(f"Lỗi khi gọi API chodongbao: {str(e)}")
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
    Tìm kiếm sản phẩm với nhiều điều kiện
    
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
        products = await get_products_by_name(keyword, page, page_size)
    
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

@router.get("/products", summary="Lấy danh sách sản phẩm theo tên")
async def get_products_api(
    name: str = Query("", description="Tên sản phẩm cần tìm"),
    page: int = Query(0, description="Số trang (bắt đầu từ 0)"),
    page_size: int = Query(20, description="Số lượng sản phẩm mỗi trang")
):
    """
    Lấy danh sách sản phẩm từ API Chợ Đồng Bào theo tên
    """
    products = await get_products_by_name(name, page, page_size)
    return {
        "success": True,
        "data": products,
        "total": len(products),
        "page": page,
        "page_size": page_size
    }

@router.get("/products/search", summary="Tìm kiếm sản phẩm với nhiều điều kiện")
async def search_products_api(
    keyword: Optional[str] = Query(None, description="Từ khóa tìm kiếm"),
    category: Optional[str] = Query(None, description="Danh mục sản phẩm"),
    min_price: Optional[float] = Query(None, description="Giá tối thiểu"),
    max_price: Optional[float] = Query(None, description="Giá tối đa"),
    page: int = Query(0, description="Số trang"),
    page_size: int = Query(20, description="Số lượng sản phẩm mỗi trang")
):
    """
    Tìm kiếm sản phẩm với nhiều điều kiện từ API Chợ Đồng Bào
    """
    if not keyword and not category:
        raise HTTPException(
            status_code=400, 
            detail="Phải cung cấp ít nhất từ khóa hoặc danh mục sản phẩm"
        )
    
    results = await search_products(
        keyword=keyword,
        category=category,
        min_price=min_price,
        max_price=max_price,
        page=page,
        page_size=page_size
    )
    
    return results

@router.get("/products/format", summary="Định dạng danh sách sản phẩm để hiển thị")
async def format_products_api(
    name: str = Query(..., description="Tên sản phẩm cần tìm")
):
    """
    Định dạng danh sách sản phẩm để hiển thị thân thiện
    """
    products = await get_products_by_name(name)
    formatted = format_product_list(products)
    return {
        "success": True,
        "formatted_text": formatted,
        "product_count": len(products)
    }

@router.get("/products/test", summary="Kiểm tra kết nối đến API")
async def test_connection():
    """
    Kiểm tra kết nối đến API Chợ Đồng Bào
    """
    try:
        # Thử gọi API với tên sản phẩm đơn giản
        products = await get_products_by_name("gạo", 0, 1)
        return {
            "success": True,
            "message": "Kết nối API thành công",
            "product_count": len(products),
            "sample": products[0] if products else None
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Lỗi kết nối API: {str(e)}"
        } 
import logging
import re
import os
import sys
from typing import List, Dict, Any, Optional
import httpx
import aiohttp
import json

# Xử lý PYTHONPATH để có thể import app module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import services
try:
    from app.services.openai_service import openai_service
    from app.services.gemini_service import gemini_service
except ImportError as e:
    # Thử import tương đối nếu import tuyệt đối không hoạt động
    logger = logging.getLogger(__name__)
    logger.warning(f"ImportError: {e}. Thử import tương đối...")
    
    try:
        # Import tương đối
        from .openai_service import openai_service
        from .gemini_service import gemini_service
        
        # Nếu import tương đối thành công, ghi log
        logger.info("Import tương đối thành công")
    except ImportError as e:
        # Nếu cả hai cách import đều thất bại, ghi log lỗi và tiếp tục
        logger.error(f"Cả hai phương pháp import đều thất bại: {e}")
        raise

logger = logging.getLogger(__name__)

class ProductService:
    def __init__(self):
        self.base_url = "http://localhost:8002/api/products"
        
    async def is_product_query(self, query: str) -> bool:
        """
        Kiểm tra xem câu hỏi có liên quan đến sản phẩm hay không
        
        Args:
            query: Câu hỏi cần kiểm tra
            
        Returns:
            True nếu câu hỏi liên quan đến sản phẩm
        """
        # Trước tiên, kiểm tra các câu hỏi về danh mục
        category_specific_phrases = [
            "danh sách danh mục", "danh mục", "category", "loại sản phẩm", 
            "phân loại", "các loại", "nhóm sản phẩm", "danh sách danh mục"
        ]
        query_lower = query.lower()
        
        # Nếu câu hỏi chỉ chứa các từ khóa danh mục (không có từ khóa sản phẩm)
        if any(phrase == query_lower.strip() for phrase in category_specific_phrases):
            logger.info(f"Câu hỏi '{query}' được xác định là về danh mục, không phải sản phẩm")
            return False
        
        # Mẫu từ khóa liên quan đến sản phẩm
        product_keywords = [
            "giá", "mua", "bán", "sản phẩm", "hàng hóa", "giá cả", 
            "mặt hàng", "đồ", "đắt", "rẻ", "tiền", "giá tiền", "mua bán",
            "bao nhiêu", "tốt", "xấu", "chất lượng", "gạo", "thực phẩm",
            "danh sách sản phẩm", "liệt kê sản phẩm", "xem sản phẩm", "tất cả sản phẩm", 
            "có những sản phẩm", "loại nào", "thông tin sản phẩm", "giới thiệu sản phẩm",
            "chủng loại"
        ]
        
        # Kiểm tra từng từ khóa
        for keyword in product_keywords:
            if keyword in query_lower:
                return True
                
        # Kiểm tra các mẫu câu hỏi về sản phẩm
        product_patterns = [
            r"(?:giá|mua|bán|tìm).*(?:sản phẩm|hàng)",
            r"(?:sản phẩm|hàng).*(?:giá|mua|bán)",
            r"(?:có|tìm|mua|bán).*(?:gạo|thực phẩm)",
            r"giá.*(?:bao nhiêu|thế nào)",
            r"(?:sản phẩm|hàng hóa).*(?:nào|có)",
            r"danh sách(?:.*)sản phẩm",
            r"liệt kê(?:.*)sản phẩm",
            r"có (?:những|các) (?:sản phẩm|loại|mặt hàng)",
            r"cửa hàng.*(?:có|bán)"
        ]
        
        for pattern in product_patterns:
            if re.search(pattern, query_lower):
                return True
                
        return False
    
    async def extract_product_name(self, query: str) -> str:
        """
        Trích xuất tên sản phẩm từ câu hỏi
        
        Args:
            query: Câu hỏi cần trích xuất
            
        Returns:
            Tên sản phẩm
        """
        try:
            # Thử dùng AI để trích xuất
            system_prompt = """
            Bạn là trợ lý trích xuất thông tin. Nhiệm vụ của bạn là trích xuất tên sản phẩm từ câu hỏi.
            Chỉ trả về tên sản phẩm, không thêm thông tin khác. Ví dụ:
            Câu hỏi: "Tôi muốn tìm thông tin về gạo ST25" -> Trả về: "gạo ST25"
            Câu hỏi: "Giá gạo nếp bao nhiêu?" -> Trả về: "gạo nếp"
            Câu hỏi: "Có loại gạo nào ngon không?" -> Trả về: "gạo"
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            # Thử dùng Gemini
            try:
                response = await gemini_service.chat(
                    messages=messages,
                    temperature=0.1,
                    max_tokens=20
                )
                product_name = response["answer"].strip()
                logger.info(f"Trích xuất tên sản phẩm bằng Gemini: '{product_name}'")
                return product_name
            except Exception as e:
                logger.warning(f"Lỗi khi dùng Gemini: {str(e)}")
                
                # Nếu Gemini lỗi, thử dùng OpenAI
                try:
                    response = await openai_service.chat(
                        messages=messages,
                        temperature=0.1,
                        max_tokens=20
                    )
                    product_name = response["answer"].strip()
                    logger.info(f"Trích xuất tên sản phẩm bằng OpenAI: '{product_name}'")
                    return product_name
                except Exception as e:
                    logger.warning(f"Lỗi khi dùng OpenAI: {str(e)}")
            
            # Nếu cả hai đều lỗi, dùng phương pháp đơn giản
            # Tìm từ khóa phổ biến
            keywords = ["gạo", "thực phẩm", "nếp", "rau", "củ", "quả"]
            for keyword in keywords:
                if keyword in query.lower():
                    return keyword
            
            # Trả về từ khóa mặc định
            return "gạo"
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất tên sản phẩm: {str(e)}")
            return "gạo"  # Mặc định trả về "gạo" nếu có lỗi
    
    async def get_products(self, product_name: str, page: int = 0, page_size: int = 20) -> Dict[str, Any]:
        """
        Lấy danh sách sản phẩm từ API
        
        Args:
            product_name: Tên sản phẩm cần tìm
            page: Số trang
            page_size: Số lượng sản phẩm mỗi trang
            
        Returns:
            Kết quả từ API
        """
        try:
            # Chuỗi rỗng name="" là hợp lệ, API sẽ trả về tất cả sản phẩm
            # Chỉ cần đảm bảo product_name không phải là None
            if product_name is None:
                product_name = ""  # Sử dụng chuỗi rỗng để lấy tất cả sản phẩm
                logger.info("Product name is None, using empty string to get all products")
            
            url = f"{self.base_url}?name={product_name}&page={page}&page_size={page_size}"
            logger.info(f"Calling API: {url}")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                if response.status_code == 200:
                    result = response.json()
                    
                    # Xử lý kết quả để đảm bảo mỗi sản phẩm có productId
                    # Nếu API không trả về productId, tạo một ID giả dựa trên vị trí
                    if "data" in result and isinstance(result["data"], list):
                        for i, product in enumerate(result["data"]):
                            if "productId" not in product or not product["productId"]:
                                # Tạo ID giả nếu không có
                                product["productId"] = f"product_{i+1}"
                            
                            # Đảm bảo mỗi sản phẩm có trường price_display
                            if "price" in product and "price_display" not in product:
                                price = product["price"]
                                product["price_display"] = f"{price:,}đ".replace(",", ".")
                    
                    return result
                else:
                    logger.error(f"Lỗi khi gọi API sản phẩm: {response.status_code} - {response.text}")
                    return {"success": False, "data": [], "total": 0}
        except Exception as e:
            logger.error(f"Lỗi khi gọi API sản phẩm: {str(e)}")
            return {"success": False, "data": [], "total": 0}
    
    async def analyze_products_with_ai(self, products: List[Dict[str, Any]], query: str) -> str:
        """
        Phân tích danh sách sản phẩm bằng AI theo câu hỏi của người dùng
        
        Args:
            products: Danh sách sản phẩm cần phân tích
            query: Câu hỏi gốc của người dùng
            
        Returns:
            Kết quả phân tích
        """
        if not products:
            return "Không tìm thấy sản phẩm nào phù hợp."
        
        try:
            # Kiểm tra nếu query là về danh sách sản phẩm
            query_lower = query.lower()
            is_product_list_query = any(phrase in query_lower for phrase in [
                "danh sách sản phẩm", "liệt kê sản phẩm", "xem sản phẩm", 
                "các sản phẩm", "tất cả sản phẩm", "có những sản phẩm nào",
                "những sản phẩm", "các loại", "có những loại nào", "danh mục"
            ])
            
            # Kiểm tra nếu là yêu cầu sắp xếp theo giá
            is_price_sort = "sắp xếp" in query_lower and ("giá" in query_lower or "đắt" in query_lower or "rẻ" in query_lower)
            
            # Kiểm tra nếu là yêu cầu lọc theo giá
            is_price_filter = False
            min_price = None
            max_price = None
            
            # Tìm giá tối thiểu từ câu hỏi
            min_price_patterns = [
                r"giá(?:\s+)(?:từ|trên|hơn|lớn hơn|>)(?:\s+)(\d+)(?:\s*)(k|nghìn|ngàn|triệu|đồng)?",
                r"(?:từ|trên|hơn|lớn hơn|>)(?:\s+)(\d+)(?:\s*)(k|nghìn|ngàn|triệu|đồng)?(?:\s+)(?:đồng|vnd)?",
                r"(\d+)(?:\s*)(k|nghìn|ngàn|triệu|đồng)?(?:\s+)(?:trở lên)"
            ]
            
            for pattern in min_price_patterns:
                matches = re.search(pattern, query_lower)
                if matches:
                    value = int(matches.group(1))
                    unit = matches.group(2) if len(matches.groups()) > 1 else None
                    
                    # Chuyển đổi giá trị theo đơn vị
                    if unit in ["k", "nghìn", "ngàn"]:
                        min_price = value * 1000
                    elif unit == "triệu":
                        min_price = value * 1000000
                    else:
                        min_price = value
                    
                    is_price_filter = True
                    break
            
            # Tìm giá tối đa từ câu hỏi
            max_price_patterns = [
                r"giá(?:\s+)(?:đến|tới|dưới|nhỏ hơn|<)(?:\s+)(\d+)(?:\s*)(k|nghìn|ngàn|triệu|đồng)?",
                r"(?:đến|tới|dưới|nhỏ hơn|<)(?:\s+)(\d+)(?:\s*)(k|nghìn|ngàn|triệu|đồng)?(?:\s+)(?:đồng|vnd)?",
                r"(?:không quá|tối đa|cao nhất|nhiều nhất)(?:\s+)(\d+)(?:\s*)(k|nghìn|ngàn|triệu|đồng)?"
            ]
            
            for pattern in max_price_patterns:
                matches = re.search(pattern, query_lower)
                if matches:
                    value = int(matches.group(1))
                    unit = matches.group(2) if len(matches.groups()) > 1 else None
                    
                    # Chuyển đổi giá trị theo đơn vị
                    if unit in ["k", "nghìn", "ngàn"]:
                        max_price = value * 1000
                    elif unit == "triệu":
                        max_price = value * 1000000
                    else:
                        max_price = value
                    
                    is_price_filter = True
                    break
            
            # Sắp xếp tất cả sản phẩm theo giá tăng dần trước
            sorted_products = sorted(products, key=lambda x: float(x.get("price", 0)))
            
            # Lọc sản phẩm theo giá nếu cần
            if is_price_filter:
                filtered_products = []
                for product in sorted_products:
                    price = float(product.get("price", 0))
                    
                    # Áp dụng lọc theo giá tối thiểu và tối đa
                    if (min_price is None or price >= min_price) and (max_price is None or price <= max_price):
                        filtered_products.append(product)
                
                sorted_products = filtered_products
            
            # Giới hạn số lượng sản phẩm để tránh token quá lớn
            max_products = 100 if is_product_list_query or is_price_sort or is_price_filter else 20
            truncated_products = sorted_products[:max_products]
            
            # Chuẩn bị mô tả ngắn gọn
            def get_short_description(description: str, max_length: int = 100) -> str:
                if not description:
                    return ""
                    
                # Nếu mô tả quá dài, cắt ngắn
                if len(description) > max_length:
                    # Cắt ở ký tự không phải dấu cách cuối cùng trước max_length
                    short_desc = description[:max_length].rsplit(' ', 1)[0]
                    return short_desc + "..."
                return description
            
            # Tạo danh sách sản phẩm dạng văn bản
            products_text = ""
            for i, product in enumerate(truncated_products, 1):
                product_id = product.get("productId", "") or product.get("product_id", "")
                name = product.get("productName", "") or product.get("name", "Không có tên")
                price = product.get("price", 0)
                price_display = product.get("price_display", f"{price:,}đ".replace(",", "."))
                unit = product.get("unit", "")
                seller = product.get("sellerName", "Không có thông tin")
                description = product.get("description", "")
                
                # Sử dụng URL từ API nếu có, nếu không thì tạo URL giả
                product_url = product.get("url_sanpham", "")
                if not product_url and product_id:
                    product_url = f"/products/detail/{product_id}"
                elif not product_url:
                    product_url = "#"
                
                products_text += f"Sản phẩm {i}: {name}\n"
                products_text += f"ID: {product_id}\n"
                products_text += f"URL: {product_url}\n"
                products_text += f"Giá: {price_display}/{unit}\n"
                products_text += f"Người bán: {seller}\n"
                
                # Thêm mô tả ngắn gọn nếu có
                if description:
                    short_desc = get_short_description(description, 150)
                    products_text += f"Mô tả: {short_desc}\n"
                
                products_text += "\n"
            
            system_prompt = ""
            if is_product_list_query or is_price_sort or is_price_filter:
              system_prompt = f"""
                Dưới đây là danh sách sản phẩm đã được sắp xếp theo giá từ thấp đến cao. Hãy trình bày các sản phẩm này một cách rõ ràng, trực quan, và đẹp mắt bằng cách tổ chức theo các nhóm giá sau:

                1. 💰 Nhóm giá rẻ (dưới 100.000đ)
                2. 💸 Nhóm giá trung bình (100.000đ - 500.000đ)
                3. 💎 Nhóm giá cao (trên 500.000đ)

                Với mỗi sản phẩm, hãy trình bày theo mẫu:
                <a href="URL_SẢN_PHẨM">TÊN_SẢN_PHẨM</a>: GIÁ/ĐƠN_VỊ (NGƯỜI_BÁN)

                Yêu cầu:
                - Các sản phẩm trong mỗi nhóm phải đúng thứ tự giá tăng dần
                - Trình bày dễ đọc, có thể sử dụng dấu đầu dòng hoặc khoảng cách hợp lý giữa các sản phẩm
                - Chỉ thêm mô tả nếu cần thiết (ví dụ: để so sánh những sản phẩm tương tự nhau)
                - Không cần nêu giá thấp nhất hay cao nhất
                - Tập trung vào tính rõ ràng và trải nghiệm người xem
                - Nếu người dùng yêu cầu sắp xếp lại (theo tên, người bán, v.v...), hãy làm theo

                Danh sách sản phẩm:
                {products_text}
                """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            # Thử dùng Gemini
            try:
                response = await gemini_service.chat(
                    messages=messages,
                    temperature=0.7,
                    max_tokens=6000 if is_product_list_query else 2000
                )
                return response["answer"]
            except Exception as e:
                logger.warning(f"Lỗi khi dùng Gemini: {str(e)}")
                
                # Nếu Gemini lỗi, thử dùng OpenAI
                try:
                    response = await openai_service.chat(
                        messages=messages,
                        temperature=0.7,
                        max_tokens=6000 if is_product_list_query else 2000
                    )
                    return response["answer"]
                except Exception as e:
                    logger.warning(f"Lỗi khi dùng OpenAI: {str(e)}")
            
            # Nếu cả hai đều lỗi, trả về danh sách định dạng đơn giản
            if is_product_list_query or is_price_sort or is_price_filter:
                # Danh sách sản phẩm đẹp hơn cho câu hỏi về danh sách
                result = f"## Danh sách {len(products)} sản phẩm"
                
                # Thêm thông tin lọc giá nếu có
                if is_price_filter:
                    if min_price is not None and max_price is not None:
                        result += f" (giá từ {min_price:,}đ đến {max_price:,}đ)".replace(",", ".")
                    elif min_price is not None:
                        result += f" (giá từ {min_price:,}đ trở lên)".replace(",", ".")
                    elif max_price is not None:
                        result += f" (giá dưới {max_price:,}đ)".replace(",", ".")
                
                result += f"\n\n"
                
                # Phân nhóm sản phẩm theo giá
                low_price = []
                mid_price = []
                high_price = []
                
                # Xác định ngưỡng giá
                for product in truncated_products:
                    price = float(product.get("price", 0))
                    if price < 100000:
                        low_price.append(product)
                    elif price <= 500000:
                        mid_price.append(product)
                    else:
                        high_price.append(product)
                
                # Hiển thị sản phẩm theo nhóm giá
                if low_price:
                    result += "**💰 Nhóm giá rẻ (dưới 100.000đ)**\n\n"
                    for i, product in enumerate(low_price, 1):
                        name = product.get("productName", "") or product.get("name", "Không có tên")
                        price = product.get("price", 0)
                        price_display = product.get("price_display", f"{price:,}đ".replace(",", "."))
                        unit = product.get("unit", "")
                        seller = product.get("sellerName", "Không có thông tin")
                        description = product.get("description", "")
                        
                        # Sử dụng URL từ API nếu có
                        product_url = product.get("url_sanpham", "")
                        if not product_url:
                            product_id = product.get("productId", "") or product.get("product_id", "")
                            if product_id:
                                product_url = f"/products/detail/{product_id}"
                            else:
                                product_url = "#"
                        
                        result += f"* <a href=\"{product_url}\">{name}</a>: {price_display}/{unit} ({seller})\n"
                        if description:
                            short_desc = get_short_description(description, 100)
                            result += f"  {short_desc}\n"
                    
                    result += "\n"
                
                if mid_price:
                    result += "**💸 Nhóm giá trung bình (100.000đ - 500.000đ)**\n\n"
                    for i, product in enumerate(mid_price, 1):
                        name = product.get("productName", "") or product.get("name", "Không có tên")
                        price = product.get("price", 0)
                        price_display = product.get("price_display", f"{price:,}đ".replace(",", "."))
                        unit = product.get("unit", "")
                        seller = product.get("sellerName", "Không có thông tin") 
                        description = product.get("description", "")
                        
                        # Sử dụng URL từ API nếu có
                        product_url = product.get("url_sanpham", "")
                        if not product_url:
                            product_id = product.get("productId", "") or product.get("product_id", "")
                            if product_id:
                                product_url = f"/products/detail/{product_id}"
                            else:
                                product_url = "#"
                        
                        result += f"* <a href=\"{product_url}\">{name}</a>: {price_display}/{unit} ({seller})\n"
                        if description:
                            short_desc = get_short_description(description, 100)
                            result += f"  {short_desc}\n"
                    
                    result += "\n"
                
                if high_price:
                    result += "**💎 Nhóm giá cao (trên 500.000đ)**\n\n"
                    for i, product in enumerate(high_price, 1):
                        name = product.get("productName", "") or product.get("name", "Không có tên")
                        price = product.get("price", 0)
                        price_display = product.get("price_display", f"{price:,}đ".replace(",", "."))
                        unit = product.get("unit", "")
                        seller = product.get("sellerName", "Không có thông tin")
                        description = product.get("description", "")
                        
                        # Sử dụng URL từ API nếu có
                        product_url = product.get("url_sanpham", "")
                        if not product_url:
                            product_id = product.get("productId", "") or product.get("product_id", "")
                            if product_id:
                                product_url = f"/products/detail/{product_id}"
                            else:
                                product_url = "#"
                        
                        result += f"* <a href=\"{product_url}\">{name}</a>: {price_display}/{unit} ({seller})\n"
                        if description:
                            short_desc = get_short_description(description, 100)
                            result += f"  {short_desc}\n"
            else:
                # Format chuẩn cho câu hỏi thông thường
                result = f"Tìm thấy {len(products)} sản phẩm:\n\n"
                for i, product in enumerate(truncated_products, 1):
                    product_id = product.get("productId", "") or product.get("product_id", "")
                    name = product.get("productName", "") or product.get("name", "Không có tên")
                    price_display = product.get("price_display", "")
                    unit = product.get("unit", "")
                    
                    # Sử dụng URL từ API nếu có
                    product_url = product.get("url_sanpham", "")
                    if not product_url and product_id:
                        product_url = f"/products/detail/{product_id}"
                    elif not product_url:
                        product_url = "#"
                    
                    result += f"{i}. <a href=\"{product_url}\">{name}</a>: {price_display}/{unit}\n"
            
            return result
        except Exception as e:
            logger.error(f"Lỗi khi phân tích sản phẩm: {str(e)}")
            return f"Đã xảy ra lỗi khi phân tích sản phẩm: {str(e)}"
    
    async def process_product_query(self, query: str) -> str:
        """
        Xử lý câu hỏi liên quan đến sản phẩm
        
        Args:
            query: Câu hỏi của người dùng
            
        Returns:
            Kết quả trả lời
        """
        try:
            query_lower = query.lower()
            logger.info(f"Xử lý câu hỏi sản phẩm: '{query_lower}'")
            
            # Kiểm tra nếu câu hỏi chỉ về danh mục (không liên quan đến sản phẩm)
            category_only_patterns = [
                r"^danh mục$",
                r"^danh sách danh mục$",
                r"^các danh mục$",
                r"^xem danh mục$", 
                r"^hiển thị danh mục$"
            ]
            
            if any(re.match(pattern, query_lower.strip()) for pattern in category_only_patterns):
                logger.info(f"Câu hỏi '{query}' là về danh mục, chuyển sang process_category_query")
                return await self.process_category_query(query)
            
            # Kiểm tra các yêu cầu tổng quan về sản phẩm/danh mục
            general_catalog_patterns = [
                r"^danh mục sản phẩm$", 
                r"^danh sách sản phẩm$",
                r"^xem sản phẩm$",
                r"^tất cả sản phẩm$"
            ]
            
            # Nếu là yêu cầu về danh mục/danh sách sản phẩm
            for pattern in general_catalog_patterns:
                if re.search(pattern, query_lower):
                    logger.info(f"Phát hiện yêu cầu xem danh mục hoặc danh sách sản phẩm: {query}")
                    return await self.process_category_query(query_lower)
            
            # Kiểm tra nếu câu hỏi là về khoảng giá
            price_range_patterns = [
                r"sản phẩm (giá|có giá) (dưới|từ|trên|trong khoảng)",
                r"sản phẩm (rẻ|đắt|mắc|cao)",
                r"(dưới|từ|trên) \d+[k]?",
                r"(tìm|có) sản phẩm (giá|khoảng giá)",
                r"danh sách sản phẩm (giá|có giá)",
                r"^sản phẩm giá (rẻ|thấp|cao|đắt)",
                r"^(giá rẻ|giá tốt|giá cao|rẻ nhất|đắt nhất)"
            ]
            
            for pattern in price_range_patterns:
                if re.search(pattern, query_lower):
                    logger.info(f"Phát hiện câu hỏi về khoảng giá: {query}")
                    return await self.process_price_range_query(query)
            
            # Xử lý trường hợp đặc biệt: "sản phẩm giá rẻ"
            if "sản phẩm giá rẻ" in query_lower or "sản phẩm rẻ" in query_lower or query_lower == "giá rẻ":
                logger.info(f"Phát hiện yêu cầu về sản phẩm giá rẻ: {query}")
                # Tạo một khoảng giá mặc định cho sản phẩm giá rẻ (0 - 100,000đ)
                return await self.format_price_range_products(
                    min_price=0, 
                    max_price=100000, 
                    query="sản phẩm giá rẻ dưới 100,000đ"
                )
            
            # Tiếp tục xử lý câu hỏi chi tiết về một sản phẩm cụ thể
            specific_product_patterns = [
                r"thông tin (về|chi tiết|) (.*)",
                r"chi tiết (về|) (.*)",
                r"sản phẩm (.*) (như thế nào|ra sao|thế nào)",
                r"(.*) có (gì|những gì)",
                r"(.*) (giá|giá bao nhiêu|bao nhiêu tiền)"
            ]
            
            product_name = None
            
            # Kiểm tra xem câu hỏi có phải là về chi tiết sản phẩm không
            for pattern in specific_product_patterns:
                match = re.search(pattern, query_lower)
                if match and len(match.groups()) >= 1:
                    # Nhóm cuối cùng trong match thường là tên sản phẩm
                    product_name = match.groups()[-1].strip()
                    # Kiểm tra xem product_name có phải một từ khóa chung không
                    if product_name in ["sản phẩm", "hàng hóa", "danh mục", "danh sách"]:
                        logger.info(f"Phát hiện tên sản phẩm là từ khóa chung, xử lý như yêu cầu chung: {product_name}")
                        return await self.get_products_from_all_categories()
                    # Kiểm tra nếu tên sản phẩm chứa từ khóa liên quan đến giá
                    if any(keyword in product_name for keyword in ["giá rẻ", "giá thấp", "giá thấp", "rẻ nhất", "rẻ tiền"]):
                        logger.info(f"Phát hiện yêu cầu về sản phẩm giá rẻ trong tên sản phẩm: {product_name}")
                        return await self.format_price_range_products(
                            min_price=0, 
                            max_price=100000, 
                            query="sản phẩm giá rẻ dưới 100,000đ"
                        )
                    logger.info(f"Phát hiện yêu cầu thông tin chi tiết về sản phẩm: {product_name}")
                    break
            
            # Kiểm tra xem có phải câu hỏi tổng quan về danh mục sản phẩm
            category_keywords = ["danh mục", "danh sách", "các loại", "tất cả"] 
            if any(keyword in query_lower for keyword in category_keywords) and "sản phẩm" in query_lower:
                logger.info(f"Phát hiện yêu cầu về danh mục/danh sách sản phẩm: {query}")
                return await self.get_products_from_all_categories()
            
            # Nếu không phát hiện tên sản phẩm từ pattern, thử tách trực tiếp từ câu hỏi
            if not product_name and len(query_lower.split()) <= 5:
                # Nếu câu hỏi ngắn (ít từ), có thể người dùng chỉ nhập tên sản phẩm
                exclude_words = ["sản phẩm", "hàng hóa", "danh mục", "thông tin", "về", "chi tiết"]
                potential_name = query_lower
                
                # Loại bỏ các từ khóa không cần thiết
                for word in exclude_words:
                    potential_name = potential_name.replace(word, "").strip()
                
                if potential_name and len(potential_name) >= 3:
                    logger.info(f"Trích xuất tên sản phẩm trực tiếp từ câu hỏi ngắn: {potential_name}")
                    product_name = potential_name
            
            # Nếu vẫn không tìm được tên sản phẩm, dùng AI để trích xuất
            if not product_name:
                product_name = await self.extract_product_name(query)
                # Kiểm tra nếu tên sản phẩm trích xuất là từ khóa chung
                if product_name in ["sản phẩm", "hàng hóa", "danh mục", "danh sách", "gạo"]:
                    # Nếu query chỉ có từ khóa chung, xử lý như yêu cầu danh sách
                    if len(query_lower.split()) <= 3:
                        logger.info(f"Phát hiện query đơn giản với từ khóa chung: {query}")
                        return await self.get_products_from_all_categories()
                
                if product_name:
                    logger.info(f"Đã trích xuất tên sản phẩm từ câu hỏi bằng AI: {product_name}")
            
            # Kiểm tra nếu sản phẩm trích xuất là "giá", thì có thể đây là yêu cầu về sản phẩm giá rẻ
            if product_name in ["giá", "giá rẻ", "rẻ"]:
                logger.info(f"Phát hiện yêu cầu về sản phẩm giá rẻ từ tên sản phẩm '{product_name}'")
                return await self.format_price_range_products(
                    min_price=0, 
                    max_price=100000, 
                    query="sản phẩm giá rẻ dưới 100,000đ"
                )
            
            # Nếu phát hiện tên sản phẩm, lấy thông tin chi tiết
            if product_name:
                # Chuẩn bị từ khóa tìm kiếm, loại bỏ các từ không cần thiết
                search_product_name = product_name.lower()
                common_words = ["thông tin", "về", "chi tiết", "cho"]
                for word in common_words:
                    search_product_name = search_product_name.replace(word, "").strip()
                
                logger.info(f"Tìm kiếm sản phẩm với từ khóa: '{search_product_name}'")
                products_result = await self.get_products(search_product_name)
                
                # Kiểm tra cấu trúc kết quả
                products = []
                if isinstance(products_result, dict) and "data" in products_result:
                    products = products_result.get("data", [])
                elif isinstance(products_result, list):
                    products = products_result
                
                if products and len(products) > 0:
                    # Thử tìm kiếm chính xác sản phẩm trong danh sách kết quả
                    exact_match = None
                    for product in products:
                        name = product.get("productName", product.get("name", "")).lower()
                        # Nếu tên sản phẩm chứa đầy đủ từ khóa tìm kiếm
                        if search_product_name in name:
                            logger.info(f"Tìm thấy sản phẩm phù hợp: {name}")
                            exact_match = product
                            break
                    
                    # Nếu tìm thấy sản phẩm khớp chính xác, hiển thị chi tiết
                    if exact_match:
                        logger.info(f"Hiển thị chi tiết sản phẩm: {exact_match.get('name', '')}")
                        return self.format_product_detail(exact_match)
                    
                    # Nếu chỉ có một sản phẩm trong kết quả, hiển thị luôn
                    if len(products) == 1:
                        logger.info(f"Chỉ tìm thấy một sản phẩm, hiển thị chi tiết: {products[0].get('name', '')}")
                        return self.format_product_detail(products[0])
                    
                    # Nếu có nhiều sản phẩm, hiển thị danh sách
                    products_text = ""
                    for i, product in enumerate(products[:10], 1):
                        name = product.get("productName", product.get("name", "Không có tên"))
                        price = product.get("price", "Không có giá")
                        seller = product.get("seller", product.get("sellerName", "Không có thông tin"))
                        
                        # Định dạng giá
                        try:
                            price_formatted = f"{int(float(price)):,}đ" if price and price != "Không có giá" else price
                        except:
                            price_formatted = f"{price}"
                            
                        product_url = product.get("productUrl", "#")
                        
                        products_text += f"{i}. <a href='{product_url}'>{name}</a>: {price_formatted} (Người bán: {seller})\n"
                    
                    return f"""Tôi tìm thấy một số sản phẩm liên quan đến '{product_name}':\n\n{products_text}\n\nHãy hỏi chi tiết về một sản phẩm cụ thể, ví dụ: "thông tin về {products[0].get('name', 'sản phẩm 1')}"."""
                else:
                    # Nếu product_name là một từ khóa chung như "danh mục", "danh sách"
                    if product_name in ["danh mục", "danh sách", "sản phẩm"]:
                        logger.info(f"Không tìm thấy sản phẩm với từ khóa '{product_name}', xử lý như yêu cầu chung")
                        return await self.get_products_from_all_categories()
                    
                    return f"Không tìm thấy thông tin về sản phẩm '{product_name}'. Vui lòng thử lại với tên sản phẩm khác."
            
            # Nếu không tìm được tên sản phẩm sau tất cả các bước
            logger.warning(f"Không xác định được tên sản phẩm từ câu hỏi: {query}")
            return "Xin lỗi, tôi không hiểu bạn đang hỏi về sản phẩm nào. Vui lòng cung cấp tên sản phẩm cụ thể."
            
        except Exception as e:
            logger.error(f"Error processing product query: {str(e)}", exc_info=True)
            return f"Đã xảy ra lỗi khi xử lý câu hỏi của bạn: {str(e)}"
    
    async def get_subcategories(self, category_id: int) -> List[Dict[str, Any]]:
        """
        Lấy danh sách danh mục con từ API
        
        Args:
            category_id: ID của danh mục cha
            
        Returns:
            Danh sách các danh mục con
        """
        try:
            # Kiểm tra category_id
            if not category_id or not str(category_id).isdigit():
                logger.warning(f"ID danh mục không hợp lệ: {category_id}")
                return []
            
            # URL chính xác để lấy danh mục con
            url = f"https://chodongbao.com/api/Categories/20?category_id={category_id}"
            logger.info(f"Gọi API danh mục con: {url}")
            
            async with httpx.AsyncClient() as client:
                try:
                    # Thêm headers cần thiết
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        "Accept": "application/json",
                        "authenticatetoken": "ChoDongBao_HueCIT"
                    }
                    
                    response = await client.get(url, timeout=10.0, headers=headers)
                    
                    # Ghi log đầy đủ phản hồi để kiểm tra
                    logger.info(f"API trả về status: {response.status_code}")
                    logger.info(f"API trả về nội dung: {response.text[:200]}...")
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Xử lý cả hai trường hợp: kết quả là list hoặc object có field "data"
                        if isinstance(result, list):
                            logger.info(f"API trả về danh sách trực tiếp với {len(result)} danh mục con")
                            subcategories = result
                        elif isinstance(result, dict) and "data" in result:
                            logger.info(f"API trả về object có trường data với {len(result.get('data', []))} danh mục con")
                            subcategories = result.get("data", [])
                        else:
                            logger.warning(f"API trả về định dạng không mong đợi: {type(result)}")
                            subcategories = []
                        
                        # Nếu danh sách con trả về rỗng, điều này bình thường
                        if not subcategories:
                            logger.info(f"Danh mục {category_id} không có danh mục con")
                        else:
                            logger.info(f"Tìm thấy {len(subcategories)} danh mục con cho category_id={category_id}")
                        
                        return subcategories
                    else:
                        logger.error(f"API danh mục con trả về lỗi: {response.status_code}")
                        # Sử dụng dữ liệu mẫu nếu API thất bại
                        return self._get_sample_subcategories(category_id)
                except Exception as e:
                    logger.error(f"Lỗi khi gọi API danh mục con: {str(e)}")
                    return self._get_sample_subcategories(category_id)
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh mục con: {str(e)}")
            return []
            
    def _get_sample_subcategories(self, category_id: int) -> List[Dict[str, Any]]:
        """
        Trả về dữ liệu mẫu cho danh mục con khi API thất bại
        
        Args:
            category_id: ID của danh mục cha
            
        Returns:
            Danh sách các danh mục con mẫu
        """
        # Dữ liệu mẫu cho một số danh mục phổ biến
        sample_data = {
            # Thủ công mỹ nghệ (giả sử ID = 3)
            3: [
                {"category_id": 31, "id": 31, "name": "Mây tre đan", "description": "Các sản phẩm làm từ mây tre đan"},
                {"category_id": 32, "id": 32, "name": "Gốm sứ thủ công", "description": "Các sản phẩm gốm sứ làm thủ công"},
                {"category_id": 33, "id": 33, "name": "Đồ thêu", "description": "Các sản phẩm thêu thủ công"},
            ],
            # Thổ cẩm (giả sử ID = 4)
            4: [
                {"category_id": 41, "id": 41, "name": "Thổ cẩm Tây Nguyên", "description": "Thổ cẩm của các dân tộc Tây Nguyên"},
                {"category_id": 42, "id": 42, "name": "Thổ cẩm vùng cao", "description": "Thổ cẩm của đồng bào dân tộc vùng núi phía Bắc"},
            ],
            # Gạo các loại (giả sử ID = 2)
            2: [
                {"category_id": 21, "id": 21, "name": "Gạo tẻ", "description": "Các loại gạo tẻ phổ biến"},
                {"category_id": 22, "id": 22, "name": "Gạo nếp", "description": "Các loại gạo nếp phổ biến"},
                {"category_id": 23, "id": 23, "name": "Gạo đặc sản", "description": "Các loại gạo đặc sản quý hiếm"},
            ],
        }
        
        # Nếu có dữ liệu mẫu cho category_id, trả về dữ liệu đó
        if category_id in sample_data:
            logger.info(f"Sử dụng dữ liệu mẫu cho danh mục con của category_id={category_id}")
            return sample_data[category_id]
            
        # Nếu không có dữ liệu mẫu, trả về danh sách rỗng
        return []

    async def process_category_query(self, query: str) -> str:
        """
        Xử lý câu hỏi về danh mục sản phẩm
        
        Args:
            query: Câu hỏi người dùng
            
        Returns:
            Kết quả xử lý
        """
        try:
            query_lower = query.lower()
            
            # Kiểm tra nếu người dùng chỉ hỏi danh sách các danh mục
            category_list_patterns = [
                "danh mục", "category", "loại sản phẩm", "phân loại", 
                "các loại", "nhóm sản phẩm", "danh sách danh mục"
            ]
            
            is_category_list_query = any(pattern in query_lower for pattern in category_list_patterns)
            
            # Kiểm tra nếu người dùng hỏi danh sách sản phẩm
            product_list_patterns = [
                "danh sách sản phẩm", "tất cả sản phẩm", "toàn bộ sản phẩm",
                "các sản phẩm", "liệt kê sản phẩm", "hiển thị sản phẩm",
                "xem sản phẩm", "sản phẩm có", "sản phẩm bán"
            ]
            
            is_product_list_query = any(pattern in query_lower for pattern in product_list_patterns)
            
            # Nếu là truy vấn về danh sách sản phẩm, xử lý theo logic riêng
            if is_product_list_query and not is_category_list_query:
                logger.info("Người dùng yêu cầu xem danh sách sản phẩm")
                try:
                    return await self.get_products_from_all_categories()
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý câu hỏi danh mục: {str(e)}")
                    return f"Đã xảy ra lỗi khi hiển thị danh sách sản phẩm: {str(e)}"
            
            # Nếu người dùng yêu cầu danh sách danh mục, hiển thị danh sách danh mục
            if is_category_list_query and not "sản phẩm" in query_lower:
                logger.info("Người dùng yêu cầu xem danh sách danh mục")
                categories = await self.get_all_categories()
                if not categories:
                    return "Hiện không có danh mục sản phẩm nào trong hệ thống."
                return self.format_categories_list(categories)
            
            # Nếu người dùng tìm sản phẩm theo danh mục, xử lý theo danh mục
            logger.info("Người dùng tìm sản phẩm theo danh mục")
            return await self.find_products_by_category(query)
        except Exception as e:
            logger.error(f"Lỗi xử lý câu hỏi danh mục: {str(e)}")
            return f"Đã xảy ra lỗi khi xử lý câu hỏi: {str(e)}"

    async def get_all_categories(self) -> List[Dict[str, Any]]:
        """
        Lấy danh sách tất cả các danh mục cha
        
        Returns:
            Danh sách các danh mục cha
        """
        try:
            # URL chính xác để lấy danh mục cha
            url = "https://chodongbao.com/api/Categories/20"
            logger.info(f"Gọi API danh mục cha: {url}")
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json",
                "authenticatetoken": "ChoDongBao_HueCIT"
            }
            
            categories = []
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, timeout=10.0, headers=headers)
                    
                    logger.info(f"Phản hồi API danh mục cha: HTTP/1.1 {response.status_code}")
                    
                    if response.status_code == 200:
                        # Phân tích response JSON
                        try:
                            response_data = response.json()
                            
                            # Kiểm tra nếu response là list trực tiếp
                            if isinstance(response_data, list):
                                logger.info(f"API trả về danh sách trực tiếp với {len(response_data)} mục")
                                # Lọc chỉ lấy danh mục cha
                                categories = [cat for cat in response_data if not cat.get("is_subcategory", False)]
                                logger.info(f"Đã lọc được {len(categories)} danh mục cha")
                            # Nếu là object có trường data
                            elif isinstance(response_data, dict) and "data" in response_data:
                                logger.info(f"API trả về object có trường data")
                                # Lọc chỉ lấy danh mục cha từ data
                                all_categories = response_data.get("data", [])
                                categories = [cat for cat in all_categories if not cat.get("is_subcategory", False)]
                                logger.info(f"Đã lọc được {len(categories)} danh mục cha từ {len(all_categories)} danh mục")
                            else:
                                logger.warning(f"API trả về định dạng không mong đợi: {type(response_data)}")
                                categories = []
                        except ValueError as e:
                            logger.error(f"Lỗi phân tích JSON từ API: {str(e)}")
                            categories = []
                    else:
                        logger.error(f"Lỗi khi gọi API danh mục: {response.status_code}")
                        categories = self._get_sample_categories()
            except Exception as e:
                logger.error(f"Lỗi khi gọi API danh mục: {str(e)}")
                categories = self._get_sample_categories()
            
            return categories
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách danh mục: {str(e)}")
            return self._get_sample_categories()
    
    def _get_sample_categories(self) -> List[Dict[str, Any]]:
        """
        Cung cấp danh sách danh mục mẫu khi API không hoạt động
        
        Returns:
            Danh sách danh mục mẫu
        """
        return [
            {"id": 1, "category_id": 1, "name": "Nông nghiệp", "description": "Sản phẩm nông nghiệp"},
            {"id": 2, "category_id": 2, "name": "Gạo & Lương thực", "description": "Các loại gạo và lương thực"},
            {"id": 3, "category_id": 3, "name": "Thủ công mỹ nghệ", "description": "Sản phẩm thủ công mỹ nghệ"},
            {"id": 4, "category_id": 4, "name": "Thổ cẩm & Dệt may", "description": "Sản phẩm thổ cẩm và dệt may"},
            {"id": 5, "category_id": 5, "name": "Đặc sản địa phương", "description": "Đặc sản của các địa phương"}
        ]
    
    def format_categories_list(self, categories: List[Dict[str, Any]]) -> str:
        """
        Định dạng danh sách danh mục thành HTML
        
        Args:
            categories: Danh sách danh mục
            
        Returns:
            Chuỗi HTML hiển thị danh sách danh mục
        """
        try:
            if not categories:
                return "Không tìm thấy danh mục sản phẩm nào."
            
            html_result = """
            <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto;">
                <h2 style="color: #2a5885; margin-bottom: 20px;">Danh sách các danh mục sản phẩm</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 20px;">
            """
            
            for category in categories:
                category_id = category.get("id") or category.get("category_id", "")
                category_name = category.get("name", "Không có tên")
                category_desc = category.get("description", "")
                
                # Tạo URL danh mục
                category_url = f"https://chodongbao.com/category/{category_id}"
                
                html_result += f"""
                    <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                        <h3 style="margin-top: 0; margin-bottom: 10px; font-size: 18px;">
                            <a href="{category_url}" style="color: #2b72c2; text-decoration: none;">{category_name}</a>
                        </h3>
                        <p style="color: #666; margin-bottom: 10px; font-size: 14px;">{category_desc}</p>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="color: #888; font-size: 13px;">ID: {category_id}</span>
                            <a href="{category_url}" style="display: inline-block; padding: 5px 10px; background: #4CAF50; color: white; text-decoration: none; border-radius: 4px; font-size: 13px;">Xem sản phẩm</a>
                        </div>
                    </div>
                """
            
            html_result += """
                </div>
            </div>
            """
            
            return html_result
        except Exception as e:
            logger.error(f"Lỗi khi định dạng danh sách danh mục: {str(e)}")
            
            # Fallback to simple format
            simple_html = "<h2>Danh sách danh mục sản phẩm:</h2><ul>"
            for category in categories:
                category_id = category.get("id") or category.get("category_id", "")
                category_name = category.get("name", "Không có tên")
                simple_html += f'<li>ID: {category_id} - {category_name}</li>'
            simple_html += "</ul>"
            
            return simple_html
    
    async def format_categories_with_hierarchy(self, categories_data: Dict[str, Any]) -> str:
        """
        Định dạng danh sách danh mục theo cấu trúc phân cấp
        
        Args:
            categories_data: Dữ liệu danh mục từ API
            
        Returns:
            Chuỗi HTML định dạng danh mục
        """
        try:
            if not categories_data.get("success", False) or not categories_data.get("data", []):
                return "Không tìm thấy danh mục sản phẩm nào."
            
            categories = categories_data.get("data", [])
            
            # Tách danh mục cha và danh mục con
            parent_categories = []
            subcategories_map = {}  # Map từ parent_id đến danh sách danh mục con
            
            for cat in categories:
                if cat.get("is_subcategory", False):
                    parent_id = cat.get("parent_id", "")
                    if parent_id not in subcategories_map:
                        subcategories_map[parent_id] = []
                    subcategories_map[parent_id].append(cat)
                else:
                    parent_categories.append(cat)
            
            # Tạo HTML
            result = f"## Danh sách {len(parent_categories)} danh mục sản phẩm\n\n"
            
            for cat in parent_categories:
                cat_id = cat.get("category_id", cat.get("id", ""))
                cat_name = cat.get("name", "Không có tên")
                cat_desc = cat.get("description", "")
                
                result += f"### {cat_name}\n"
                if cat_desc:
                    result += f"{cat_desc}\n"
                
                # Thêm thông tin về số lượng danh mục con
                subcategories = subcategories_map.get(cat_id, [])
                if subcategories:
                    result += f"\n**Danh mục con ({len(subcategories)}):**\n"
                    for subcat in subcategories:
                        subcat_id = subcat.get("category_id", subcat.get("id", ""))
                        subcat_name = subcat.get("name", "Không có tên")
                        subcat_desc = subcat.get("description", "")
                        
                        result += f"- **{subcat_name}** (ID: {subcat_id})"
                        if subcat_desc:
                            result += f": {subcat_desc}"
                        result += "\n"
                
                result += "\n"
            
            return result
        except Exception as e:
            logger.error(f"Lỗi khi định dạng danh mục: {str(e)}")
            # Fallback vào định dạng cũ
            try:
                from app.api.query_demo.product_api import format_categories
                return format_categories(categories_data)
            except:
                return f"Đã xảy ra lỗi khi định dạng danh mục: {str(e)}"

    async def find_products_by_category(self, query: str) -> str:
        """
        Tìm sản phẩm theo danh mục
        
        Args:
            query: Câu hỏi của người dùng
            
        Returns:
            Kết quả tìm kiếm
        """
        try:
            # Tìm tên danh mục từ câu hỏi
            query_lower = query.lower()
            category_name = None
            
            # Từ khóa đặc biệt cần bỏ qua khi xác định tên danh mục
            special_keywords = ["từng", "mọi", "tất cả", "các"]
            
            # Danh sách từ khóa danh mục có thể nhận dạng
            category_keywords = ["thủ công mỹ nghệ", "thổ cẩm", "gạo", "thực phẩm", "đặc sản", "gạo các loại"]
            
            # Ưu tiên tìm các từ khóa danh mục cụ thể trong câu hỏi
            for keyword in category_keywords:
                if keyword in query_lower:
                    category_name = keyword
                    logger.info(f"Tìm thấy danh mục từ từ khóa: {category_name}")
                    break
            
            # Nếu không tìm thấy từ khóa danh mục, thử phân tích cấu trúc câu
            if not category_name:
                # Các mẫu phổ biến để tìm tên danh mục
                patterns = [
                    r"sản phẩm của (.*?)(?:$|\s+|\?)",
                    r"sản phẩm thuộc (.*?)(?:$|\s+|\?)",
                    r"sản phẩm trong (.*?)(?:$|\s+|\?)",
                    r"sản phẩm ở (.*?)(?:$|\s+|\?)",
                    r"sản phẩm danh mục (.*?)(?:$|\s+|\?)",
                    r"sản phẩm loại (.*?)(?:$|\s+|\?)",
                    r"danh sách sản phẩm của (.*?)(?:$|\s+|\?)",
                    r"danh sách sản phẩm thuộc (.*?)(?:$|\s+|\?)",
                    r"danh sách sản phẩm trong (.*?)(?:$|\s+|\?)",
                    r"hàng hóa của (.*?)(?:$|\s+|\?)",
                    r"mặt hàng của (.*?)(?:$|\s+|\?)",
                ]
                
                # Kiểm tra từng mẫu
                import re
                for pattern in patterns:
                    matches = re.search(pattern, query_lower)
                    if matches:
                        potential_name = matches.group(1).strip()
                        # Loại bỏ các từ không cần thiết
                        for word in ["sản phẩm", "hàng hóa", "về", "gì", "nào", "các", "những", "danh mục", *special_keywords]:
                            potential_name = potential_name.replace(word, "").strip()
                        if potential_name:
                            category_name = potential_name
                            break
                
                # Nếu vẫn không tìm thấy, thử phân tích đơn giản hơn
                if not category_name:
                    for phrase in ["của", "thuộc", "trong", "danh mục", "loại"]:
                        if phrase in query_lower:
                            # Tìm tên danh mục sau từ khóa
                            parts = query_lower.split(phrase)
                            if len(parts) > 1 and parts[1].strip():
                                potential_name = parts[1].strip()
                                # Loại bỏ các từ không cần thiết
                                for word in ["sản phẩm", "hàng hóa", "về", "gì", "nào", "các", "những", *special_keywords]:
                                    potential_name = potential_name.replace(word, "").strip()
                                if potential_name:
                                    category_name = potential_name
                                    break
            
            # Kiểm tra lại nếu category_name là một trong các từ khóa đặc biệt
            if category_name in special_keywords:
                return await self.get_products_from_all_categories()
                
            if not category_name:
                return "Vui lòng nêu rõ danh mục cần tìm sản phẩm. Ví dụ: 'Tìm sản phẩm thuộc danh mục thủ công mỹ nghệ'."
            
            logger.info(f"Đang tìm sản phẩm cho danh mục: {category_name}")
            
            # Lấy danh sách danh mục từ API
            all_categories = await self.get_all_categories()
            
            # Cập nhật danh sách với danh mục con
            extended_categories = list(all_categories)  # Tạo bản sao
            for cat in all_categories:
                cat_id = cat.get("category_id", cat.get("id"))
                if cat_id:
                    subcategories = await self.get_subcategories(cat_id)
                    extended_categories.extend(subcategories)
            
            # Tìm category_id từ tên danh mục
            category_id = None
            category_exact_name = None
            
            # Trước tiên, tìm kiếm trùng khớp chính xác
            for cat in extended_categories:
                cat_name = cat.get("name", "").lower()
                if category_name.lower() == cat_name:
                    category_id = cat.get("category_id", cat.get("id"))
                    category_exact_name = cat.get("name")
                    break
            
            # Nếu không tìm thấy trùng khớp chính xác, tìm kiếm trùng khớp một phần
            if not category_id:
                for cat in extended_categories:
                    cat_name = cat.get("name", "").lower()
                    if category_name.lower() in cat_name or cat_name in category_name.lower():
                        category_id = cat.get("category_id", cat.get("id"))
                        category_exact_name = cat.get("name")
                        break
            
            if not category_id:
                # Ghi log chi tiết hơn để debug
                logger.error(f"Không thể tìm thấy category_id cho danh mục: {category_name}")
                
                # Tạm thời hardcode category ID cho trường hợp thủ công mỹ nghệ
                if "thủ công" in category_name or "mỹ nghệ" in category_name:
                    category_id = 3
                    category_exact_name = "Thủ công mỹ nghệ"
                elif "thổ cẩm" in category_name:
                    category_id = 4
                    category_exact_name = "Thổ cẩm"
                elif "gạo" in category_name:
                    category_id = 2
                    category_exact_name = "Gạo các loại"
                else:
                    return f"Không tìm thấy danh mục nào phù hợp với '{category_name}'."
            
            # Lấy sản phẩm theo category_id sử dụng hàm đã được cải thiện
            logger.info(f"Tìm sản phẩm theo category_id: {category_id}")
            products = await self.get_products_by_category_id(category_id)
            
            if not products:
                return f"Không tìm thấy sản phẩm nào thuộc danh mục '{category_exact_name}'."
            
            # Phân tích sản phẩm bằng AI
            analysis = await self.analyze_products_with_ai(products, f"Danh sách sản phẩm thuộc danh mục {category_exact_name}")
            
            # Thêm thông tin về danh mục vào kết quả
            intro = f"## Sản phẩm thuộc danh mục {category_exact_name} (ID: {category_id})\n\n"
            
            return intro + analysis
            
        except Exception as e:
            logger.error(f"Lỗi khi tìm sản phẩm theo danh mục: {str(e)}")
            return f"Đã xảy ra lỗi khi tìm sản phẩm theo danh mục: {str(e)}"

    async def get_products_from_all_categories(self) -> str:
        """
        Lấy sản phẩm từ tất cả các danh mục
        
        Returns:
            Thông tin sản phẩm theo các danh mục
        """
        try:
            logger.info("Lấy sản phẩm từ tất cả các danh mục")
            
            # Lấy danh sách sản phẩm tổng hợp - cần truyền chuỗi rỗng cho product_name
            products_response = await self.get_products("")
            
            # Kiểm tra xem kết quả có phải là từ điển với khóa "data" không
            if isinstance(products_response, dict) and "data" in products_response:
                products = products_response.get("data", [])
                logger.info(f"Tìm thấy {len(products)} sản phẩm từ response.data")
            elif isinstance(products_response, list):
                products = products_response
                logger.info(f"Tìm thấy {len(products)} sản phẩm từ response list")
            else:
                logger.error(f"Định dạng phản hồi API không hợp lệ: {type(products_response)}")
                return "Lỗi khi tải danh sách sản phẩm: Định dạng dữ liệu không hợp lệ."
            
            if not products:
                return "Hiện không có sản phẩm nào trong hệ thống."
            
            # Phân tích sản phẩm bằng AI
            analysis = await self.analyze_products_with_ai(products, "Danh sách tất cả các sản phẩm")
            
            # Thêm tiêu đề
            intro = "## Danh sách tất cả các sản phẩm\n\n"
            
            return intro + analysis
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy sản phẩm từ tất cả các danh mục: {str(e)}")
            return f"Đã xảy ra lỗi khi lấy danh sách sản phẩm: {str(e)}"

    async def get_product_by_id(self, product_id: str) -> Dict[str, Any]:
        """
        Lấy thông tin chi tiết của một sản phẩm dựa trên ID
        
        Args:
            product_id: ID của sản phẩm cần tìm
            
        Returns:
            Thông tin chi tiết về sản phẩm
        """
        try:
            url = f"{self.base_url}/{product_id}"
            logger.info(f"Gọi API chi tiết sản phẩm: {url}")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=30.0)
                if response.status_code == 200:
                    product = response.json()
                    
                    # Đảm bảo sản phẩm có trường price_display
                    if "price" in product and "price_display" not in product:
                        price = product["price"]
                        product["price_display"] = f"{price:,}đ".replace(",", ".")
                    
                    return {"success": True, "data": product}
                else:
                    logger.error(f"Lỗi khi gọi API chi tiết sản phẩm: {response.status_code} - {response.text}")
                    
                    # Tìm trong dữ liệu mẫu nếu API thất bại
                    try:
                        from app.api.query_demo import product_api
                        
                        # Tìm trong dữ liệu mẫu gạo
                        for product in product_api.SAMPLE_RICE_DATA:
                            if product.get("productId") == product_id:
                                # Đảm bảo sản phẩm có trường price_display
                                if "price" in product and "price_display" not in product:
                                    price = product["price"]
                                    product["price_display"] = f"{price:,}đ".replace(",", ".")
                                return {"success": True, "data": product}
                        
                        # Tìm trong dữ liệu mẫu thủ công mỹ nghệ
                        for product in product_api.SAMPLE_HANDCRAFT_DATA:
                            if product.get("productId") == product_id:
                                # Đảm bảo sản phẩm có trường price_display
                                if "price" in product and "price_display" not in product:
                                    price = product["price"]
                                    product["price_display"] = f"{price:,}đ".replace(",", ".")
                                return {"success": True, "data": product}
                    except Exception as e:
                        logger.warning(f"Không thể tìm sản phẩm trong dữ liệu mẫu: {str(e)}")
                    
                    return {"success": False, "message": f"Không tìm thấy sản phẩm với ID: {product_id}"}
        except Exception as e:
            logger.error(f"Lỗi khi lấy chi tiết sản phẩm: {str(e)}")
            return {"success": False, "message": f"Lỗi khi lấy chi tiết sản phẩm: {str(e)}"}

    def format_product_detail(self, product: Dict[str, Any]) -> str:
        """
        Định dạng thông tin chi tiết sản phẩm để hiển thị
        
        Args:
            product: Thông tin chi tiết sản phẩm
            
        Returns:
            Chuỗi HTML định dạng thông tin sản phẩm
        """
        product_name = product.get("productName", product.get("name", "Không có tên"))
        product_id = product.get("productId", product.get("id", product.get("product_id", "")))
        price = product.get("price", 0)
        price_display = product.get("price_display", f"{price:,}đ".replace(",", "."))
        unit = product.get("unit", "")
        seller = product.get("sellerName", product.get("seller_name", "Không có thông tin"))
        description = product.get("description", "Không có mô tả")
        images = product.get("images", [])
        image_url = ""
        
        # Xử lý biến images để lấy URL hình ảnh
        if isinstance(images, list) and images:
            if isinstance(images[0], dict) and "url" in images[0]:
                image_url = images[0]["url"]
            elif isinstance(images[0], str):
                image_url = images[0]
        elif isinstance(images, str):
            image_url = images
            
        category_id = product.get("category_id", product.get("categoryId", ""))
        category_name = product.get("category_name", product.get("categoryName", ""))
        quantity = product.get("quantity", 0)
        
        # Tạo URL sản phẩm
        product_url = product.get("url_sanpham", "")
        if not product_url and product_id:
            product_url = f"https://chodongbao.com/product/{product_id}"
            
        # Xử lý mô tả sản phẩm (loại bỏ các thẻ HTML không cần thiết)
        cleaned_description = description
        if description and ("<" in description or "&nbsp;" in description):
            import re
            # Loại bỏ các thẻ HTML nhưng giữ lại nội dung
            cleaned_description = re.sub(r'<br\s*/?>|<div[^>]*>|</div>|<span[^>]*>|</span>|&nbsp;', ' ', description)
            cleaned_description = re.sub(r'<[^>]*>', '', cleaned_description)
            # Loại bỏ khoảng trắng thừa
            cleaned_description = re.sub(r'\s+', ' ', cleaned_description).strip()
        
        # Tạo HTML hiển thị sản phẩm đẹp hơn
        html_template = f"""
        <div style="margin: 0 auto; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); display: flex; flex-direction: column;">
            <div style="padding: 20px; display: flex; flex-direction: row; flex-wrap: wrap;">
                <!-- Hình ảnh sản phẩm -->
                <div style="flex: 1; min-width: 250px; margin-right: 20px; margin-bottom: 20px;">
                    {f'<img src="{image_url}" alt="{product_name}" style="width: 100%; height: auto; max-height: 300px; object-fit: contain; border-radius: 5px; border: 1px solid #eee;">' if image_url else '<div style="width: 100%; height: 200px; background-color: #f5f5f5; display: flex; align-items: center; justify-content: center; border-radius: 5px;"><span style="color: #999;">Không có hình ảnh</span></div>'}
                </div>
                
                <!-- Thông tin sản phẩm -->
                <div style="flex: 2; min-width: 300px;">
                    <h1 style="margin-top: 0; margin-bottom: 15px; color: #333; font-size: 24px; line-height: 1.3;">{product_name}</h1>
                    
                    <div style="margin-bottom: 15px;">
                        <div style="font-size: 26px; color: #e74c3c; font-weight: bold; margin-bottom: 5px;">
                            {price_display}{f' / {unit}' if unit else ''}
                        </div>
                        <div style="color: #777; font-size: 14px;">
                            {f'Người bán: <span style="color: #333; font-weight: 500;">{seller}</span>' if seller else ''}
                        </div>
                    </div>
                    
                    <div style="margin-top: 20px;">
                        <a href="{product_url}" style="display: inline-block; background-color: #2b72c2; color: white; padding: 10px 20px; text-decoration: none; border-radius: 4px; font-weight: 500; transition: background-color 0.3s;">
                            Xem chi tiết sản phẩm
                        </a>
                    </div>
                    
                    <!-- Thông tin kỹ thuật -->
                    <div style="margin-top: 25px; background-color: #f9f9f9; padding: 15px; border-radius: 5px;">
                        <h3 style="margin-top: 0; font-size: 16px; color: #555; margin-bottom: 10px;">Thông tin sản phẩm</h3>
                        <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                            <tr>
                                <td style="padding: 5px 0; color: #777; width: 40%;">Mã sản phẩm:</td>
                                <td style="padding: 5px 0; color: #333;">{product_id}</td>
                            </tr>
                            {f'<tr><td style="padding: 5px 0; color: #777;">Danh mục:</td><td style="padding: 5px 0; color: #333;">{category_name}</td></tr>' if category_name else ''}
                            {f'<tr><td style="padding: 5px 0; color: #777;">Số lượng:</td><td style="padding: 5px 0; color: #333;">{quantity} {unit}</td></tr>' if quantity else ''}
                        </table>
                    </div>
                </div>
            </div>
            
            <!-- Mô tả sản phẩm -->
            <div style="padding: 20px; border-top: 1px solid #eee;">
                <h3 style="margin-top: 0; font-size: 18px; color: #333; margin-bottom: 15px;">Mô tả sản phẩm</h3>
                <div style="color: #555; line-height: 1.6;">
                    {cleaned_description}
                </div>
            </div>
        </div>
        """
        
        # Đảm bảo loại bỏ tất cả các thẻ <br> có thể còn sót lại
        import re
        result = re.sub(r'<br\s*/?>|\n', '', html_template)
        
        return result

    async def process_price_range_query(self, query: str) -> str:
        """
        Xử lý câu hỏi liên quan đến tìm sản phẩm theo khoảng giá
        
        Args:
            query: Câu hỏi của người dùng
            
        Returns:
            Kết quả trả lời
        """
        try:
            query_lower = query.lower()
            
            # Mặc định giá tối thiểu là 0
            min_price = 0
            max_price = None
            
            # Tìm giá tối đa
            max_price_patterns = [
                r"dưới\s*(\d+)[k\s]*đồng",
                r"dưới\s*(\d+)[k\s]*",
                r"rẻ\s*hơn\s*(\d+)[k\s]*đồng",
                r"rẻ\s*hơn\s*(\d+)[k\s]*",
                r"giá\s*dưới\s*(\d+)[k\s]*đồng",
                r"giá\s*dưới\s*(\d+)[k\s]*",
                r"không\s*quá\s*(\d+)[k\s]*đồng",
                r"không\s*quá\s*(\d+)[k\s]*",
                r"tối\s*đa\s*(\d+)[k\s]*đồng",
                r"tối\s*đa\s*(\d+)[k\s]*",
                r"thấp\s*hơn\s*(\d+)[k\s]*đồng",
                r"thấp\s*hơn\s*(\d+)[k\s]*"
            ]
            
            for pattern in max_price_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    max_price_value = int(match.group(1))
                    # Kiểm tra xem có phải đơn vị là k (nghìn) không
                    if 'k' in pattern.lower():
                        max_price = max_price_value * 1000
                    else:
                        max_price = max_price_value
                    logger.info(f"Tìm thấy giá tối đa: {max_price}")
                    break
            
            # Tìm giá tối thiểu
            min_price_patterns = [
                r"trên\s*(\d+)[k\s]*đồng",
                r"trên\s*(\d+)[k\s]*",
                r"đắt\s*hơn\s*(\d+)[k\s]*đồng",
                r"đắt\s*hơn\s*(\d+)[k\s]*",
                r"giá\s*trên\s*(\d+)[k\s]*đồng",
                r"giá\s*trên\s*(\d+)[k\s]*",
                r"ít\s*nhất\s*(\d+)[k\s]*đồng",
                r"ít\s*nhất\s*(\d+)[k\s]*",
                r"tối\s*thiểu\s*(\d+)[k\s]*đồng",
                r"tối\s*thiểu\s*(\d+)[k\s]*",
                r"cao\s*hơn\s*(\d+)[k\s]*đồng",
                r"cao\s*hơn\s*(\d+)[k\s]*"
            ]
            
            for pattern in min_price_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    min_price_value = int(match.group(1))
                    # Kiểm tra xem có phải đơn vị là k (nghìn) không
                    if 'k' in pattern.lower():
                        min_price = min_price_value * 1000
                    else:
                        min_price = min_price_value
                    logger.info(f"Tìm thấy giá tối thiểu: {min_price}")
                    break
            
            # Tìm cả khoảng giá từ... đến...
            range_patterns = [
                r"từ\s*(\d+)[k\s]*đến\s*(\d+)[k\s]*đồng",
                r"từ\s*(\d+)[k\s]*đến\s*(\d+)[k\s]*",
                r"từ\s*(\d+)\s*đến\s*(\d+)[k\s]*đồng",
                r"từ\s*(\d+)\s*đến\s*(\d+)[k\s]*",
                r"trong\s*khoảng\s*(\d+)[k\s]*đến\s*(\d+)[k\s]*đồng",
                r"trong\s*khoảng\s*(\d+)[k\s]*đến\s*(\d+)[k\s]*",
                r"khoảng\s*(\d+)[k\s]*đến\s*(\d+)[k\s]*đồng",
                r"khoảng\s*(\d+)[k\s]*đến\s*(\d+)[k\s]*",
                r"giá\s*từ\s*(\d+)[k\s]*đến\s*(\d+)[k\s]*đồng",
                r"giá\s*từ\s*(\d+)[k\s]*đến\s*(\d+)[k\s]*"
            ]
            
            for pattern in range_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    min_price_value = int(match.group(1))
                    max_price_value = int(match.group(2))
                    
                    # Kiểm tra xem có phải đơn vị là k (nghìn) không
                    if 'k' in pattern.lower():
                        min_price = min_price_value * 1000
                        max_price = max_price_value * 1000
                    else:
                        min_price = min_price_value
                        max_price = max_price_value
                    
                    logger.info(f"Tìm thấy khoảng giá từ {min_price} đến {max_price}")
                    break
            
            # Nếu không tìm thấy khoảng giá nào, sử dụng mặc định
            if min_price == 0 and max_price is None:
                # Thử tìm các số trong câu hỏi
                numbers = re.findall(r'\d+k?', query_lower)
                if numbers:
                    # Nếu chỉ có một số, giả định là giá tối đa
                    if len(numbers) == 1:
                        value = numbers[0]
                        if 'k' in value:
                            max_price = int(value.replace('k', '')) * 1000
                        else:
                            max_price = int(value)
                    # Nếu có hai số, giả định là khoảng giá
                    elif len(numbers) >= 2:
                        value1 = numbers[0]
                        value2 = numbers[1]
                        
                        min_val = 0
                        max_val = 0
                        
                        if 'k' in value1:
                            min_val = int(value1.replace('k', '')) * 1000
                        else:
                            min_val = int(value1)
                            
                        if 'k' in value2:
                            max_val = int(value2.replace('k', '')) * 1000
                        else:
                            max_val = int(value2)
                        
                        # Sắp xếp lại để min <= max
                        min_price = min(min_val, max_val)
                        max_price = max(min_val, max_val)
            
            # Lấy và định dạng sản phẩm theo khoảng giá
            formatted_products = await self.format_price_range_products(min_price, max_price)
            return formatted_products
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý câu hỏi về khoảng giá: {str(e)}")
            return f"Đã xảy ra lỗi khi tìm sản phẩm theo khoảng giá: {str(e)}"
    
    async def get_products_by_price_range(self, min_price: float, max_price: float) -> list:
        """
        Lấy danh sách sản phẩm theo khoảng giá
        
        Args:
            min_price: Giá tối thiểu
            max_price: Giá tối đa
            
        Returns:
            Danh sách sản phẩm
        """
        try:
            # Xử lý giá tối đa nếu không được chỉ định
            if max_price is None or max_price == float('inf'):
                url = f"{self.api_base_url}/products?minPrice={min_price}"
            else:
                url = f"{self.api_base_url}/products?minPrice={min_price}&maxPrice={max_price}"
            
            # Gọi API để lấy sản phẩm
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                logger.info(f"API response status: {response.status_code}")
                
                # Kiểm tra trạng thái phản hồi
                if response.status_code != 200:
                    logger.warning(f"API trả về lỗi: {response.status_code}")
                    return []
                
                # Phân tích phản hồi
                data = response.json()
            
            # Kiểm tra dữ liệu trả về
            if isinstance(data, dict):
                products = data.get("data", [])
                logger.info(f"Tìm thấy {len(products)} sản phẩm từ API")
            elif isinstance(data, list):
                products = data
                logger.info(f"Tìm thấy {len(products)} sản phẩm từ API (dạng list)")
            else:
                logger.warning(f"Định dạng dữ liệu không phù hợp: {type(data)}")
                return []
            
            # Trả về danh sách sản phẩm
            return products
        except Exception as e:
            logger.error(f"Lỗi khi lấy sản phẩm theo khoảng giá: {str(e)}")
            return []
    
    async def format_price_range_products(self, min_price: float, max_price: float, query: str) -> str:
        """
        Format danh sách sản phẩm theo khoảng giá thành HTML đẹp
        
        Args:
            min_price: Giá tối thiểu
            max_price: Giá tối đa
            query: Câu hỏi người dùng
            
        Returns:
            Chuỗi HTML hiển thị sản phẩm
        """
        try:
            products = await self.get_products_by_price_range(min_price, max_price)
            if not products or len(products) == 0:
                return f"Không tìm thấy sản phẩm nào trong khoảng giá từ {int(min_price):,}đ đến {int(max_price):,}đ."
            
            # Giới hạn số lượng sản phẩm hiển thị
            display_products = products[:15] if len(products) > 15 else products
            
            # Định dạng khoảng giá
            price_range_text = ""
            if min_price > 0 and max_price < float('inf'):
                price_range_text = f"từ {int(min_price):,}đ đến {int(max_price):,}đ"
            elif min_price > 0:
                price_range_text = f"từ {int(min_price):,}đ trở lên"
            elif max_price < float('inf'):
                price_range_text = f"dưới {int(max_price):,}đ"
            else:
                price_range_text = "tất cả các mức giá"
            
            # Tạo HTML cho danh sách sản phẩm
            html = f"""
            <div style="font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 15px;">
                <h2 style="color: #2a5885; margin-bottom: 20px;">Danh sách sản phẩm {price_range_text}</h2>
                <p style="margin-bottom: 15px;">Tìm thấy {len(products)} sản phẩm. Hiển thị {len(display_products)} sản phẩm đầu tiên:</p>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 20px;">
            """
            
            for product in display_products:
                name = product.get("productName", product.get("name", "Không có tên"))
                price = product.get("price", 0)
                image_url = product.get("imageUrl", product.get("image", ""))
                seller = product.get("seller", product.get("sellerName", ""))
                product_url = product.get("productUrl", "#")
                
                # Định dạng giá
                try:
                    price_formatted = f"{int(float(price)):,}đ" if price else "Liên hệ"
                except:
                    price_formatted = "Liên hệ"
                
                # Đảm bảo image_url có giá trị
                if not image_url:
                    image_url = "https://via.placeholder.com/150"
                
                # Thêm sản phẩm vào grid
                html += f"""
                    <div style="border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); transition: transform 0.3s, box-shadow 0.3s; display: flex; flex-direction: column; height: 100%;">
                        <div style="height: 180px; overflow: hidden; position: relative; background: #f5f5f5;">
                            <img src="{image_url}" alt="{name}" style="width: 100%; height: 100%; object-fit: cover;">
                        </div>
                        <div style="padding: 15px; flex-grow: 1; display: flex; flex-direction: column;">
                            <h3 style="margin: 0 0 10px; font-size: 16px; line-height: 1.3; height: 42px; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;">
                                <a href="{product_url}" style="color: #333; text-decoration: none; font-weight: 600;" target="_blank">{name}</a>
                            </h3>
                            <div style="margin-top: auto;">
                                <div style="color: #e74c3c; font-weight: bold; font-size: 18px; margin: 10px 0;">{price_formatted}</div>
                                <div style="color: #7f8c8d; font-size: 13px; margin-bottom: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">Người bán: {seller}</div>
                            </div>
                        </div>
                    </div>
                """
            
            html += """
                </div>
            </div>
            """
            
            return html
        except Exception as e:
            logger.error(f"Lỗi khi format danh sách sản phẩm theo khoảng giá: {str(e)}")
            return f"Đã xảy ra lỗi khi hiển thị sản phẩm theo khoảng giá: {str(e)}"

    async def get_products_by_category_id(self, category_id: int) -> List[Dict[str, Any]]:
        """
        Lấy danh sách sản phẩm thuộc một danh mục cụ thể
        
        Args:
            category_id: ID của danh mục
            
        Returns:
            Danh sách sản phẩm thuộc danh mục
        """
        try:
            if not category_id:
                return []
                
            # URL chính xác để lấy sản phẩm theo danh mục
            url = f"https://chodongbao.com/api/Products?category_id={category_id}&page=0&page_size=10"
            logger.info(f"Gọi API sản phẩm theo danh mục: {url}")
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json",
                "authenticatetoken": "ChoDongBao_HueCIT"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0, headers=headers)
                
                logger.info(f"API sản phẩm theo danh mục trả về status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Kiểm tra định dạng kết quả
                    if isinstance(result, dict) and "data" in result:
                        products = result.get("data", [])
                        logger.info(f"Tìm thấy {len(products)} sản phẩm thuộc danh mục {category_id} (từ object)")
                    elif isinstance(result, list):
                        products = result
                        logger.info(f"Tìm thấy {len(products)} sản phẩm thuộc danh mục {category_id} (từ list)")
                    else:
                        logger.warning(f"API trả về định dạng không mong đợi: {type(result)}")
                        products = []
                    
                    # Đảm bảo mỗi sản phẩm có trường price_display
                    for product in products:
                        if "price" in product and "price_display" not in product:
                            price = product["price"]
                            product["price_display"] = f"{price:,}đ".replace(",", ".")
                    
                    return products
                else:
                    logger.error(f"Lỗi khi gọi API sản phẩm theo danh mục: {response.status_code}")
                    return self._get_sample_products_by_category(category_id)
        except Exception as e:
            logger.error(f"Lỗi khi lấy sản phẩm theo danh mục: {str(e)}")
            return self._get_sample_products_by_category(category_id)
    
    def _get_sample_products_by_category(self, category_id: int) -> List[Dict[str, Any]]:
        """
        Cung cấp danh sách sản phẩm mẫu cho một danh mục khi API không hoạt động
        
        Args:
            category_id: ID của danh mục
            
        Returns:
            Danh sách sản phẩm mẫu
        """
        # Dữ liệu mẫu cho một số danh mục phổ biến
        sample_products = {
            # Thủ công mỹ nghệ (ID = 3)
            3: [
                {"id": "p31", "name": "Mây tre đan cao cấp", "price": 350000, "seller": "Làng nghề Phú Vinh"},
                {"id": "p32", "name": "Tượng gỗ mỹ nghệ", "price": 450000, "seller": "Làng nghề Bát Tràng"},
                {"id": "p33", "name": "Tranh thêu tay", "price": 850000, "seller": "Làng nghề Văn Lâm"}
            ],
            # Gạo & Lương thực (ID = 2)
            2: [
                {"id": "p21", "name": "Gạo ST25 đặc sản", "price": 35000, "seller": "HTX Sóc Trăng"},
                {"id": "p22", "name": "Gạo nếp cái hoa vàng", "price": 30000, "seller": "HTX An Giang"},
                {"id": "p23", "name": "Gạo lứt hữu cơ", "price": 45000, "seller": "Organic Rice"}
            ],
            # Nông nghiệp (ID = 1)
            1: [
                {"id": "p11", "name": "Rau sạch đảm bảo VietGAP", "price": 15000, "seller": "HTX Rau sạch Đà Lạt"},
                {"id": "p12", "name": "Hoa quả tươi theo mùa", "price": 25000, "seller": "Vườn trái cây Tiền Giang"},
                {"id": "p13", "name": "Nấm hữu cơ các loại", "price": 55000, "seller": "Nông trại nấm sạch"}
            ]
        }
        
        # Nếu có dữ liệu mẫu cho category_id, trả về dữ liệu đó
        if category_id in sample_products:
            logger.info(f"Sử dụng dữ liệu mẫu cho sản phẩm của danh mục {category_id}")
            # Đảm bảo mỗi sản phẩm có trường price_display
            for product in sample_products[category_id]:
                if "price" in product and "price_display" not in product:
                    price = product["price"]
                    product["price_display"] = f"{price:,}đ".replace(",", ".")
            return sample_products[category_id]
        
        # Nếu không có dữ liệu mẫu, trả về danh sách rỗng
        return []
        
    def format_price(self, price) -> str:
        """
        Định dạng giá tiền
        
        Args:
            price: Giá tiền cần định dạng
            
        Returns:
            Chuỗi giá tiền đã định dạng
        """
        try:
            if price == 0:
                return "Liên hệ"
            return f"{int(price):,}đ".replace(",", ".")
        except:
            return f"{price}đ"

    async def find_products_by_district(self, query: str) -> str:
        """
        Tìm sản phẩm theo quận/huyện
        
        Args:
            query: Câu truy vấn từ người dùng
            
        Returns:
            Danh sách sản phẩm theo quận/huyện
        """
        try:
            # Trích xuất tên quận/huyện từ câu truy vấn
            patterns = [
                r'quận\s+(\w+)',
                r'huyện\s+(\w+)',
                r'thị xã\s+(\w+)',
                r'thành phố\s+(\w+)',
                r'thị trấn\s+(\w+)',
                r'xã\s+(\w+)',
                r'phường\s+(\w+)',
                r'tại\s+(\w+)',
                r'ở\s+(\w+)'
            ]
            
            district_name = None
            for pattern in patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    district_name = match.group(1)
                    break
                    
            if not district_name:
                return "Vui lòng cung cấp tên quận/huyện/thành phố cụ thể để tôi có thể tìm sản phẩm."
            
            logger.info(f"Tìm sản phẩm tại: {district_name}")
            
            # Lấy sản phẩm từ API
            products_response = await self.get_products("")
            
            # Kiểm tra xem kết quả có phải là từ điển với khóa "data" không
            if isinstance(products_response, dict) and "data" in products_response:
                products = products_response.get("data", [])
                logger.info(f"Tìm thấy {len(products)} sản phẩm từ response.data")
            elif isinstance(products_response, list):
                products = products_response
                logger.info(f"Tìm thấy {len(products)} sản phẩm từ response list")
            else:
                logger.error(f"Định dạng phản hồi API không hợp lệ: {type(products_response)}")
                return "Lỗi khi tải danh sách sản phẩm: Định dạng dữ liệu không hợp lệ."
            
            if not products:
                return f"Không tìm thấy sản phẩm nào tại {district_name}."
            
            # Lọc sản phẩm theo quận/huyện
            district_products = []
            for product in products:
                # Kiểm tra nếu tên quận/huyện xuất hiện trong địa chỉ người bán
                seller_address = product.get("sellerAddress", "").lower()
                if district_name.lower() in seller_address:
                    district_products.append(product)
            
            if not district_products:
                return f"Không tìm thấy sản phẩm nào tại {district_name}."
            
            # Phân tích sản phẩm bằng AI
            analysis = await self.analyze_products_with_ai(district_products, f"Sản phẩm tại {district_name}")
            
            # Thêm tiêu đề
            intro = f"## Danh sách sản phẩm tại {district_name}\n\n"
            
            return intro + analysis
            
        except Exception as e:
            logger.error(f"Lỗi khi tìm sản phẩm theo quận/huyện: {str(e)}")
            return f"Đã xảy ra lỗi khi tìm sản phẩm theo quận/huyện: {str(e)}"

# Khởi tạo service
product_service = ProductService()
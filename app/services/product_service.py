import logging
import re
import os
import sys
from typing import List, Dict, Any, Optional
import httpx

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
        # Mẫu từ khóa liên quan đến sản phẩm
        product_keywords = [
            "giá", "mua", "bán", "sản phẩm", "hàng hóa", "giá cả", 
            "mặt hàng", "đồ", "đắt", "rẻ", "tiền", "giá tiền", "mua bán",
            "bao nhiêu", "tốt", "xấu", "chất lượng", "gạo", "thực phẩm",
            "danh sách", "liệt kê", "xem", "tất cả", "có những", "loại nào",
            "danh mục", "thông tin", "giới thiệu", "tên sản phẩm", "category",
            "phân loại", "loại sản phẩm", "nhóm", "loại", "chủng loại"
        ]
        
        # Kiểm tra từng từ khóa
        query_lower = query.lower()
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
            r"cửa hàng.*(?:có|bán)",
            r"danh mục.*(?:sản phẩm|hàng hóa)",
            r"phân loại.*(?:sản phẩm|hàng hóa)",
            r"loại sản phẩm"
        ]
        
        for pattern in product_patterns:
            if re.search(pattern, query_lower):
                return True
                
        # Kiểm tra các câu hỏi về danh sách sản phẩm
        product_list_phrases = [
            "danh sách sản phẩm", "liệt kê sản phẩm", "xem sản phẩm", 
            "các sản phẩm", "tất cả sản phẩm", "có những sản phẩm nào",
            "những sản phẩm", "các loại", "có những loại nào",
            "danh mục", "thông tin sản phẩm", "giới thiệu sản phẩm",
            "cửa hàng có gì", "bán những gì", "danh mục sản phẩm",
            "phân loại sản phẩm", "nhóm sản phẩm", "loại sản phẩm"
        ]
        
        for phrase in product_list_phrases:
            if phrase in query_lower:
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
            
            # Sắp xếp tất cả sản phẩm theo giá tăng dần trước
            sorted_products = sorted(products, key=lambda x: float(x.get("price", 0)))
            
            # Giới hạn số lượng sản phẩm để tránh token quá lớn
            max_products = 100 if is_product_list_query or is_price_sort else 20
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
            if is_product_list_query or is_price_sort:
                system_prompt = f"""
                Dưới đây là danh sách sản phẩm đã được sắp xếp theo giá từ thấp đến cao. Hãy liệt kê các sản phẩm này một cách rõ ràng, tổ chức theo nhóm giá:
                
                1. Nhóm giá rẻ: Giá dưới 100.000đ
                2. Nhóm giá trung bình: Giá từ 100.000đ đến 500.000đ
                3. Nhóm giá cao: Giá trên 500.000đ
                
                Với mỗi sản phẩm, hãy tạo một liên kết HTML (<a>) như sau:
                * <a href="URL_SẢN_PHẨM">TÊN_SẢN_PHẨM</a>: GIÁ/ĐƠN_VỊ (NGƯỜI_BÁN)
                
                Đảm bảo sản phẩm được hiển thị đúng thứ tự từ giá thấp đến cao trong mỗi nhóm.
                Chỉ hiển thị mô tả sản phẩm khi thực sự cần thiết như khi so sánh giữa các sản phẩm, và giữ mô tả ngắn gọn.
                Thêm thông tin tổng quan về giá thấp nhất, cao nhất.
                Nếu người dùng có yêu cầu sắp xếp lại, hãy sắp xếp theo yêu cầu đó.
                
                Danh sách sản phẩm:
                {products_text}
                """
            else:
                system_prompt = f"""
                Dựa trên danh sách sản phẩm sau đây, hãy trả lời câu hỏi của người dùng một cách chi tiết và hữu ích.
                Hãy phân tích giá cả, so sánh sản phẩm nếu cần thiết, và đưa ra gợi ý phù hợp.
                Nếu người dùng hỏi về giá, hãy nêu rõ giá của từng sản phẩm.
                Nếu người dùng so sánh sản phẩm, hãy so sánh dựa trên giá cả, người bán, và các thông tin có sẵn.
                
                Với mỗi sản phẩm, hãy tạo một liên kết HTML (<a>) như sau:
                <a href="URL_SẢN_PHẨM">TÊN_SẢN_PHẨM</a>: GIÁ/ĐƠN_VỊ
                
                Chỉ hiển thị mô tả sản phẩm khi cần thiết và giữ mô tả ngắn gọn.
                
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
            if is_product_list_query or is_price_sort:
                # Danh sách sản phẩm đẹp hơn cho câu hỏi về danh sách
                result = f"Danh sách {len(products)} sản phẩm (hiển thị {len(truncated_products)}):\n\n"
                
                # Tìm giá cao nhất và thấp nhất
                min_price = sorted_products[0].get("price", 0) if sorted_products else 0
                max_price = sorted_products[-1].get("price", 0) if sorted_products else 0
                
                # Thêm thông tin tổng quan
                result += f"**Giá thấp nhất:** {min_price:,}đ\n"
                result += f"**Giá cao nhất:** {max_price:,}đ\n\n"
                
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
                    result += "**Nhóm giá rẻ (dưới 100.000đ):**\n\n"
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
                    result += "**Nhóm giá trung bình (100.000đ - 500.000đ):**\n\n"
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
                    result += "**Nhóm giá cao (trên 500.000đ):**\n\n"
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
            # Kiểm tra nếu câu hỏi liên quan đến danh mục
            query_lower = query.lower()
            is_category_query = any(phrase in query_lower for phrase in [
                "danh mục", "phân loại", "loại sản phẩm", "nhóm", "category",
                "chủng loại", "các loại"
            ])
            
            if is_category_query:
                # Xử lý câu hỏi về danh mục
                return await self.process_category_query(query)
            
            # Kiểm tra nếu là yêu cầu về danh sách sản phẩm chung
            is_product_list_query = any(phrase in query_lower for phrase in [
                "danh sách sản phẩm", "liệt kê sản phẩm", "xem sản phẩm", 
                "các sản phẩm", "tất cả sản phẩm", "có những sản phẩm nào",
                "những sản phẩm", "các loại", "có những loại nào"
            ])
            
            # Xác định tên sản phẩm dựa vào loại câu hỏi
            product_name = ""
            if is_product_list_query:
                # Nếu là câu hỏi về danh sách sản phẩm chung, sử dụng chuỗi rỗng
                logger.info(f"Phát hiện yêu cầu danh sách sản phẩm chung, sử dụng chuỗi rỗng để lấy tất cả sản phẩm")
                product_name = ""  # Chuỗi rỗng để lấy tất cả sản phẩm
            else:
                # Nếu không phải là câu hỏi về danh sách chung, trích xuất tên sản phẩm từ câu hỏi
                product_name = await self.extract_product_name(query)
                
                if not product_name:
                    return "Không thể xác định sản phẩm cần tìm. Vui lòng nêu rõ tên sản phẩm."
            
            # Lấy danh sách sản phẩm từ API
            result = await self.get_products(product_name)
            
            if not result.get("success", False) or not result.get("data", []):
                if product_name:
                    return f"Không tìm thấy thông tin về sản phẩm '{product_name}'."
                else:
                    return "Không tìm thấy thông tin sản phẩm nào."
            
            # Phân tích danh sách sản phẩm bằng AI
            products = result.get("data", [])
            analysis = await self.analyze_products_with_ai(products, query)
            
            return analysis
        except Exception as e:
            logger.error(f"Lỗi khi xử lý câu hỏi sản phẩm: {str(e)}")
            return f"Đã xảy ra lỗi khi xử lý thông tin sản phẩm: {str(e)}"
    
    async def process_category_query(self, query: str) -> str:
        """
        Xử lý câu hỏi liên quan đến danh mục sản phẩm
        
        Args:
            query: Câu hỏi của người dùng
            
        Returns:
            Kết quả trả lời
        """
        try:
            # Import module với kiểm tra lỗi
            try:
                from app.api.query_demo.product_api import get_categories, format_categories
            except ImportError:
                logger.error(f"Không thể import module product_api")
                return "Không thể tải thông tin danh mục sản phẩm. Vui lòng thử lại sau."
            
            # Lấy danh mục từ API
            categories = await get_categories(page_size=50)
            
            # Kiểm tra xem có muốn biết thông tin chi tiết về danh mục cụ thể không
            category_name = None
            for phrase in ["danh mục", "loại", "chủng loại", "phân loại"]:
                if phrase in query.lower():
                    # Tìm tên danh mục sau từ khóa
                    parts = query.lower().split(phrase)
                    if len(parts) > 1 and parts[1].strip():
                        potential_name = parts[1].strip()
                        # Loại bỏ các từ không cần thiết
                        for word in ["sản phẩm", "hàng hóa", "về", "của"]:
                            potential_name = potential_name.replace(word, "").strip()
                        if potential_name:
                            category_name = potential_name
                            break
            
            # Nếu có tên danh mục cụ thể, lọc kết quả
            if category_name:
                logger.info(f"Đang tìm thông tin về danh mục: {category_name}")
                filtered_categories = []
                for cat in categories.get("data", []):
                    if category_name.lower() in cat.get("name", "").lower():
                        filtered_categories.append(cat)
                
                if filtered_categories:
                    result = {
                        "success": True,
                        "data": filtered_categories,
                        "total": len(filtered_categories),
                        "message": f"Tìm thấy {len(filtered_categories)} danh mục khớp với '{category_name}'"
                    }
                    return format_categories(result)
                else:
                    return f"Không tìm thấy danh mục nào phù hợp với '{category_name}'."
            
            # Nếu không có tên cụ thể, trả về toàn bộ danh mục
            formatted_result = format_categories(categories)
            
            # Thử phân tích thêm bằng AI
            try:
                # Tạo prompt cho AI
                system_prompt = """
                Dưới đây là danh sách danh mục sản phẩm. Hãy tóm tắt và phân tích để cung cấp thông tin hữu ích cho người dùng.
                Đưa ra một số gợi ý về sản phẩm có thể tìm thấy trong từng danh mục chính.
                
                Danh sách danh mục:
                """
                
                messages = [
                    {"role": "system", "content": system_prompt + formatted_result},
                    {"role": "user", "content": query}
                ]
                
                # Thử dùng Gemini
                try:
                    response = await gemini_service.chat(
                        messages=messages,
                        temperature=0.7,
                        max_tokens=1000
                    )
                    return response["answer"]
                except Exception as e:
                    logger.warning(f"Lỗi khi dùng Gemini cho phân tích danh mục: {str(e)}")
                    
                    # Nếu Gemini lỗi, thử dùng OpenAI
                    try:
                        response = await openai_service.chat(
                            messages=messages,
                            temperature=0.7,
                            max_tokens=1000
                        )
                        return response["answer"]
                    except Exception as e:
                        logger.warning(f"Lỗi khi dùng OpenAI cho phân tích danh mục: {str(e)}")
            except Exception as e:
                logger.warning(f"Lỗi khi phân tích danh mục với AI: {str(e)}")
            
            # Trả về định dạng chuẩn nếu phân tích AI thất bại
            return formatted_result
            
        except Exception as e:
            logger.error(f"Lỗi khi xử lý câu hỏi danh mục: {str(e)}")
            return f"Đã xảy ra lỗi khi xử lý thông tin danh mục: {str(e)}"

# Khởi tạo service
product_service = ProductService() 
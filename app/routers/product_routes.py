from fastapi import APIRouter, Request, Query, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from app.services.product_service import product_service
from app.services.ai_service import ai_service
from app.templates.base_template import render_template

router = APIRouter()

@router.get("/products/price-range", response_class=HTMLResponse)
async def get_products_by_price_range(
    request: Request, 
    min_price: int = Query(0, description="Giá tối thiểu"), 
    max_price: int = Query(None, description="Giá tối đa")
):
    """
    Hiển thị sản phẩm trong khoảng giá xác định
    """
    try:
        # Kiểm tra giá trị nhập vào
        if min_price < 0:
            min_price = 0
        
        if max_price is not None and max_price < min_price:
            return render_template(
                request,
                "Sản phẩm theo giá",
                "Lỗi: Giá tối đa phải lớn hơn giá tối thiểu",
                is_error=True
            )
        
        # Tạo query string mô tả khoảng giá
        if max_price is not None:
            query = f"sản phẩm có giá từ {min_price} đến {max_price}"
        else:
            query = f"sản phẩm có giá từ {min_price} trở lên"
            
        # Chuyển đổi max_price None thành float('inf') cho function
        max_value = float('inf') if max_price is None else float(max_price)
            
        # Lấy và định dạng sản phẩm theo khoảng giá
        formatted_products = await product_service.format_price_range_products(
            min_price=float(min_price),
            max_price=max_value,
            query=query
        )
        
        # Tạo tiêu đề trang
        price_range_text = f"từ {min_price:,}đ đến {max_price:,}đ" if max_price else f"từ {min_price:,}đ trở lên"
        page_title = f"Sản phẩm {price_range_text}"
        
        # Render template với dữ liệu sản phẩm
        return render_template(
            request,
            page_title,
            formatted_products,
            is_error=False
        )
    except Exception as e:
        return render_template(
            request,
            "Lỗi hiển thị sản phẩm theo giá",
            f"Đã xảy ra lỗi: {str(e)}",
            is_error=True
        ) 
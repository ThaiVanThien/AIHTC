from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

# Khởi tạo Jinja2Templates với thư mục templates
templates = Jinja2Templates(directory="app/templates")

# Chuẩn bị template cơ bản nếu chưa tồn tại
BASE_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .error-message {
            color: #dc3545;
            background-color: #f8d7da;
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        h1 {
            color: #2a5885;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <div {% if is_error %}class="error-message"{% endif %}>
            {{ content|safe }}
        </div>
    </div>
</body>
</html>
"""

def ensure_base_template_exists():
    """Đảm bảo file template cơ bản tồn tại"""
    template_path = os.path.join("app", "templates", "base.html")
    if not os.path.exists(template_path):
        with open(template_path, "w", encoding="utf-8") as f:
            f.write(BASE_TEMPLATE)
        return True
    return False

# Đảm bảo template cơ bản đã được tạo
ensure_base_template_exists()

async def render_template(
    request: Request, 
    title: str, 
    content: str, 
    is_error: bool = False
) -> HTMLResponse:
    """
    Render template cơ bản với nội dung
    
    Args:
        request: Request object
        title: Tiêu đề trang
        content: Nội dung HTML
        is_error: Có phải thông báo lỗi không
        
    Returns:
        HTMLResponse
    """
    try:
        # Thử dùng Jinja2Templates
        ensure_base_template_exists()
        return templates.TemplateResponse(
            "base.html", 
            {"request": request, "title": title, "content": content, "is_error": is_error}
        )
    except Exception as e:
        # Fallback: trả về HTML trực tiếp
        html_content = BASE_TEMPLATE.replace("{{ title }}", title)
        html_content = html_content.replace("{{ content|safe }}", content)
        if is_error:
            html_content = html_content.replace(
                '<div ', '<div class="error-message" '
            )
        else:
            html_content = html_content.replace(
                '{% if is_error %}class="error-message"{% endif %}', ''
            )
        return HTMLResponse(content=html_content) 
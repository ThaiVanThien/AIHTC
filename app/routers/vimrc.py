from fastapi import APIRouter, HTTPException, BackgroundTasks, Form, Depends, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import os
import shutil
from pathlib import Path
from datetime import datetime

from app.core.config import settings
from app.services.nlp_factory import nlp_factory
from app.services.vimrc_service import vimrc_service

router = APIRouter(
    prefix="/vimrc",
    tags=["vi-mrc"],
    responses={404: {"description": "Not found"}},
)

# Templates configuration
templates = Jinja2Templates(directory="app/templates")

class TrainingRequest(BaseModel):
    """Mô hình yêu cầu huấn luyện"""
    model_name: str = Field(..., description="Tên mô hình sẽ được lưu", example="accounting_model_v1")
    epochs: int = Field(3, description="Số epochs huấn luyện", example=3)
    batch_size: int = Field(8, description="Kích thước batch", example=8)

class ModelDownloadRequest(BaseModel):
    """Mô hình yêu cầu tải mô hình từ URL"""
    url: str = Field(..., description="URL của mô hình (zip file)", example="https://example.com/models/vi-mrc-model.zip")
    model_name: str = Field("vi-mrc-custom", description="Tên mô hình sẽ được lưu", example="vi-mrc-custom")

@router.get("/status", response_model=Dict[str, Any], summary="Trạng thái dịch vụ vi-mrc")
async def get_vimrc_status():
    """
    Lấy trạng thái hiện tại của dịch vụ vi-mrc
    
    Thông tin trả về bao gồm:
    - Trạng thái tải mô hình
    - Tên mô hình đang sử dụng
    - Thiết bị đang sử dụng (CPU/GPU)
    - Trạng thái huấn luyện (nếu có)
    """
    return vimrc_service.get_status()

@router.post("/answer", response_model=Dict[str, Any], summary="Trả lời câu hỏi với vi-mrc")
async def answer_question(question: str, context: str):
    """
    Trả lời câu hỏi dựa trên ngữ cảnh sử dụng mô hình vi-mrc
    
    - **question**: Câu hỏi cần trả lời
    - **context**: Ngữ cảnh/đoạn văn chứa câu trả lời
    
    Ví dụ:
    - Câu hỏi: "Doanh thu quý 1 là bao nhiêu?"
    - Ngữ cảnh: "Doanh thu quý 1 là 500 tỷ đồng, tăng 20% so với cùng kỳ năm ngoái."
    - Kết quả: "500 tỷ đồng"
    """
    try:
        result = vimrc_service.answer_question(question, context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý câu hỏi: {str(e)}")

@router.post("/train", response_model=Dict[str, Any], summary="Huấn luyện mô hình vi-mrc")
async def train_vimrc_model(
    background_tasks: BackgroundTasks,
    model_name: str = Form(...),
    epochs: int = Form(3),
    batch_size: int = Form(8)
):
    """
    Huấn luyện mô hình vi-mrc với dữ liệu trong thư mục training
    
    - **model_name**: Tên mô hình sẽ được lưu
    - **epochs**: Số epochs huấn luyện
    - **batch_size**: Kích thước batch
    
    Huấn luyện sẽ diễn ra ở background và không chặn API. Sử dụng endpoint /training-status để theo dõi tiến trình.
    """
    # Kiểm tra trạng thái huấn luyện
    training_status = vimrc_service.get_training_status()
    if training_status["is_training"]:
        return {
            "success": False,
            "message": "Đã có quá trình huấn luyện đang diễn ra. Vui lòng đợi hoàn tất trước khi bắt đầu huấn luyện mới.",
            "status": training_status
        }
    
    # Thêm tác vụ huấn luyện vào background
    background_tasks.add_task(
        vimrc_service.train_model,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size
    )
    
    return {
        "success": True,
        "message": f"Đã bắt đầu huấn luyện mô hình {model_name} với {epochs} epochs",
        "status": vimrc_service.get_training_status()
    }

@router.get("/training-status", response_model=Dict[str, Any], summary="Trạng thái huấn luyện")
async def get_training_status():
    """
    Lấy trạng thái huấn luyện hiện tại của mô hình vi-mrc
    
    Thông tin trả về bao gồm:
    - Trạng thái: đang huấn luyện, hoàn thành, lỗi, v.v.
    - Tiến độ: phần trăm hoàn thành
    - Epoch hiện tại và tổng số epoch
    - Thời gian đã trôi qua và ước tính thời gian còn lại
    """
    return vimrc_service.get_training_status()

@router.post("/download-model", response_model=Dict[str, Any], summary="Tải mô hình từ URL")
async def download_model(url: str, model_name: str = "vi-mrc-model"):
    """
    Tải mô hình từ URL (file zip) và lưu vào thư mục models
    
    - **url**: URL của mô hình (file zip)
    - **model_name**: Tên thư mục sẽ lưu mô hình
    
    Mô hình tải xuống cần phải có định dạng đúng cho mô hình Question Answering
    """
    if vimrc_service.download_model(url, model_name):
        return {
            "success": True,
            "message": f"Đã tải và cài đặt mô hình từ {url}"
        }
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Không thể tải mô hình từ {url}. Kiểm tra log để biết thêm chi tiết."
        )

@router.post("/download-huggingface", response_model=Dict[str, Any], summary="Tải mô hình từ Hugging Face")
async def download_from_huggingface(
    model_id: str = "vinai/vi-mrc-large", 
    revision: str = "main",
    local_name: str = None,
    use_cache: bool = True
):
    """
    Tải mô hình trực tiếp từ Hugging Face Hub vào thư mục models
    
    - **model_id**: ID mô hình trên Hugging Face (ví dụ: vinai/vi-mrc-large)
    - **revision**: Phiên bản mô hình (default: main)
    - **local_name**: Tên thư mục local để lưu mô hình (mặc định lấy từ model_id)
    - **use_cache**: Sử dụng cache Hugging Face nếu có (tiết kiệm băng thông)
    
    Tải mô hình trực tiếp từ Hugging Face Hub, không cần thông qua file zip trung gian.
    Hỗ trợ các mô hình như vinai/vi-mrc-large, vinai/phobert-base-v2, và các mô hình QA khác.
    """
    try:
        # Xác định tên thư mục lưu trữ local
        if not local_name:
            # Lấy phần cuối của model_id (sau dấu /)
            local_name = model_id.split("/")[-1] if "/" in model_id else model_id
        
        # Gọi phương thức download_from_huggingface từ vimrc_service
        success = await vimrc_service.download_from_huggingface(
            model_id=model_id,
            revision=revision,
            local_dir_name=local_name,
            use_cache=use_cache
        )
        
        if success:
            cache_note = "sử dụng cache" if use_cache else "không sử dụng cache"
            return {
                "success": True,
                "message": f"Đã tải và cài đặt mô hình {model_id} (revision: {revision}) vào {settings.MODELS_DIR}/{local_name} ({cache_note})"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Không thể tải mô hình {model_id} từ Hugging Face Hub. Kiểm tra log để biết thêm chi tiết."
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi tải mô hình từ Hugging Face: {str(e)}"
        )

@router.get("/models", response_model=Dict[str, Any], summary="Danh sách mô hình vi-mrc")
async def get_models():
    """
    Lấy danh sách tất cả mô hình vi-mrc có sẵn trong hệ thống
    
    Thông tin trả về bao gồm:
    - Danh sách tên các mô hình
    - Mô hình mặc định
    - Đường dẫn thư mục chứa mô hình
    """
    try:
        models_dir = Path(settings.MODELS_DIR)
        if not models_dir.exists():
            models_dir.mkdir(parents=True, exist_ok=True)
            
        available_models = []
        for item in models_dir.iterdir():
            if item.is_dir():
                available_models.append(item.name)
                
        return {
            "success": True,
            "models": available_models,
            "default_model": settings.DEFAULT_MODEL_NAME,
            "models_directory": str(models_dir)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy danh sách mô hình: {str(e)}")

@router.get("/training-data", response_model=Dict[str, Any], summary="Danh sách dữ liệu huấn luyện")
async def get_training_data():
    """
    Lấy danh sách tất cả tệp dữ liệu huấn luyện đã tải lên
    
    Thông tin trả về bao gồm:
    - Tên file
    - Kích thước file
    - Thời gian chỉnh sửa cuối cùng
    """
    try:
        training_dir = Path(settings.TRAINING_DATA_DIR)
        if not training_dir.exists():
            training_dir.mkdir(parents=True, exist_ok=True)
            
        files = []
        for item in training_dir.iterdir():
            if item.is_file():
                files.append({
                    "name": item.name,
                    "size": item.stat().st_size,
                    "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                })
                
        return {
            "success": True,
            "files": files,
            "directory": str(training_dir)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy danh sách dữ liệu huấn luyện: {str(e)}")

@router.delete("/training-data/{filename}", response_model=Dict[str, Any], summary="Xóa tệp dữ liệu huấn luyện")
async def delete_training_file(filename: str):
    """
    Xóa tệp dữ liệu huấn luyện đã tải lên
    
    - **filename**: Tên tệp cần xóa
    """
    try:
        file_path = Path(settings.TRAINING_DATA_DIR) / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Tệp '{filename}' không tồn tại")
            
        file_path.unlink()
        return {
            "success": True,
            "message": f"Đã xóa tệp '{filename}'"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa tệp: {str(e)}")

@router.delete("/models/{model_name}", response_model=Dict[str, Any], summary="Xóa mô hình vi-mrc")
async def delete_model(model_name: str):
    """
    Xóa mô hình vi-mrc
    
    - **model_name**: Tên mô hình cần xóa
    
    Không thể xóa mô hình mặc định.
    """
    try:
        if model_name == settings.DEFAULT_MODEL_NAME:
            raise HTTPException(status_code=400, detail=f"Không thể xóa mô hình mặc định '{settings.DEFAULT_MODEL_NAME}'")
            
        model_path = Path(settings.MODELS_DIR) / model_name
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Mô hình '{model_name}' không tồn tại")
            
        # Xóa thư mục mô hình
        shutil.rmtree(model_path)
        
        return {
            "success": True,
            "message": f"Đã xóa mô hình '{model_name}'"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa mô hình: {str(e)}")

@router.post("/upload-training-file", response_model=Dict[str, Any], summary="Tải lên tệp huấn luyện")
async def upload_training_file(
    file: UploadFile = File(...),
    file_type: str = Form(..., description="Loại tập tin (json, csv, excel)")
):
    """
    Tải lên tệp dữ liệu huấn luyện cho mô hình vi-mrc
    
    - **file**: Tệp dữ liệu huấn luyện
    - **file_type**: Loại tệp (json, csv, excel)
    
    Dữ liệu trong tệp phải có định dạng phù hợp:
    - JSON: Mảng các đối tượng với các trường "question", "context", "answer"
    - CSV/Excel: Các cột "question", "context", "answer"
    
    Tệp được tải lên sẽ được lưu trong thư mục data/training và sẽ được sử dụng khi huấn luyện mô hình.
    """
    try:
        # Tạo thư mục lưu trữ file nếu chưa tồn tại
        upload_dir = Path(settings.TRAINING_DATA_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo đường dẫn lưu file
        file_extension = file_type.lower()
        if file_extension not in ["json", "csv", "xlsx", "xls"]:
            raise HTTPException(status_code=400, detail="Định dạng tệp không được hỗ trợ. Chỉ chấp nhận JSON, CSV, hoặc Excel.")
        
        # Tạo tên file mới với timestamp để tránh trùng lặp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"training_data_{timestamp}.{file_extension}"
        file_path = upload_dir / file_name
        
        # Lưu file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {
            "filename": file_name,
            "file_path": str(file_path),
            "file_type": file_type,
            "status": "success",
            "message": f"Đã tải lên tệp huấn luyện thành công. Sử dụng endpoint /vimrc/train để bắt đầu huấn luyện."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tải lên tệp huấn luyện: {str(e)}")

@router.get("/chat", response_class=HTMLResponse, summary="VI-MRC Chat UI")
async def get_chat_ui(request: Request):
    """
    Hiển thị giao diện chat để tương tác với mô hình VI-MRC
    
    Giao diện cho phép người dùng đặt câu hỏi và cung cấp ngữ cảnh
    để nhận câu trả lời từ mô hình VI-MRC
    """
    return templates.TemplateResponse("vimrc_chat.html", {"request": request}) 
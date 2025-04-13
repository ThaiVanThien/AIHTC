from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import os
import shutil
from pathlib import Path

from app.services.nlp_service import nlp_service

router = APIRouter()

class QuestionRequest(BaseModel):
    """Mô hình yêu cầu câu hỏi"""
    question: str = Field(..., description="Câu hỏi cần trả lời", example="Doanh thu quý 1 là bao nhiêu?")
    context: str = Field(..., description="Ngữ cảnh chứa câu trả lời", 
                      example="Doanh thu quý 1 là 500 tỷ đồng, tăng 20% so với cùng kỳ năm ngoái.")

class TrainingRequest(BaseModel):
    """Mô hình yêu cầu huấn luyện"""
    model_name: str = Field(..., description="Tên mô hình sẽ được lưu", example="accounting_model_v1")
    epochs: int = Field(3, description="Số epochs huấn luyện", example=3)
    batch_size: int = Field(8, description="Kích thước batch", example=8)


@router.get("/status", response_model=Dict[str, Any], summary="Trạng thái NLP")
def get_nlp_status():
    """
    Kiểm tra trạng thái của dịch vụ NLP
    """
    return {
        "model_loaded": nlp_service.is_model_loaded,
        "model_type": "QA (Question Answering)"
    }


@router.post("/answer-question", response_model=Dict[str, Any], summary="Trả lời câu hỏi")
def answer_question(request: QuestionRequest):
    """
    Trả lời câu hỏi dựa trên ngữ cảnh sử dụng mô hình vi-mrc
    
    - **question**: Câu hỏi cần trả lời
    - **context**: Đoạn văn bản chứa câu trả lời
    
    Ví dụ:
    - Câu hỏi: "Doanh thu quý 1 là bao nhiêu?"
    - Ngữ cảnh: "Doanh thu quý 1 là 500 tỷ đồng, tăng 20% so với cùng kỳ năm ngoái."
    - Kết quả: "500 tỷ đồng"
    """
    try:
        result = nlp_service.answer_question(request.question, request.context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý câu hỏi: {str(e)}")


@router.post("/upload-training-file", response_model=Dict[str, Any], summary="Tải lên tệp huấn luyện")
async def upload_training_file(
    file: UploadFile = File(...),
    file_type: str = Form(..., description="Loại tập tin (json, csv, excel)")
):
    """
    Tải lên tệp dữ liệu huấn luyện cho mô hình NLP
    
    - **file**: Tệp dữ liệu huấn luyện
    - **file_type**: Loại tệp (json, csv, excel)
    
    Dữ liệu trong tệp phải có định dạng phù hợp:
    - JSON: Mảng các đối tượng với các trường "question", "context", "answer"
    - CSV/Excel: Các cột "question", "context", "answer"
    """
    try:
        # Tạo thư mục lưu trữ file nếu chưa tồn tại
        upload_dir = Path("./data/training")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo đường dẫn lưu file
        file_extension = file_type.lower()
        if file_extension not in ["json", "csv", "xlsx", "xls"]:
            raise HTTPException(status_code=400, detail="Định dạng tệp không được hỗ trợ. Chỉ chấp nhận JSON, CSV, hoặc Excel.")
        
        # Tạo tên file mới với timestamp để tránh trùng lặp
        timestamp = int(os.path.getmtime(upload_dir) if os.path.exists(upload_dir) else 0)
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
            "message": f"Đã tải lên tệp huấn luyện thành công. Sử dụng endpoint /train-model để bắt đầu huấn luyện."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tải lên tệp huấn luyện: {str(e)}")


@router.post("/train-model", response_model=Dict[str, Any], summary="Huấn luyện mô hình")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Bắt đầu huấn luyện mô hình với dữ liệu đã tải lên
    
    - **model_name**: Tên mô hình sẽ được lưu
    - **epochs**: Số epochs huấn luyện
    - **batch_size**: Kích thước batch
    
    Quá trình huấn luyện sẽ được thực hiện trong background và có thể mất một thời gian.
    Trạng thái huấn luyện có thể kiểm tra qua endpoint /training-status.
    """
    try:
        # Kiểm tra xem có tệp huấn luyện nào đã được tải lên chưa
        upload_dir = Path("./data/training")
        if not upload_dir.exists() or not any(upload_dir.iterdir()):
            raise HTTPException(
                status_code=400, 
                detail="Không tìm thấy tệp huấn luyện. Vui lòng tải lên tệp huấn luyện trước khi bắt đầu huấn luyện."
            )
        
        # Bắt đầu quá trình huấn luyện trong background
        background_tasks.add_task(
            nlp_service.train_model, 
            model_name=request.model_name,
            training_dir=str(upload_dir),
            epochs=request.epochs,
            batch_size=request.batch_size
        )
        
        return {
            "status": "started",
            "message": "Quá trình huấn luyện đã bắt đầu. Kiểm tra trạng thái thông qua endpoint /training-status.",
            "model_name": request.model_name,
            "config": {
                "epochs": request.epochs,
                "batch_size": request.batch_size
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi bắt đầu huấn luyện: {str(e)}")


@router.get("/training-status", response_model=Dict[str, Any], summary="Trạng thái huấn luyện")
def get_training_status():
    """
    Kiểm tra trạng thái huấn luyện hiện tại
    """
    try:
        status = nlp_service.get_training_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy trạng thái huấn luyện: {str(e)}") 
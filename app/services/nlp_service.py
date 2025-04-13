import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import time
import json
import pandas as pd
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class NLPService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_model_loaded = False
        self.training_status = {
            "is_training": False,
            "current_epoch": 0,
            "total_epochs": 0,
            "progress": 0.0,
            "model_name": "",
            "start_time": None,
            "end_time": None,
            "status": "idle",
            "message": "Không có quá trình huấn luyện nào đang diễn ra."
        }
        self.training_lock = threading.Lock()

    def load_models(self):
        """
        Tải mô hình Question Answering cho tiếng Việt
        """
        try:
            logger.info("Đang tải mô hình NLP...")
            model_name = "nguyenvulebinh/vi-mrc-base"
            
            # Tải mô hình và tokenizer từ Hugging Face
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            
            # Đặt model ở chế độ evaluation
            self.model.eval()
            
            # Kiểm tra nếu có sẵn CUDA
            if torch.cuda.is_available():
                logger.info(f"Sử dụng GPU: {torch.cuda.get_device_name(0)}")
                self.model = self.model.to("cuda")
            else:
                logger.info("Sử dụng CPU vì không tìm thấy GPU")
                
            self.is_model_loaded = True
            logger.info("Đã tải xong mô hình NLP")
            
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình NLP: {str(e)}")
            self.is_model_loaded = False
            raise

    def get_status(self):
        """
        Lấy trạng thái hiện tại của dịch vụ NLP
        """
        device = "gpu" if torch.cuda.is_available() and self.is_model_loaded else "cpu"
        return {
            "model_loaded": self.is_model_loaded,
            "device": device,
            "model_name": "nguyenvulebinh/vi-mrc-base" if self.is_model_loaded else None
        }

    def answer_question(self, question: str, context: str):
        """
        Trả lời câu hỏi dựa trên ngữ cảnh sử dụng mô hình vi-mrc
        
        Args:
            question: Câu hỏi
            context: Ngữ cảnh chứa câu trả lời
            
        Returns:
            Dict chứa câu trả lời và thông tin liên quan
        """
        # Kiểm tra xem mô hình đã được tải chưa
        if not self.is_model_loaded:
            logger.warning("Yêu cầu trả lời câu hỏi khi mô hình chưa được tải")
            self.load_models()
        
        # Tokenize input
        inputs = self.tokenizer(
            question,
            context,
            add_special_tokens=True,
            return_tensors="pt"
        )
        
        # Đưa input lên GPU nếu có
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Dự đoán
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Lấy điểm bắt đầu và kết thúc cho câu trả lời
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits)
        
        # Convert sang vị trí token
        input_ids = inputs["input_ids"].tolist()[0]
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(
                input_ids[answer_start:answer_end+1]
            )
        )
        
        # Tính điểm tin cậy
        start_scores = outputs.start_logits.cpu().numpy()[0]
        end_scores = outputs.end_logits.cpu().numpy()[0]
        confidence = float(start_scores[answer_start] + end_scores[answer_end]) / 2
        
        # Tìm vị trí trong văn bản gốc
        char_start = len(self.tokenizer.decode(input_ids[:answer_start], skip_special_tokens=True))
        char_end = len(self.tokenizer.decode(input_ids[:answer_end+1], skip_special_tokens=True))
        
        # Trả về kết quả
        return {
            "answer": answer,
            "confidence": confidence,
            "char_start": char_start,
            "char_end": char_end,
            "context": context
        }

    def train_model(self, model_name: str, training_dir: str, epochs: int = 3, batch_size: int = 8):
        """
        Huấn luyện mô hình với dữ liệu được cung cấp
        
        Args:
            model_name: Tên mô hình sẽ được lưu
            training_dir: Thư mục chứa dữ liệu huấn luyện
            epochs: Số epochs huấn luyện
            batch_size: Kích thước batch
        """
        # Kiểm tra xem có đang huấn luyện không
        with self.training_lock:
            if self.training_status["is_training"]:
                logger.warning("Đã có quá trình huấn luyện đang diễn ra")
                return
            
            # Cập nhật trạng thái huấn luyện
            self.training_status = {
                "is_training": True,
                "current_epoch": 0,
                "total_epochs": epochs,
                "progress": 0.0,
                "model_name": model_name,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "status": "preparing",
                "message": "Đang chuẩn bị dữ liệu huấn luyện..."
            }
        
        try:
            # Tạo thư mục models nếu chưa tồn tại
            models_dir = Path("./data/models")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            # Cập nhật trạng thái
            with self.training_lock:
                self.training_status["status"] = "loading_data"
                self.training_status["message"] = "Đang tải dữ liệu huấn luyện..."
            
            # Tải dữ liệu huấn luyện
            training_data = self._load_training_data(training_dir)
            
            # Kiểm tra dữ liệu
            if not training_data or len(training_data) == 0:
                with self.training_lock:
                    self.training_status["is_training"] = False
                    self.training_status["status"] = "error"
                    self.training_status["message"] = "Không có dữ liệu huấn luyện hợp lệ."
                    self.training_status["end_time"] = datetime.now().isoformat()
                logger.error("Không có dữ liệu huấn luyện hợp lệ")
                return
            
            logger.info(f"Đã tải {len(training_data)} mẫu dữ liệu huấn luyện")
            
            # Cập nhật trạng thái
            with self.training_lock:
                self.training_status["status"] = "loading_model"
                self.training_status["message"] = "Đang tải mô hình cơ sở..."
            
            # Tải mô hình cơ sở nếu chưa được tải
            if not self.is_model_loaded:
                self.load_models()
            
            # Chuẩn bị mô hình để huấn luyện
            self.model.train()
            
            # Cập nhật trạng thái
            with self.training_lock:
                self.training_status["status"] = "training"
                self.training_status["message"] = "Đang huấn luyện mô hình..."
            
            # Mô phỏng quá trình huấn luyện
            self._simulate_training(epochs, len(training_data), batch_size)
            
            # Lưu mô hình
            model_save_path = models_dir / model_name
            self.model.save_pretrained(str(model_save_path))
            self.tokenizer.save_pretrained(str(model_save_path))
            
            # Cập nhật trạng thái hoàn thành
            with self.training_lock:
                self.training_status["is_training"] = False
                self.training_status["progress"] = 100.0
                self.training_status["current_epoch"] = epochs
                self.training_status["status"] = "completed"
                self.training_status["message"] = f"Huấn luyện hoàn tất. Mô hình đã được lưu tại {model_save_path}"
                self.training_status["end_time"] = datetime.now().isoformat()
            
            logger.info(f"Huấn luyện hoàn tất. Mô hình đã được lưu tại {model_save_path}")
            
            # Đặt lại model về chế độ evaluation
            self.model.eval()
            
        except Exception as e:
            # Cập nhật trạng thái lỗi
            with self.training_lock:
                self.training_status["is_training"] = False
                self.training_status["status"] = "error"
                self.training_status["message"] = f"Lỗi khi huấn luyện: {str(e)}"
                self.training_status["end_time"] = datetime.now().isoformat()
            
            logger.error(f"Lỗi khi huấn luyện mô hình: {str(e)}")
            
            # Đặt lại model về chế độ evaluation nếu có thể
            if self.is_model_loaded:
                self.model.eval()

    def _load_training_data(self, training_dir: str) -> List[Dict[str, str]]:
        """
        Tải dữ liệu huấn luyện từ thư mục
        
        Args:
            training_dir: Thư mục chứa dữ liệu huấn luyện
            
        Returns:
            Danh sách dữ liệu huấn luyện dạng [{"question": "", "context": "", "answer": ""}, ...]
        """
        training_data = []
        training_path = Path(training_dir)
        
        for file_path in training_path.glob("*"):
            try:
                # Xử lý tệp JSON
                if file_path.suffix.lower() == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Kiểm tra nếu data là list
                    if isinstance(data, list):
                        for item in data:
                            if "question" in item and "context" in item and "answer" in item:
                                training_data.append(item)
                    
                # Xử lý tệp CSV
                elif file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path)
                    if all(col in df.columns for col in ["question", "context", "answer"]):
                        for _, row in df.iterrows():
                            training_data.append({
                                "question": row["question"],
                                "context": row["context"],
                                "answer": row["answer"]
                            })
                
                # Xử lý tệp Excel
                elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path)
                    if all(col in df.columns for col in ["question", "context", "answer"]):
                        for _, row in df.iterrows():
                            training_data.append({
                                "question": row["question"],
                                "context": row["context"],
                                "answer": row["answer"]
                            })
            
            except Exception as e:
                logger.warning(f"Lỗi khi đọc tệp {file_path}: {str(e)}")
        
        return training_data

    def _simulate_training(self, epochs: int, num_samples: int, batch_size: int):
        """
        Mô phỏng quá trình huấn luyện (giả lập)
        Trong ứng dụng thực tế, đây sẽ là nơi thực hiện quá trình huấn luyện thực sự
        
        Args:
            epochs: Số epochs huấn luyện
            num_samples: Số lượng mẫu dữ liệu
            batch_size: Kích thước batch
        """
        # Tính tổng số batch
        num_batches = max(1, num_samples // batch_size)
        
        # Mô phỏng huấn luyện qua từng epoch
        for epoch in range(1, epochs + 1):
            # Cập nhật epoch hiện tại
            with self.training_lock:
                self.training_status["current_epoch"] = epoch
                self.training_status["message"] = f"Đang huấn luyện epoch {epoch}/{epochs}..."
            
            # Mô phỏng huấn luyện qua từng batch
            for batch in range(1, num_batches + 1):
                # Tính tiến độ
                progress = ((epoch - 1) * num_batches + batch) / (epochs * num_batches) * 100
                
                # Cập nhật tiến độ
                with self.training_lock:
                    self.training_status["progress"] = progress
                
                # Tạm dừng để mô phỏng thời gian huấn luyện
                time.sleep(0.2)  # 0.2 giây cho mỗi batch

    def get_training_status(self) -> Dict[str, Any]:
        """
        Lấy trạng thái huấn luyện hiện tại
        
        Returns:
            Trạng thái huấn luyện
        """
        with self.training_lock:
            status_copy = self.training_status.copy()
        
        # Tính thời gian huấn luyện nếu đang huấn luyện
        if status_copy["is_training"] and status_copy["start_time"]:
            start_time = datetime.fromisoformat(status_copy["start_time"])
            elapsed_seconds = (datetime.now() - start_time).total_seconds()
            
            # Ước tính thời gian còn lại dựa trên tiến độ
            if status_copy["progress"] > 0:
                remaining_seconds = elapsed_seconds * (100 - status_copy["progress"]) / status_copy["progress"]
                status_copy["estimated_remaining"] = f"{int(remaining_seconds // 60)} phút {int(remaining_seconds % 60)} giây"
            
            # Thêm thời gian đã trôi qua
            status_copy["elapsed_time"] = f"{int(elapsed_seconds // 60)} phút {int(elapsed_seconds % 60)} giây"
        
        return status_copy

# Khởi tạo dịch vụ NLP
nlp_service = NLPService() 
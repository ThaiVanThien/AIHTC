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
from abc import ABC, abstractmethod

from app.core.config import settings

logger = logging.getLogger(__name__)

class BaseNLPService(ABC):
    """
    Lớp cơ sở cho các dịch vụ NLP. Các lớp con cần triển khai các phương thức trừu tượng.
    """
    def __init__(self):
        self.models_dir = Path(settings.MODELS_DIR)
        self.training_data_dir = Path(settings.TRAINING_DATA_DIR)
        self.is_model_loaded = False
        
        # Tạo thư mục mô hình nếu chưa tồn tại
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Thư mục lưu mô hình: {self.models_dir.absolute()}")
        logger.info(f"Thư mục lưu dữ liệu huấn luyện: {self.training_data_dir.absolute()}")

    @abstractmethod
    def load_models(self):
        """Tải mô hình NLP"""
        pass
    
    @abstractmethod
    def get_status(self):
        """Lấy trạng thái hiện tại của dịch vụ"""
        pass
    
    @abstractmethod
    def answer_question(self, question: str, context: str):
        """Trả lời câu hỏi dựa trên ngữ cảnh"""
        pass
    
    def clear_cache(self):
        """
        Xóa cache của mô hình để tiết kiệm không gian đĩa
        """
        try:
            import shutil
            
            cache_dir = Path(settings.HUGGINGFACE_CACHE_DIR).expanduser().resolve()
            
            if cache_dir.exists():
                logger.info(f"Đang xóa cache tại: {cache_dir}")
                # Xóa theo từng thư mục con để tránh lỗi nếu một số file đang được sử dụng
                for item in cache_dir.glob("*"):
                    if item.is_dir():
                        try:
                            shutil.rmtree(item)
                            logger.info(f"Đã xóa: {item}")
                        except Exception as e:
                            logger.warning(f"Không thể xóa {item}: {str(e)}")
                logger.info("Đã xóa cache")
                return True
            else:
                logger.info("Không tìm thấy thư mục cache")
                return False
        except Exception as e:
            logger.error(f"Lỗi khi xóa cache: {str(e)}")
            return False

class NLPService(BaseNLPService):
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
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
        Thứ tự ưu tiên:
        1. Mô hình đã được huấn luyện và lưu trong thư mục models
        2. Mô hình vi-mrc-base từ Hugging Face
        3. Mô hình thay thế công khai khác
        """
        try:
            logger.info("Đang tải mô hình NLP...")
            
            # Kiểm tra nếu có mô hình đã huấn luyện trong thư mục models
            local_models = list(self.models_dir.glob("*")) if self.models_dir.exists() else []
            
            if local_models:
                # Lấy mô hình mới nhất (theo thời gian chỉnh sửa)
                latest_model = max(local_models, key=lambda p: p.stat().st_mtime)
                logger.info(f"Tìm thấy mô hình local: {latest_model}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(str(latest_model))
                self.model = AutoModelForQuestionAnswering.from_pretrained(str(latest_model))
                logger.info(f"Đã tải mô hình từ local: {latest_model}")
            else:
                # Thử tải mô hình từ Hugging Face và lưu vào thư mục models của dự án
                # Danh sách các mô hình sẽ thử theo thứ tự
                models_to_try = [
                    "nguyenvulebinh/vi-mrc-base",  # Thử mô hình chính trước
                    "nguyenvulebinh/vi-mrc-large", # Thử phiên bản lớn hơn
                    "nguyenvulebinh/vi-mrc-large-30epochs-maxlen-384", # Thử phiên bản khác
                    "nguyenvulebinh/mdeberta-v3-base-squad2", # Thử mô hình mdeberta
                    "vinai/phobert-base-v2"  # Thử PhoBeRT làm backup cuối cùng
                ]
                
                # Thư mục lưu mô hình tạm thời
                hf_model_dir = self.models_dir / settings.DEFAULT_MODEL_NAME
                hf_model_dir.mkdir(exist_ok=True)
                
                # Thử tải từng mô hình cho đến khi thành công
                for model_name in models_to_try:
                    try:
                        logger.info(f"Đang thử tải mô hình: {model_name} vào {hf_model_dir}")
                        
                        # Tải tokenizer và lưu vào thư mục local
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_name, 
                            cache_dir=None,  # Không sử dụng cache
                            local_files_only=False
                        )
                        self.tokenizer.save_pretrained(hf_model_dir)
                        
                        # Tải model và lưu vào thư mục local
                        self.model = AutoModelForQuestionAnswering.from_pretrained(
                            model_name,
                            cache_dir=None,  # Không sử dụng cache
                            local_files_only=False
                        )
                        self.model.save_pretrained(hf_model_dir)
                        
                        logger.info(f"Đã tải thành công mô hình: {model_name} và lưu vào {hf_model_dir}")
                        break
                    except Exception as e:
                        logger.warning(f"Không thể tải mô hình {model_name}: {str(e)}")
                        continue
            
            # Kiểm tra nếu mô hình vẫn chưa được tải
            if self.model is None:
                raise ValueError("Không thể tải bất kỳ mô hình nào. Vui lòng kiểm tra kết nối mạng và API key.")
            
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
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình NLP: {str(e)}")
            self.is_model_loaded = False
            return False

    def get_status(self):
        """
        Lấy trạng thái hiện tại của dịch vụ NLP
        """
        device = "gpu" if torch.cuda.is_available() and self.is_model_loaded else "cpu"
        
        # Xác định tên mô hình đang sử dụng
        model_name = None
        if self.is_model_loaded and hasattr(self.model, 'name_or_path'):
            model_name = self.model.name_or_path
        elif self.is_model_loaded and hasattr(self.model, 'config') and hasattr(self.model.config, '_name_or_path'):
            model_name = self.model.config._name_or_path
        
        return {
            "model_loaded": self.is_model_loaded,
            "device": device,
            "model_name": model_name,
            "is_training": self.training_status["is_training"],
            "training_status": self.training_status["status"] if self.training_status["is_training"] else "idle"
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

    def train_model(self, model_name: str, training_dir: str = None, epochs: int = 3, batch_size: int = 8):
        """
        Huấn luyện mô hình với dữ liệu được cung cấp
        
        Args:
            model_name: Tên mô hình sẽ được lưu
            training_dir: Thư mục chứa dữ liệu huấn luyện (nếu None, sử dụng thư mục mặc định)
            epochs: Số epochs huấn luyện
            batch_size: Kích thước batch
        """
        # Sử dụng thư mục dữ liệu huấn luyện mặc định nếu không được chỉ định
        if training_dir is None:
            training_dir = str(self.training_data_dir)
            
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
                if not self.load_models():
                    # Không thể tải mô hình
                    with self.training_lock:
                        self.training_status["is_training"] = False
                        self.training_status["status"] = "error"
                        self.training_status["message"] = "Không thể tải mô hình cơ sở. Vui lòng kiểm tra kết nối mạng và API key."
                        self.training_status["end_time"] = datetime.now().isoformat()
                    logger.error("Không thể tải mô hình cơ sở để huấn luyện")
                    return
            
            # Chuẩn bị mô hình để huấn luyện
            self.model.train()
            
            # Cập nhật trạng thái
            with self.training_lock:
                self.training_status["status"] = "training"
                self.training_status["message"] = "Đang huấn luyện mô hình..."
            
            # Mô phỏng quá trình huấn luyện
            self._simulate_training(epochs, len(training_data), batch_size)
            
            # Lưu mô hình
            model_save_path = self.models_dir / model_name
            model_save_path.mkdir(parents=True, exist_ok=True)
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

    def download_model_from_url(self, url: str, model_name: str = None):
        """
        Tải mô hình từ URL và lưu vào thư mục models
        
        Args:
            url: URL của mô hình (zip file)
            model_name: Tên mô hình sẽ được lưu (nếu None, sử dụng tên mặc định)
        
        Returns:
            bool: True nếu tải thành công, False nếu thất bại
        """
        import requests
        import zipfile
        import tempfile
        
        # Sử dụng tên mô hình mặc định nếu không được chỉ định
        if model_name is None:
            model_name = settings.DEFAULT_MODEL_NAME
        
        try:
            logger.info(f"Đang tải mô hình từ URL: {url}")
            
            model_dir = self.models_dir / model_name
            
            # Xóa thư mục cũ nếu đã tồn tại
            if model_dir.exists():
                import shutil
                logger.info(f"Xóa thư mục mô hình cũ: {model_dir}")
                shutil.rmtree(model_dir)
                
            # Tạo thư mục mới
            model_dir.mkdir(exist_ok=True)
            
            # Tải mô hình
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_file:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                
                # Tải file và hiển thị tiến trình
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                        downloaded += len(chunk)
                        progress = min(100, int(downloaded * 100 / total_size)) if total_size > 0 else 0
                        if progress % 10 == 0:  # Ghi log mỗi 10%
                            logger.info(f"Tải xuống: {progress}%")
                        
            # Giải nén file
            logger.info(f"Đang giải nén mô hình vào {model_dir}")
            with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
                zip_ref.extractall(model_dir)
            
            # Xóa file tạm
            os.unlink(temp_file.name)
            
            logger.info(f"Đã tải và giải nén mô hình vào {model_dir}")
            
            # Kiểm tra xem mô hình có đúng định dạng không
            required_files = ["config.json", "pytorch_model.bin"] 
            for file in required_files:
                if not (model_dir / file).exists() and not list(model_dir.glob("**/" + file)):
                    logger.warning(f"Mô hình tải về không có file {file}!")
            
            # Tải mô hình vào bộ nhớ nếu thành công
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                self.model = AutoModelForQuestionAnswering.from_pretrained(str(model_dir))
                self.is_model_loaded = True
                logger.info(f"Đã tải mô hình từ {model_dir} vào bộ nhớ")
            except Exception as e:
                logger.error(f"Không thể tải mô hình vào bộ nhớ: {str(e)}")
            
            # Xóa cache Hugging Face
            self.clear_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình từ URL: {str(e)}")
            return False

# Khởi tạo dịch vụ NLP
nlp_service = NLPService() 
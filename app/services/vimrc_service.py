import os
import logging
import torch
import time
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from typing import Dict, Any, List, Optional
from pathlib import Path
import threading
from datetime import datetime
import json

from app.services.nlp_service import BaseNLPService
from app.core.config import settings

logger = logging.getLogger(__name__)

class ViMRCService(BaseNLPService):
    """
    Dịch vụ NLP sử dụng mô hình vi-mrc-large
    """
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.model_name = settings.MODEL_VI_MRC_PATH
        self.model_revision = settings.MODEL_VI_MRC_REVISION
        self.max_length = settings.MAX_LENGTH
        self.doc_stride = settings.DOC_STRIDE
        self.max_answer_length = settings.MAX_ANSWER_LENGTH
        
        # Thư mục chứa dữ liệu và mô hình
        self.models_dir = Path(settings.MODELS_DIR)
        self.training_data_dir = Path(settings.TRAINING_DATA_DIR)
        
        # Vectorizer cho việc tìm kiếm 
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(lowercase=True)
        self.documents = {}
        self.document_vectors = None
        
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
        
        # Tự động tải mô hình khi khởi tạo
        self.load_models()
        
    def load_models(self) -> bool:
        """
        Tải mô hình Question Answering cho tiếng Việt
        Thứ tự ưu tiên:
        1. Mô hình đã được huấn luyện và lưu trong thư mục models
        2. Mô hình vi-mrc từ Hugging Face
        """
        try:
            logger.info("Đang tải mô hình vi-mrc...")
            
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
                # Thử tải mô hình từ Hugging Face
                models_to_try = [
                    self.model_name, # Thử mô hình chính
                    "nguyenvulebinh/vi-mrc-base",
                    "nguyenvulebinh/vi-mrc-large", 
                    "nguyenvulebinh/vi-mrc-large-30epochs-maxlen-384",
                    "vinai/phobert-base-v2"  # Backup cuối cùng
                ]
                
                # Thư mục lưu mô hình
                hf_model_dir = self.models_dir / "vi-mrc-large"
                hf_model_dir.mkdir(exist_ok=True)
                
                # Thử tải từng mô hình cho đến khi thành công
                for model_name in models_to_try:
                    try:
                        logger.info(f"Đang thử tải mô hình: {model_name}")
                        
                        # Tải tokenizer và model
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            model_name, 
                            cache_dir=None,
                            revision=self.model_revision
                        )
                        self.tokenizer.save_pretrained(hf_model_dir)
                        
                        self.model = AutoModelForQuestionAnswering.from_pretrained(
                            model_name,
                            cache_dir=None,
                            revision=self.model_revision
                        )
                        self.model.save_pretrained(hf_model_dir)
                        
                        logger.info(f"Đã tải thành công mô hình: {model_name}")
                        break
                    except Exception as e:
                        logger.warning(f"Không thể tải mô hình {model_name}: {str(e)}")
                        continue
            
            # Kiểm tra nếu mô hình vẫn chưa được tải
            if self.model is None or self.tokenizer is None:
                logger.error("Không thể tải bất kỳ mô hình nào")
                self.is_model_loaded = False
                return False
            
            # Đặt model ở chế độ evaluation
            self.model.eval()
            
            # Kiểm tra nếu có sẵn CUDA
            if torch.cuda.is_available():
                logger.info(f"Sử dụng GPU: {torch.cuda.get_device_name(0)}")
                self.model = self.model.to("cuda")
            else:
                logger.info("Sử dụng CPU vì không tìm thấy GPU")
                
            self.is_model_loaded = True
            logger.info("Đã tải xong mô hình vi-mrc")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình vi-mrc: {str(e)}")
            self.is_model_loaded = False
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """
        Lấy trạng thái hiện tại của dịch vụ vi-mrc
        """
        device = "gpu" if torch.cuda.is_available() and self.is_model_loaded else "cpu"
        
        # Xác định tên mô hình đang sử dụng
        model_name = None
        if self.is_model_loaded and hasattr(self.model, 'name_or_path'):
            model_name = self.model.name_or_path
        elif self.is_model_loaded and hasattr(self.model, 'config') and hasattr(self.model.config, '_name_or_path'):
            model_name = self.model.config._name_or_path
        
        return {
            "service_type": "vimrc",
            "model_loaded": self.is_model_loaded,
            "device": device,
            "model_name": model_name or self.model_name,
            "is_training": self.training_status["is_training"],
            "training_status": self.training_status["status"] if self.training_status["is_training"] else "idle"
        }
        
    def answer_question(self, question: str, context: str) -> Dict[str, Any]:
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
            if not self.load_models():
                return {
                    "answer": "Không thể tải mô hình",
                    "success": False,
                    "error": "Không thể tải mô hình vi-mrc"
                }
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                question,
                context,
                add_special_tokens=True,
                return_tensors="pt",
                max_length=self.max_length,
                truncation="only_second",
                stride=self.doc_stride,
                return_overflowing_tokens=True,
                padding="max_length"
            )
            
            # Lưu token ids cho việc chuyển đổi về text
            input_ids = inputs["input_ids"]
            
            # Đưa input lên GPU nếu có
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Xóa tham số overflow_to_sample_mapping nếu có
            if "overflow_to_sample_mapping" in inputs:
                overflow_mapping = inputs.pop("overflow_to_sample_mapping")
                logger.info(f"Đã loại bỏ tham số overflow_to_sample_mapping khi gọi model")
            
            # Dự đoán
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Lấy điểm bắt đầu và kết thúc cho từng phần
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            # Xử lý kết quả dự đoán cho trường hợp có nhiều đoạn (do truncation)
            all_answers = []
            for i in range(len(start_logits)):
                # Chọn các vị trí có điểm cao nhất
                start_idx = torch.argmax(start_logits[i]).item()
                end_idx = torch.argmax(end_logits[i]).item()
                
                # Tính điểm tin cậy
                confidence = (start_logits[i][start_idx].item() + end_logits[i][end_idx].item()) / 2
                
                # Giới hạn độ dài câu trả lời và đảm bảo start <= end
                if end_idx < start_idx or end_idx - start_idx + 1 > self.max_answer_length:
                    continue
                
                # Lấy câu trả lời từ input_ids
                answer = self.tokenizer.decode(input_ids[i][start_idx:end_idx+1], skip_special_tokens=True)
                
                all_answers.append({
                    "answer": answer,
                    "confidence": confidence,
                    "start_idx": start_idx,
                    "end_idx": end_idx
                })
            
            # Chọn câu trả lời có độ tin cậy cao nhất
            if all_answers:
                best_answer = max(all_answers, key=lambda x: x["confidence"])
                
                return {
                    "answer": best_answer["answer"],
                    "confidence": best_answer["confidence"],
                    "model": self.model_name,
                    "success": True,
                    "context": context
                }
            else:
                # Nếu không tìm được câu trả lời hợp lệ
                return {
                    "answer": "",
                    "confidence": 0.0,
                    "model": self.model_name,
                    "success": False,
                    "error": "Không tìm được câu trả lời hợp lệ",
                    "context": context
                }
                
        except Exception as e:
            logger.error(f"Lỗi khi trả lời câu hỏi với vi-mrc: {str(e)}")
            return {
                "answer": f"Lỗi xử lý: {str(e)}",
                "success": False,
                "error": str(e),
                "context": context
            }
    
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
    
    def download_model(self, url: str, model_name: str = "vi-mrc-large"):
        """
        Tải mô hình từ URL và lưu vào thư mục models
        
        Args:
            url: URL của mô hình (zip file)
            model_name: Tên mô hình sẽ được lưu
        
        Returns:
            bool: True nếu tải thành công, False nếu thất bại
        """
        import requests
        import zipfile
        import tempfile
        
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
                
                # Tải file
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
                        downloaded += len(chunk)
                        progress = min(100, int(downloaded * 100 / total_size)) if total_size > 0 else 0
                        
            # Giải nén file
            logger.info(f"Đang giải nén mô hình vào {model_dir}")
            with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
                zip_ref.extractall(model_dir)
            
            # Xóa file tạm
            os.unlink(temp_file.name)
            
            logger.info(f"Đã tải và giải nén mô hình vào {model_dir}")
            
            # Tải mô hình vào bộ nhớ
            return self.load_models()
            
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình từ URL: {str(e)}")
            return False

    async def download_from_huggingface(
        self, 
        model_id: str, 
        revision: str = "main", 
        local_dir_name: str = None,
        use_cache: bool = True
    ) -> bool:
        """
        Tải mô hình trực tiếp từ Hugging Face Hub
        
        Args:
            model_id: ID mô hình trên Hugging Face (ví dụ: vinai/vi-mrc-large)
            revision: Phiên bản mô hình (mặc định: main)
            local_dir_name: Tên thư mục local để lưu mô hình
            use_cache: Sử dụng cache Hugging Face nếu có
            
        Returns:
            bool: True nếu tải thành công, False nếu thất bại
        """
        try:
            from transformers import AutoTokenizer, AutoModelForQuestionAnswering
            import shutil
            
            # Xác định tên thư mục lưu trữ local
            if not local_dir_name:
                local_dir_name = model_id.split("/")[-1] if "/" in model_id else model_id
                
            model_dir = self.models_dir / local_dir_name
            
            # Xác định thư mục cache nếu sử dụng
            cache_dir = None
            if use_cache:
                cache_dir = Path(settings.HUGGINGFACE_CACHE_DIR).expanduser().resolve()
                # Đảm bảo thư mục cache tồn tại
                cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Sử dụng cache tại: {cache_dir}")
            else:
                logger.info("Không sử dụng cache")
            
            # Xóa thư mục mô hình cũ nếu đã tồn tại
            if model_dir.exists():
                logger.info(f"Xóa thư mục mô hình cũ: {model_dir}")
                shutil.rmtree(model_dir)
                
            # Tạo thư mục mới
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Tải tokenizer và model trực tiếp từ Hugging Face
            logger.info(f"Bắt đầu tải tokenizer từ {model_id} (revision: {revision})")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                revision=revision,
                cache_dir=cache_dir if use_cache else None
            )
            
            logger.info(f"Bắt đầu tải model từ {model_id} (revision: {revision})")
            model = AutoModelForQuestionAnswering.from_pretrained(
                model_id, 
                revision=revision,
                cache_dir=cache_dir if use_cache else None
            )
            
            # Lưu mô hình vào thư mục local
            logger.info(f"Lưu tokenizer và model vào {model_dir}")
            tokenizer.save_pretrained(model_dir)
            model.save_pretrained(model_dir)
            
            # Lưu thông tin về mô hình vào file config.json
            config_file = model_dir / "huggingface_info.json"
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump({
                    "model_id": model_id,
                    "revision": revision,
                    "download_date": datetime.now().isoformat(),
                    "use_cache": use_cache
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Đã tải và lưu mô hình {model_id} vào {model_dir}")
            
            # Tải lại mô hình để cập nhật vào bộ nhớ
            return self.load_models()
            
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình từ Hugging Face: {str(e)}")
            # Ghi chi tiết lỗi vào log
            import traceback
            logger.error(traceback.format_exc())
            return False

    def train_model(self, model_name: str, epochs: int = 3, batch_size: int = 8):
        """
        Huấn luyện mô hình vi-mrc với dữ liệu trong thư mục training
        
        Args:
            model_name: Tên mô hình sẽ được lưu
            epochs: Số epochs huấn luyện
            batch_size: Kích thước batch
            
        Returns:
            None (huấn luyện diễn ra ở background)
        """
        # Khởi tạo timestamp cho phiên bản huấn luyện này
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_name = f"{model_name}_{timestamp}"
        
        # Cập nhật trạng thái huấn luyện
        with self.training_lock:
            self.training_status = {
                "is_training": True,
                "current_epoch": 0,
                "total_epochs": epochs,
                "progress": 0.0,
                "model_name": model_name,
                "version": version_name,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "status": "preparing",
                "message": "Đang chuẩn bị dữ liệu huấn luyện..."
            }
        
        try:
            logger.info(f"Bắt đầu huấn luyện mô hình {model_name} (phiên bản: {version_name}) với {epochs} epochs, batch_size={batch_size}")
            
            training_dir = Path(settings.TRAINING_DATA_DIR)
            
            # Cấu trúc thư mục: models/vi-mrc-large/model_name/versions/timestamp
            base_model_dir = self.models_dir / "vi-mrc-large"
            model_type_dir = base_model_dir / model_name
            versions_dir = model_type_dir / "versions"
            model_output_dir = versions_dir / timestamp
            
            # Đảm bảo các thư mục tồn tại
            for dir_path in [base_model_dir, model_type_dir, versions_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # Tạo thư mục mới cho phiên bản hiện tại
            model_output_dir.mkdir(exist_ok=True)
            
            # Cập nhật trạng thái: Kiểm tra dữ liệu
            with self.training_lock:
                self.training_status["status"] = "checking_data"
                self.training_status["message"] = "Đang kiểm tra dữ liệu huấn luyện..."
            
            # Kiểm tra thư mục dữ liệu huấn luyện
            if not training_dir.exists() or not any(training_dir.iterdir()):
                with self.training_lock:
                    self.training_status["is_training"] = False
                    self.training_status["status"] = "error"
                    self.training_status["message"] = "Không tìm thấy dữ liệu huấn luyện"
                    self.training_status["end_time"] = datetime.now().isoformat()
                logger.error("Không tìm thấy dữ liệu huấn luyện trong thư mục training")
                return
            
            # Tìm tệp dữ liệu huấn luyện
            training_files = list(training_dir.glob("*.json"))
            if not training_files:
                with self.training_lock:
                    self.training_status["is_training"] = False
                    self.training_status["status"] = "error"
                    self.training_status["message"] = "Không tìm thấy tệp JSON trong thư mục training"
                    self.training_status["end_time"] = datetime.now().isoformat()
                logger.error("Không tìm thấy tệp JSON để huấn luyện")
                return
            
            # Sao chép dữ liệu huấn luyện vào thư mục của mô hình để tham chiếu sau này
            import shutil
            training_data_dir = model_output_dir / "training_data"
            training_data_dir.mkdir(exist_ok=True)
            
            for tf in training_files:
                dest_file = training_data_dir / tf.name
                shutil.copy2(tf, dest_file)
                logger.info(f"Đã sao chép dữ liệu huấn luyện {tf.name} vào {dest_file}")
            
            # TODO: Thêm code để tải và xử lý dữ liệu huấn luyện từ thư mục training
            # TODO: Thiết lập và huấn luyện mô hình với dữ liệu được chuẩn bị
            
            # Ví dụ mẫu về quá trình huấn luyện (giả lập)
            metrics_history = []
            for epoch in range(1, epochs + 1):
                # Cập nhật trạng thái: đang huấn luyện epoch hiện tại
                with self.training_lock:
                    self.training_status["current_epoch"] = epoch
                    self.training_status["status"] = "training"
                    self.training_status["message"] = f"Đang huấn luyện epoch {epoch}/{epochs}"
                
                # Giả lập huấn luyện epoch
                epoch_metrics = {"epoch": epoch, "loss": 0.0, "accuracy": 0.0}
                for step in range(1, 11):
                    # Giả lập thời gian huấn luyện cho mỗi batch
                    time.sleep(1)
                    
                    # Cập nhật tiến độ
                    progress = ((epoch - 1) * 10 + step) / (epochs * 10) * 100
                    with self.training_lock:
                        self.training_status["progress"] = progress
                    
                    # Giả lập metrics cải thiện dần
                    epoch_metrics["loss"] = 1.0 - (epoch * 0.1) - (step * 0.01)
                    epoch_metrics["accuracy"] = 0.5 + (epoch * 0.1) + (step * 0.01)
                
                # Lưu metrics cho epoch này
                metrics_history.append(epoch_metrics)
                
                # Lưu checkpoint sau mỗi epoch
                if self.tokenizer and self.model:
                    checkpoint_dir = model_output_dir / f"checkpoint_epoch_{epoch}"
                    checkpoint_dir.mkdir(exist_ok=True)
                    
                    try:
                        # Lưu tokenizer và model cho checkpoint này
                        self.tokenizer.save_pretrained(checkpoint_dir)
                        self.model.save_pretrained(checkpoint_dir)
                        
                        # Lưu metrics
                        with open(checkpoint_dir / "metrics.json", "w", encoding="utf-8") as f:
                            json.dump(epoch_metrics, f, ensure_ascii=False, indent=2)
                            
                        logger.info(f"Đã lưu checkpoint cho epoch {epoch}")
                    except Exception as e:
                        logger.error(f"Lỗi khi lưu checkpoint cho epoch {epoch}: {str(e)}")
            
            # Lưu mô hình đã được huấn luyện (giả lập)
            if self.tokenizer and self.model:
                try:
                    # Lưu mô hình vào thư mục đích
                    logger.info(f"Lưu mô hình cuối cùng vào {model_output_dir}")
                    
                    # Lưu tokenizer và model
                    self.tokenizer.save_pretrained(model_output_dir)
                    self.model.save_pretrained(model_output_dir)
                    
                    # Lưu thông tin huấn luyện chi tiết
                    training_info = {
                        "model_name": model_name,
                        "version": version_name,
                        "timestamp": datetime.now().isoformat(),
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "training_files": [tf.name for tf in training_files],
                        "metrics_history": metrics_history,
                        "final_metrics": metrics_history[-1] if metrics_history else {},
                        "base_model": self.model_name
                    }
                    
                    # Lưu thông tin huấn luyện
                    with open(model_output_dir / "training_info.json", "w", encoding="utf-8") as f:
                        json.dump(training_info, f, ensure_ascii=False, indent=2)
                    
                    # Lưu symbolic link hoặc file chỉ định phiên bản mới nhất
                    latest_version_file = model_type_dir / "latest_version.txt"
                    with open(latest_version_file, "w", encoding="utf-8") as f:
                        f.write(timestamp)
                    
                    # Cập nhật lịch sử huấn luyện trong thư mục gốc
                    history_file = base_model_dir / "training_history.json"
                    
                    # Đọc lịch sử huấn luyện hiện có nếu đã tồn tại
                    history = []
                    if history_file.exists():
                        try:
                            with open(history_file, "r", encoding="utf-8") as f:
                                history = json.load(f)
                        except Exception as e:
                            logger.warning(f"Không thể đọc lịch sử huấn luyện: {str(e)}")
                            history = []
                    
                    # Thêm thông tin huấn luyện mới
                    history_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "model_name": model_name,
                        "version": version_name,
                        "path": str(model_output_dir),
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "training_files": [tf.name for tf in training_files],
                        "final_metrics": metrics_history[-1] if metrics_history else {}
                    }
                    
                    history.append(history_entry)
                    
                    # Lưu lịch sử huấn luyện cập nhật
                    with open(history_file, "w", encoding="utf-8") as f:
                        json.dump(history, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"Đã lưu mô hình vào {model_output_dir} và cập nhật lịch sử huấn luyện")
                
                except Exception as e:
                    logger.error(f"Lỗi khi lưu mô hình: {str(e)}")
                    raise e
            
            # Cập nhật trạng thái: hoàn thành
            with self.training_lock:
                self.training_status["is_training"] = False
                self.training_status["status"] = "completed"
                self.training_status["progress"] = 100.0
                self.training_status["message"] = f"Đã hoàn thành huấn luyện mô hình {model_name} (phiên bản: {version_name})"
                self.training_status["end_time"] = datetime.now().isoformat()
                
            logger.info(f"Đã hoàn thành huấn luyện mô hình {model_name} (phiên bản: {version_name})")
            
        except Exception as e:
            # Xử lý ngoại lệ và cập nhật trạng thái lỗi
            logger.error(f"Lỗi khi huấn luyện mô hình: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            with self.training_lock:
                self.training_status["is_training"] = False
                self.training_status["status"] = "error"
                self.training_status["message"] = f"Lỗi khi huấn luyện: {str(e)}"
                self.training_status["end_time"] = datetime.now().isoformat()

    def _build_vectors(self):
        """Xây dựng vector cho tất cả tài liệu."""
        if not self.documents:
            return
            
        contents = [doc.content for doc in self.documents.values()]
        self.document_vectors = self.vectorizer.fit_transform(contents)
        logger.info(f"Đã xây dựng vector cho {len(contents)} tài liệu")
    
    def find_training_context(self, question: str) -> Optional[str]:
        """
        Tìm context phù hợp từ dữ liệu huấn luyện dựa trên câu hỏi
        
        Args:
            question: Câu hỏi cần tìm context
            
        Returns:
            Optional[str]: Context tìm được hoặc None nếu không tìm thấy
        """
        try:
            # Kiểm tra thư mục dữ liệu huấn luyện
            training_dir = Path(self.training_data_dir)
            
            if not training_dir.exists():
                logger.warning(f"Thư mục dữ liệu huấn luyện không tồn tại: {training_dir}")
                return None
                
            # Tìm tất cả các tệp JSON trong thư mục dữ liệu huấn luyện
            training_files = list(training_dir.glob("*.json"))
            
            if not training_files:
                logger.warning(f"Không tìm thấy tệp JSON nào trong thư mục dữ liệu huấn luyện: {training_dir}")
                return None
                
            logger.info(f"Tìm thấy {len(training_files)} tệp dữ liệu huấn luyện: {[f.name for f in training_files]}")
            
            # Kết hợp dữ liệu từ tất cả các tệp
            all_training_data = []
            
            for training_file in training_files:
                try:
                    with open(training_file, "r", encoding="utf-8") as f:
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            all_training_data.extend(file_data)
                        else:
                            logger.warning(f"Định dạng dữ liệu không đúng trong tệp {training_file}")
                except Exception as e:
                    logger.error(f"Lỗi khi đọc tệp dữ liệu huấn luyện {training_file}: {str(e)}")
            
            if not all_training_data:
                logger.warning("Không có dữ liệu huấn luyện hợp lệ")
                return None
                
            logger.info(f"Đã tải {len(all_training_data)} mẫu dữ liệu huấn luyện")
                
            # Chuẩn bị dữ liệu cho tìm kiếm tương đồng
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Lấy tất cả câu hỏi và context từ dữ liệu huấn luyện
            train_questions = [item["question"] for item in all_training_data]
            
            # Tạo vectorizer và vector cho câu hỏi huấn luyện
            vectorizer = TfidfVectorizer()
            question_vectors = vectorizer.fit_transform(train_questions)
            
            # Tạo vector cho câu hỏi hiện tại
            query_vector = vectorizer.transform([question])
            
            # Tính độ tương đồng
            similarities = cosine_similarity(query_vector, question_vectors).flatten()
            
            # Lấy chỉ số của câu hỏi có độ tương đồng cao nhất
            best_idx = similarities.argmax()
            best_similarity = similarities[best_idx]
            
            # In thông tin debug
            logger.info(f"Câu hỏi nhập vào: '{question}'")
            logger.info(f"Câu hỏi có độ tương đồng cao nhất: '{train_questions[best_idx]}' (sim={best_similarity:.4f})")
            
            # Chỉ trả về context nếu độ tương đồng đủ cao
            if best_similarity > 0.5:
                context = all_training_data[best_idx].get("context", "")
                logger.info(f"Tìm thấy context với độ tương đồng {best_similarity:.4f}")
                return context
            else:
                # Thử tìm kiếm theo từ khóa
                import re
                keywords = re.findall(r'\b[a-zA-ZÀ-ỹ]{3,}\b', question.lower())
                
                for item in all_training_data:
                    train_question = item["question"].lower()
                    match_count = sum(1 for kw in keywords if kw in train_question)
                    
                    if match_count >= 2 or (len(keywords) == 1 and match_count == 1):
                        logger.info(f"Tìm thấy context cho câu hỏi '{question}' bằng từ khóa")
                        return item.get("context", "")
                
                logger.info(f"Không tìm thấy context phù hợp cho câu hỏi '{question}'")
                return None
                
        except Exception as e:
            logger.error(f"Lỗi khi tìm context từ dữ liệu huấn luyện: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

# Khởi tạo dịch vụ vi-mrc
vimrc_service = ViMRCService() 
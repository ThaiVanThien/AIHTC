import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

class Document:
    """Lớp đại diện cho một tài liệu trong kho lưu trữ."""
    def __init__(self, content: str, metadata: Dict[str, Any] = None, doc_id: str = None):
        self.content = content
        self.metadata = metadata or {}
        self.doc_id = doc_id or self._generate_id()
        
    def _generate_id(self) -> str:
        """Tạo ID ngẫu nhiên cho tài liệu."""
        import uuid
        return str(uuid.uuid4())
        
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi tài liệu thành từ điển để lưu trữ."""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Tạo tài liệu từ từ điển."""
        return cls(
            content=data["content"],
            metadata=data.get("metadata", {}),
            doc_id=data.get("doc_id")
        )

class DocumentStore:
    """Kho lưu trữ và tìm kiếm tài liệu."""
    def __init__(self):
        self.data_dir = Path(settings.TRAINING_DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.data_dir / "document_index.json"
        self.documents: Dict[str, Document] = {}
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        self.document_vectors = None
        
        # Tải tài liệu nếu có sẵn
        self._load_documents()
        
    def _load_documents(self):
        """Tải tài liệu từ file index nếu tồn tại."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for doc_data in data:
                        doc = Document.from_dict(doc_data)
                        self.documents[doc.doc_id] = doc
                logger.info(f"Đã tải {len(self.documents)} tài liệu từ index")
                
                # Xây dựng vector cho tài liệu đã tải
                self._build_vectors()
            except Exception as e:
                logger.error(f"Lỗi khi tải tài liệu: {str(e)}")
        
    def _save_documents(self):
        """Lưu tài liệu vào file index."""
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump([doc.to_dict() for doc in self.documents.values()], f, ensure_ascii=False, indent=2)
            logger.info(f"Đã lưu {len(self.documents)} tài liệu vào index")
        except Exception as e:
            logger.error(f"Lỗi khi lưu tài liệu: {str(e)}")
    
    def _build_vectors(self):
        """Xây dựng vector cho tất cả tài liệu."""
        if not self.documents:
            return
            
        contents = [doc.content for doc in self.documents.values()]
        self.document_vectors = self.vectorizer.fit_transform(contents)
        logger.info(f"Đã xây dựng vector cho {len(contents)} tài liệu")
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> Document:
        """Thêm tài liệu mới vào kho lưu trữ."""
        doc = Document(content=content, metadata=metadata)
        self.documents[doc.doc_id] = doc
        
        # Cập nhật vectors
        self._build_vectors()
        
        # Lưu tài liệu
        self._save_documents()
        
        return doc
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Thêm nhiều tài liệu cùng lúc."""
        added = 0
        for doc_data in documents:
            content = doc_data.get("content")
            metadata = doc_data.get("metadata", {})
            
            if content:
                self.add_document(content, metadata)
                added += 1
                
        logger.info(f"Đã thêm {added} tài liệu mới")
        
        # Cập nhật vectors
        self._build_vectors()
        
        # Lưu tài liệu
        self._save_documents()
        
        return added
        
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Lấy tài liệu theo ID."""
        return self.documents.get(doc_id)
        
    def delete_document(self, doc_id: str) -> bool:
        """Xóa tài liệu theo ID."""
        if doc_id in self.documents:
            del self.documents[doc_id]
            
            # Cập nhật vectors
            self._build_vectors()
            
            # Lưu tài liệu
            self._save_documents()
            
            return True
        return False
    
    def search(self, query: str, top_k: int = 3) -> List[Document]:
        """Tìm kiếm tài liệu liên quan đến truy vấn."""
        if not self.documents or self.document_vectors is None:
            return []
            
        # Tạo vector cho truy vấn
        query_vector = self.vectorizer.transform([query])
        
        # Tính độ tương đồng cosine
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        
        # Sắp xếp theo độ tương đồng và lấy top_k kết quả
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Lọc các kết quả có độ tương đồng thấp
        results = []
        doc_ids = list(self.documents.keys())
        
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity > 0.1:  # Ngưỡng tối thiểu để một tài liệu được coi là liên quan
                doc_id = doc_ids[idx]
                doc = self.documents[doc_id]
                # Thêm điểm tương đồng vào metadata tạm thời
                doc_with_score = Document(
                    content=doc.content, 
                    metadata={**doc.metadata, "similarity_score": float(similarity)},
                    doc_id=doc.doc_id
                )
                results.append(doc_with_score)
        
        return results
    
    def keyword_search(self, keywords: List[str], top_k: int = 3) -> List[Document]:
        """Tìm kiếm tài liệu theo từ khóa."""
        if not self.documents:
            return []
            
        results = []
        
        # Tạo biểu thức chính quy từ danh sách từ khóa
        pattern = r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
        regex = re.compile(pattern, re.IGNORECASE)
        
        # Kiểm tra từng tài liệu
        for doc in self.documents.values():
            matches = regex.findall(doc.content)
            if matches:
                # Tính điểm dựa trên số lượng từ khóa khớp
                score = len(matches) / len(doc.content.split())
                doc_with_score = Document(
                    content=doc.content, 
                    metadata={**doc.metadata, "similarity_score": score},
                    doc_id=doc.doc_id
                )
                results.append(doc_with_score)
        
        # Sắp xếp kết quả theo điểm và lấy top_k
        results.sort(key=lambda x: x.metadata.get("similarity_score", 0), reverse=True)
        
        return results[:top_k]
    
    def classify_question_type(self, question: str) -> str:
        """Phân loại loại câu hỏi (factual/analytical)."""
        # Từ khóa cho câu hỏi dựa trên sự kiện cụ thể
        factual_keywords = [
            "bao nhiêu", "khi nào", "ở đâu", "là gì", "ai", "số tiền", 
            "tỷ lệ", "doanh thu", "chi phí", "quy định", "thời hạn",
            "hạn mức", "qui trình", "cách", "làm thế nào để", "định nghĩa"
        ]
        
        # Từ khóa cho câu hỏi phân tích
        analytical_keywords = [
            "tại sao", "giải thích", "phân tích", "đánh giá", 
            "so sánh", "tốt hay xấu", "nên", "có nên",
            "lợi ích", "hạn chế", "ảnh hưởng", "dự đoán"
        ]
        
        # Xử lý đặc biệt cho các câu hỏi định nghĩa "X là gì?"
        question_lower = question.lower()
        
        # Ưu tiên cao cho các câu hỏi định nghĩa
        if "là gì" in question_lower or "định nghĩa" in question_lower or "khái niệm" in question_lower:
            return "factual"
            
        # Đếm số từ khóa của mỗi loại
        factual_count = sum(1 for kw in factual_keywords if kw in question_lower)
        analytical_count = sum(1 for kw in analytical_keywords if kw in question_lower)
        
        # Quyết định dựa trên số lượng từ khóa và độ dài câu hỏi
        if factual_count > analytical_count:
            return "factual"  # Sử dụng VI-MRC
        elif analytical_count > factual_count:
            return "analytical"  # Sử dụng LLM
        else:
            # Nếu không có từ khóa rõ ràng, dựa vào độ dài câu hỏi
            # Câu hỏi ngắn thường là factual, câu hỏi dài thường là analytical
            return "factual" if len(question.split()) < 10 else "analytical"

    def extract_keywords(self, text: str) -> List[str]:
        """Trích xuất từ khóa quan trọng từ văn bản."""
        # Danh sách từ dừng tiếng Việt
        stop_words = [
            "và", "hay", "hoặc", "là", "của", "mà", "trong", "có", "được", "không",
            "những", "các", "với", "để", "cho", "về", "vì", "nhưng", "bởi", "bởi vì",
            "nên", "theo", "từ", "như", "thì", "khi", "vậy", "vào", "ra", "các"
        ]
        
        # Tách từ
        words = re.findall(r'\b[a-zA-ZÀ-ỹ]+\b', text.lower())
        
        # Loại bỏ từ dừng
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Đếm tần suất
        from collections import Counter
        word_counts = Counter(keywords)
        
        # Lấy các từ có tần suất cao
        top_keywords = [word for word, count in word_counts.most_common(10)]
        
        return top_keywords

# Khởi tạo DocumentStore singleton
document_store = DocumentStore() 
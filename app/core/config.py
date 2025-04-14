from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any, List, Union
from pydantic import validator
import os
from pathlib import Path


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "FastAPI Demo App"
    DESCRIPTION: str = "Ứng dụng demo sử dụng FastAPI và Swagger UI phiên bản mới nhất"
    VERSION: str = "0.1.0"
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Đường dẫn cho mô hình và dữ liệu
    MODELS_DIR: str = "./app/models"  # Thư mục lưu mô hình
    TRAINING_DATA_DIR: str = "./data/training"  # Thư mục lưu dữ liệu huấn luyện
    DEFAULT_MODEL_NAME: str = "vi-mrc-model"  # Tên mô hình mặc định
    HUGGINGFACE_CACHE_DIR: str = "~/.cache/huggingface"  # Thư mục cache của Hugging Face
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    
    # Model configs
    MODEL_VI_MRC_PATH: str = "vinai/vi-mrc-large"
    MODEL_VI_MRC_REVISION: str = "main"
    MAX_LENGTH: int = 512
    MAX_ANSWER_LENGTH: int = 100
    DOC_STRIDE: int = 128
    BATCH_SIZE: int = 16
    
    # Database settings
    DB_HOST: str = "localhost"
    DB_PORT: str = "5432"
    DB_NAME: str = "your_database_name"
    DB_USER: str = "your_database_user"
    DB_PASSWORD: str = "your_database_password"
    
    # Server settings
    PORT: int = 3000
    HOST: str = "0.0.0.0"
    DEBUG: bool = False
    
    @validator("MODELS_DIR", "TRAINING_DATA_DIR", pre=True)
    def create_directories(cls, v):
        # Tạo thư mục nếu không tồn tại
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())
    
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Swagger UI settings
    SWAGGER_UI_PARAMETERS: Dict[str, Any] = {
        "deepLinking": True,
        "persistAuthorization": True,
        "displayOperationId": True,
    }
    
    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()

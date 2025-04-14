from enum import Enum
from typing import Dict, List, Optional, Any
import os
from pathlib import Path
from pydantic import BaseModel, validator, Field


class AIProvider(str, Enum):
    """Định nghĩa các nhà cung cấp AI được hỗ trợ"""
    VIMRC = "vimrc"
    OPENAI = "openai"
    GEMINI = "gemini"


class ModelConfig(BaseModel):
    """Cấu hình cơ bản cho mô hình AI"""
    name: str
    display_name: str
    description: str = ""
    context_window: int = 4096
    max_output_tokens: int = 2048
    default_temperature: float = 0.7
    capabilities: List[str] = []
    is_available: bool = True


class ViMRCConfig(BaseModel):
    """Cấu hình cho mô hình VI-MRC"""
    model_path: str = "vinai/vi-mrc-large"
    model_revision: str = "main"
    max_length: int = 512
    max_answer_length: int = 100
    doc_stride: int = 128
    batch_size: int = 16
    model_name: str = "vi-mrc-large"
    models: List[ModelConfig] = [
        ModelConfig(
            name="vi-mrc-large",
            display_name="VI-MRC Large",
            description="Mô hình trả lời câu hỏi lớn cho tiếng Việt",
            capabilities=["question-answering"],
        ),
        ModelConfig(
            name="vi-mrc-base",
            display_name="VI-MRC Base",
            description="Mô hình trả lời câu hỏi cơ bản cho tiếng Việt",
            capabilities=["question-answering"],
        )
    ]


class OpenAIConfig(BaseModel):
    """Cấu hình cho dịch vụ OpenAI"""
    api_key: Optional[str] = None
    model_name: str = "gpt-3.5-turbo"
    base_url: str = "https://api.openai.com/v1"
    models: List[ModelConfig] = [
        ModelConfig(
            name="gpt-3.5-turbo",
            display_name="GPT-3.5 Turbo",
            description="Mô hình GPT-3.5 Turbo của OpenAI",
            context_window=4096,
            capabilities=["chat", "code", "embedding"],
        ),
        ModelConfig(
            name="gpt-4",
            display_name="GPT-4",
            description="Mô hình GPT-4 của OpenAI",
            context_window=8192,
            capabilities=["chat", "code", "reasoning"],
        ),
        ModelConfig(
            name="gpt-4-turbo",
            display_name="GPT-4 Turbo",
            description="Phiên bản cải tiến của GPT-4",
            context_window=128000,
            capabilities=["chat", "code", "reasoning", "vision"],
        ),
        ModelConfig(
            name="gpt-3.5-turbo-16k",
            display_name="GPT-3.5 Turbo 16K",
            description="GPT-3.5 với context window lớn hơn",
            context_window=16384,
            capabilities=["chat", "code"],
        )
    ]


class GeminiConfig(BaseModel):
    """Cấu hình cho dịch vụ Gemini"""
    api_key: Optional[str] = None
    model_name: str = "gemini-1.5-flash"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    temperature: float = 0.7
    max_output_tokens: int = 1024
    top_p: float = 0.8
    top_k: int = 40
    models: List[ModelConfig] = [
        ModelConfig(
            name="gemini-pro",
            display_name="Gemini Pro",
            description="Mô hình Gemini Pro của Google",
            context_window=32768,
            capabilities=["chat", "code", "reasoning"],
        ),
        ModelConfig(
            name="gemini-ultra",
            display_name="Gemini Ultra",
            description="Mô hình Gemini Ultra của Google",
            context_window=32768,
            capabilities=["chat", "code", "reasoning", "vision"],
        ),
        ModelConfig(
            name="gemini-pro-vision",
            display_name="Gemini Pro Vision",
            description="Gemini Pro với khả năng xử lý hình ảnh",
            context_window=16384,
            capabilities=["chat", "vision"],
        ),
        ModelConfig(
            name="gemini-1.5-flash",
            display_name="Gemini 1.5 Flash",
            description="Mô hình Gemini 1.5 Flash - nhanh và hiệu quả",
            context_window=1000000,
            capabilities=["chat", "code", "reasoning", "vision"],
        ),
        ModelConfig(
            name="gemini-1.5-pro",
            display_name="Gemini 1.5 Pro",
            description="Mô hình Gemini 1.5 Pro - cân bằng giữa hiệu suất và tốc độ",
            context_window=1000000,
            capabilities=["chat", "code", "reasoning", "vision"],
        )
    ]


class AISettings(BaseModel):
    """Cấu hình chung cho toàn bộ hệ thống AI"""
    default_provider: AIProvider = AIProvider.VIMRC
    vimrc: ViMRCConfig = ViMRCConfig()
    openai: OpenAIConfig = OpenAIConfig()
    gemini: GeminiConfig = GeminiConfig()
    
    # Thư mục mô hình và dữ liệu
    models_dir: str = "./app/models"
    training_data_dir: str = "./data/training"
    
    # Cấu hình đánh giá
    enable_evaluation: bool = False
    log_prompts: bool = False
    log_responses: bool = False


# Tạo cấu hình AI mặc định, lấy thông tin từ biến môi trường
ai_settings = AISettings(
    default_provider=AIProvider(os.getenv("DEFAULT_AI_PROVIDER", AIProvider.VIMRC)),
    vimrc=ViMRCConfig(
        model_path=os.getenv("MODEL_VI_MRC_PATH", "vinai/vi-mrc-large"),
        model_revision=os.getenv("MODEL_VI_MRC_REVISION", "main"),
        model_name=os.getenv("DEFAULT_VIMRC_MODEL", "vi-mrc-large"),
    ),
    openai=OpenAIConfig(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=os.getenv("DEFAULT_OPENAI_MODEL", "gpt-3.5-turbo"),
    ),
    gemini=GeminiConfig(
        api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_KEY"),
        model_name=os.getenv("DEFAULT_GEMINI_MODEL", "gemini-1.5-flash"),
    ),
    models_dir=os.getenv("MODELS_DIR", "./app/models"),
    training_data_dir=os.getenv("TRAINING_DATA_DIR", "./data/training"),
)


def get_all_model_names() -> Dict[str, List[str]]:
    """Lấy danh sách tên tất cả các mô hình theo provider"""
    return {
        AIProvider.VIMRC: [model.name for model in ai_settings.vimrc.models],
        AIProvider.OPENAI: [model.name for model in ai_settings.openai.models],
        AIProvider.GEMINI: [model.name for model in ai_settings.gemini.models],
    }


def get_model_config(provider: AIProvider, model_name: str) -> Optional[ModelConfig]:
    """Lấy cấu hình của một mô hình cụ thể"""
    if provider == AIProvider.VIMRC:
        models = ai_settings.vimrc.models
    elif provider == AIProvider.OPENAI:
        models = ai_settings.openai.models
    elif provider == AIProvider.GEMINI:
        models = ai_settings.gemini.models
    else:
        return None
        
    for model in models:
        if model.name == model_name:
            return model
    
    return None


def is_valid_model(provider: AIProvider, model_name: str) -> bool:
    """Kiểm tra xem một mô hình có tồn tại và hợp lệ không"""
    model = get_model_config(provider, model_name)
    return model is not None and model.is_available 
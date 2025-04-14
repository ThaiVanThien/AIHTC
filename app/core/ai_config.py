from pydantic import BaseModel, Field
from enum import Enum
import os
from typing import Dict, List, Optional, Union, Any

class AIProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"

class OpenAIConfig(BaseModel):
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 30

class GeminiConfig(BaseModel):
    api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_KEY", ""))
    model: str = "gemini-1.5-pro"
    temperature: float = 0.7
    max_output_tokens: int = 1000
    top_p: float = 1.0
    top_k: int = 40
    timeout: int = 30

class AISettings(BaseModel):
    default_provider: AIProvider = AIProvider.OPENAI
    openai: OpenAIConfig = OpenAIConfig()
    gemini: GeminiConfig = GeminiConfig()

# Cấu hình mặc định cho AI
ai_settings = AISettings(
    default_provider=AIProvider(os.getenv("DEFAULT_AI_PROVIDER", AIProvider.OPENAI)),
    openai=OpenAIConfig(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1000")),
    ),
    gemini=GeminiConfig(
        api_key=os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_KEY", ""),
        model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
        temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.7")),
        max_output_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "1000")),
    )
) 
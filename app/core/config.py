from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any, List, Union
from pydantic import validator


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "FastAPI Demo App"
    DESCRIPTION: str = "Ứng dụng demo sử dụng FastAPI và Swagger UI phiên bản mới nhất"
    VERSION: str = "0.1.0"
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
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


settings = Settings()

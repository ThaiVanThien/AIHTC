import uvicorn
from app.main import APP_PORT

if __name__ == "__main__":
    # Chạy với workers=1 và không có reload để tránh xung đột CUDA
    uvicorn.run("app.main:app", host="127.0.0.1", port=APP_PORT, reload=False, workers=1) 
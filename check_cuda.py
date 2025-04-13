import torch
import sys

print("=== Thông tin PyTorch và CUDA ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Kiểm tra bộ nhớ GPU
    print("\n=== Thông tin bộ nhớ GPU ===")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:")
        print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  Allocated memory: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"  Reserved memory: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    
    # Kiểm tra cuDNN
    print("\n=== Thông tin cuDNN ===")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
    print(f"cuDNN benchmark mode: {torch.backends.cudnn.benchmark}")
    print(f"cuDNN deterministic mode: {torch.backends.cudnn.deterministic}")
else:
    print("CUDA không khả dụng. Kiểm tra cài đặt GPU driver và CUDA.")

print("\nThử khởi tạo tensor trên CUDA...")
try:
    # Thử tạo tensor trên GPU
    x = torch.rand(10, 10)
    if torch.cuda.is_available():
        x = x.cuda()
        print(f"Tensor đã được tạo trên {x.device}")
    else:
        print(f"Tensor đã được tạo trên {x.device} (không có CUDA)")
except Exception as e:
    print(f"Lỗi khi tạo tensor: {e}") 
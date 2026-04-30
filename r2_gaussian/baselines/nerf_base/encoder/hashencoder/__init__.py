# 延迟导入：使 HashEncoder 在 CUDA 不可用时也能导入
# 实际的后端加载在第一次使用时才触发
try:
    from .hashgrid import HashEncoder
except (ImportError, OSError, EnvironmentError) as e:
    print(f"[WARNING] Could not import hashencoder.HashEncoder: {e}")
    print("[WARNING] DN-Gaussian will fall back to frequency encoding.")
    HashEncoder = None

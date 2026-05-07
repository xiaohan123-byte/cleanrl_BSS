import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.get_device_name(0)}")
else:
    print("【原因】PyTorch 无法调用 GPU，请检查：")
    print("  1. 是否安装了 CPU-only 的 PyTorch？")
    print("  2. NVIDIA 驱动是否正常？")
    print("  3. CUDA/cuDNN 是否与 PyTorch 版本匹配？")
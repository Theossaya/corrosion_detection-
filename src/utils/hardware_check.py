import torch

def check_hardware():
    print("="*50)
    print("HARDWARE CHECK")
    print("="*50)
    
    # CPU
    import multiprocessing
    print(f"CPU cores: {multiprocessing.cpu_count()}")
    
    # GPU
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Test GPU speed
        import time
        x = torch.randn(1000, 1000).cuda()
        start = time.time()
        for _ in range(100):
            y = torch.mm(x, x)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"  GPU test (100 matmuls): {elapsed:.3f}s")
    else:
        print("⚠ No GPU detected - training will be slower")
        print("  Consider using Google Colab or Kaggle Notebooks")
    
    # RAM
    import psutil
    ram_gb = psutil.virtual_memory().total / 1e9
    print(f"RAM: {ram_gb:.1f} GB")
    
    if ram_gb < 8:
        print("⚠ Low RAM - may need to reduce batch sizes")
    
    print("="*50)

if __name__ == "__main__":
    check_hardware()
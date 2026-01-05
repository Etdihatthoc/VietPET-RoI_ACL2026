#!/usr/bin/env python
"""
Script kiểm tra versions của các thư viện ảnh hưởng đến GPU memory usage
"""

import sys
import subprocess

print("=" * 70)
print("GPU LIBRARIES VERSION CHECK")
print("=" * 70)

# 1. Python version
print(f"\n[Python]")
print(f"  Version: {sys.version}")
print(f"  Executable: {sys.executable}")

# 2. PyTorch
try:
    import torch
    print(f"\n[PyTorch]")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA Version (compiled): {torch.version.cuda}")
        print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
except ImportError:
    print("\n[PyTorch] NOT INSTALLED")

# 3. CUDA Runtime Version (from nvidia-smi)
print(f"\n[CUDA Runtime]")
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version,cuda_version',
                            '--format=csv,noheader'],
                           capture_output=True, text=True)
    if result.returncode == 0:
        driver, cuda = result.stdout.strip().split(', ')
        print(f"  Driver Version: {driver}")
        print(f"  CUDA Runtime Version: {cuda}")
    else:
        print("  nvidia-smi not available")
except FileNotFoundError:
    print("  nvidia-smi not found")

# 4. Transformers (PhoBERT)
try:
    import transformers
    print(f"\n[Transformers]")
    print(f"  Version: {transformers.__version__}")
except ImportError:
    print("\n[Transformers] NOT INSTALLED")

# 5. BitsAndBytes (4-bit quantization)
try:
    import bitsandbytes as bnb
    print(f"\n[BitsAndBytes]")
    print(f"  Version: {bnb.__version__}")
except ImportError:
    print("\n[BitsAndBytes] NOT INSTALLED")
except AttributeError:
    try:
        import bitsandbytes
        print(f"\n[BitsAndBytes]")
        print(f"  Installed (version not available)")
    except:
        print("\n[BitsAndBytes] NOT INSTALLED")

# 6. Einops
try:
    import einops
    print(f"\n[Einops]")
    print(f"  Version: {einops.__version__}")
except ImportError:
    print("\n[Einops] NOT INSTALLED")

# 7. TQDM
try:
    import tqdm
    print(f"\n[TQDM]")
    print(f"  Version: {tqdm.__version__}")
except ImportError:
    print("\n[TQDM] NOT INSTALLED")

# 8. NumPy
try:
    import numpy as np
    print(f"\n[NumPy]")
    print(f"  Version: {np.__version__}")
except ImportError:
    print("\n[NumPy] NOT INSTALLED")

# 9. Memory allocator info (PyTorch)
if torch.cuda.is_available():
    print(f"\n[PyTorch Memory Allocator]")
    print(f"  Allocator Backend: {torch.cuda.get_allocator_backend()}")
    try:
        # Get current memory allocator config
        import os
        alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not set')
        print(f"  PYTORCH_CUDA_ALLOC_CONF: {alloc_conf}")
    except:
        pass

print("\n" + "=" * 70)
print("END OF CHECK")
print("=" * 70)

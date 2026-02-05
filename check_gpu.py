#!/usr/bin/env python3
"""
Quick GPU diagnostic script to verify GPU is available and working.
Run this before training to ensure GPU is properly configured.
"""

import sys

def check_gpu():
    """Check GPU availability and configuration."""
    print("="*60)
    print("GPU DIAGNOSTIC CHECK")
    print("="*60)
    
    # Check PyTorch
    try:
        import torch
        print(f"\n✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available!")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  cuDNN version: {torch.backends.cudnn.version()}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\n  GPU {i}:")
                print(f"    Name: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
            
            # Test GPU computation
            print(f"\n  Testing GPU computation...")
            device = torch.device('cuda:0')
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            z = torch.matmul(x, y)
            print(f"  ✓ GPU computation test passed!")
            print(f"    Result tensor device: {z.device}")
            print(f"    Result tensor shape: {z.shape}")
            
            # Check memory
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f"\n  GPU Memory Status:")
            print(f"    Allocated: {allocated:.2f} GB")
            print(f"    Reserved: {reserved:.2f} GB")
            
        else:
            print("\n✗ CUDA is NOT available!")
            print("  Possible reasons:")
            print("    1. No NVIDIA GPU installed")
            print("    2. CUDA drivers not installed")
            print("    3. PyTorch was installed without CUDA support")
            print("    4. GPU is being used by another process")
            print("\n  To install PyTorch with CUDA support:")
            print("    Visit: https://pytorch.org/get-started/locally/")
            return False
            
    except ImportError:
        print("\n✗ PyTorch is not installed!")
        print("  Install with: pip install torch")
        return False
    
    # Check sb3-contrib
    try:
        from sb3_contrib import RecurrentPPO
        print(f"\n✓ sb3-contrib is installed")
        print(f"  RecurrentPPO is available")
    except ImportError:
        print(f"\n✗ sb3-contrib is not installed!")
        print(f"  Install with: pip install sb3-contrib")
        return False
    
    print(f"\n{'='*60}")
    print("GPU CHECK COMPLETE")
    print("="*60)
    print("\nIf GPU is available, training will use GPU for neural network operations.")
    print("Note: Environment rollouts will still run on CPU (this is normal for RL).")
    print("GPU usage will be highest during training steps.\n")
    
    return True

if __name__ == "__main__":
    success = check_gpu()
    sys.exit(0 if success else 1)

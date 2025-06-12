#!/usr/bin/env python3
"""
Utility script to check system shared memory and provide recommendations
for fixing PyTorch DataLoader shared memory issues.
"""

import os
import sys
import subprocess
import psutil
from pathlib import Path

def get_shared_memory_info():
    """Get shared memory information"""
    try:
        # Check /dev/shm
        shm_stats = psutil.disk_usage('/dev/shm')
        
        print("=== Shared Memory Information ===")
        print(f"Shared memory mount: /dev/shm")
        print(f"Total: {shm_stats.total / (1024**3):.2f} GB")
        print(f"Used: {shm_stats.used / (1024**3):.2f} GB")
        print(f"Free: {shm_stats.free / (1024**3):.2f} GB")
        print(f"Usage: {(shm_stats.used / shm_stats.total) * 100:.1f}%")
        
        return shm_stats
        
    except Exception as e:
        print(f"Error checking shared memory: {e}")
        return None

def check_system_memory():
    """Check overall system memory"""
    memory = psutil.virtual_memory()
    
    print("\n=== System Memory Information ===")
    print(f"Total RAM: {memory.total / (1024**3):.2f} GB")
    print(f"Available: {memory.available / (1024**3):.2f} GB")
    print(f"Used: {memory.used / (1024**3):.2f} GB")
    print(f"Usage: {memory.percent:.1f}%")
    
    return memory

def check_current_processes():
    """Check processes using shared memory"""
    print("\n=== Processes Using Shared Memory ===")
    
    try:
        # Run ipcs command to show shared memory segments
        result = subprocess.run(['ipcs', '-m'], capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("Could not run 'ipcs -m' command")
    except FileNotFoundError:
        print("'ipcs' command not found")
    except Exception as e:
        print(f"Error checking shared memory segments: {e}")

def provide_recommendations(shm_stats, memory):
    """Provide recommendations based on system state"""
    print("\n=== Recommendations ===")
    
    if shm_stats is None:
        print("❌ Could not check shared memory. Try running with sudo or check if /dev/shm exists.")
        return
    
    shm_free_gb = shm_stats.free / (1024**3)
    shm_total_gb = shm_stats.total / (1024**3)
    memory_total_gb = memory.total / (1024**3)
    
    print(f"Current shared memory: {shm_free_gb:.2f} GB free / {shm_total_gb:.2f} GB total")
    
    if shm_free_gb < 0.5:
        print("⚠️  CRITICAL: Very low shared memory (<500MB)")
        print("   Solutions:")
        print("   1. Run training with: python train_model.py --disable_multiprocessing")
        print("   2. Or use: python train_model.py --num_workers 0")
        print("   3. Increase shared memory size (see below)")
        
    elif shm_free_gb < 1.0:
        print("⚠️  WARNING: Low shared memory (<1GB)")
        print("   Solutions:")
        print("   1. Run training with: python train_model.py --num_workers 2")
        print("   2. Or increase shared memory size (see below)")
        
    elif shm_free_gb < 2.0:
        print("✅ Shared memory should be sufficient, but monitor usage")
        print("   If you still get errors, try: python train_model.py --num_workers 2")
        
    else:
        print("✅ Shared memory looks good!")
        print("   You can use the default settings")
    
    # Shared memory increase recommendations
    print(f"\n=== How to Increase Shared Memory ===")
    
    # Calculate recommended size (25% of RAM or current + 2GB, whichever is larger)
    recommended_size = max(memory_total_gb * 0.25, shm_total_gb + 2)
    recommended_size = min(recommended_size, memory_total_gb * 0.5)  # Cap at 50% of RAM
    
    print(f"Recommended shared memory size: {recommended_size:.0f}GB")
    print(f"Current size: {shm_total_gb:.1f}GB")
    
    if recommended_size > shm_total_gb:
        print("\n📝 To increase shared memory temporarily (until reboot):")
        print(f"   sudo mount -o remount,size={recommended_size:.0f}G /dev/shm")
        
        print("\n📝 To increase shared memory permanently:")
        print("   1. Edit /etc/fstab (backup first!):")
        print("      sudo cp /etc/fstab /etc/fstab.backup")
        print("      sudo nano /etc/fstab")
        print("   2. Find the line with 'tmpfs /dev/shm' and modify it to:")
        print(f"      tmpfs /dev/shm tmpfs defaults,size={recommended_size:.0f}G 0 0")
        print("   3. Reboot the system")
        
        print("\n📝 Alternative: Use Docker with increased shared memory:")
        print("   docker run --shm-size=4g your_image")

def check_pytorch_multiprocessing():
    """Check PyTorch multiprocessing configuration"""
    print("\n=== PyTorch Multiprocessing Check ===")
    
    try:
        import torch
        import torch.multiprocessing as mp
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Check multiprocessing method
        try:
            method = mp.get_start_method()
            print(f"Current multiprocessing method: {method}")
            
            if method == 'fork':
                print("⚠️  WARNING: Using 'fork' method which can cause issues")
                print("   The training script now uses 'spawn' method to avoid this")
            else:
                print("✅ Multiprocessing method looks good")
                
        except Exception as e:
            print(f"Could not check multiprocessing method: {e}")
            
    except ImportError:
        print("❌ PyTorch not installed")

def main():
    """Main function"""
    print("🔍 Checking system for PyTorch DataLoader shared memory issues...\n")
    
    # Check shared memory
    shm_stats = get_shared_memory_info()
    
    # Check system memory
    memory = check_system_memory()
    
    # Check current processes
    check_current_processes()
    
    # Check PyTorch
    check_pytorch_multiprocessing()
    
    # Provide recommendations
    provide_recommendations(shm_stats, memory)
    
    print("\n" + "="*60)
    print("💡 Quick Solutions:")
    print("   • For immediate fix: python train_model.py --disable_multiprocessing")
    print("   • For reduced workers: python train_model.py --num_workers 2")
    print("   • For batch size reduction: python train_model.py --batch_size 4")
    print("="*60)

if __name__ == "__main__":
    main() 
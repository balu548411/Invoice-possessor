#!/usr/bin/env python3
"""
Validation script to check if the invoice processing setup is working correctly
"""

import sys
import importlib
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'torch',
        'torchvision', 
        'transformers',
        'pytorch_lightning',
        'cv2',
        'numpy',
        'pandas',
        'matplotlib',
        'PIL',
        'sklearn',
        'albumentations'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                importlib.import_module('cv2')
            elif package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'sklearn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not found")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies are installed!")
        return True

def check_data_structure():
    """Check if training data structure is correct"""
    print("\n📁 Checking data structure...")
    
    training_data_path = Path("training_data")
    images_path = training_data_path / "images"
    labels_path = training_data_path / "labels"
    
    if not training_data_path.exists():
        print("❌ training_data/ directory not found")
        return False
    
    if not images_path.exists():
        print("❌ training_data/images/ directory not found")
        return False
    
    if not labels_path.exists():
        print("❌ training_data/labels/ directory not found")
        return False
    
    # Check for sample files
    image_files = list(images_path.glob("*.jpg"))
    json_files = list(labels_path.glob("*.json"))
    
    print(f"📸 Found {len(image_files)} image files")
    print(f"📄 Found {len(json_files)} annotation files")
    
    if len(image_files) == 0:
        print("⚠️  No image files found in training_data/images/")
    
    if len(json_files) == 0:
        print("⚠️  No annotation files found in training_data/labels/")
    
    if len(image_files) > 0 and len(json_files) > 0:
        print("✅ Data structure looks good!")
        return True
    else:
        print("⚠️  Data structure needs attention")
        return False

def check_src_modules():
    """Check if source modules can be imported"""
    print("\n🧩 Checking source modules...")
    
    modules_to_check = [
        'src.data_processing',
        'src.model_architecture', 
        'src.training',
        'src.inference'
    ]
    
    success = True
    
    for module in modules_to_check:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module} - {str(e)}")
            success = False
    
    return success

def test_basic_functionality():
    """Test basic functionality of key components"""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test data processing
        from src.data_processing import InvoiceDataProcessor
        print("✅ InvoiceDataProcessor import successful")
        
        # Test model architecture
        from src.model_architecture import InvoiceProcessingModel
        print("✅ InvoiceProcessingModel import successful")
        
        # Test training module
        from src.training import InvoiceProcessingLightningModule
        print("✅ Training module import successful")
        
        # Test inference module
        from src.inference import InvoiceProcessor
        print("✅ Inference module import successful")
        
        print("✅ All modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during functionality test: {e}")
        return False

def test_torch_setup():
    """Test PyTorch setup and CUDA availability"""
    print("\n🔥 Checking PyTorch setup...")
    
    try:
        import torch
        
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available - {torch.cuda.device_count()} GPU(s)")
            print(f"   Current device: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA not available - will use CPU")
        
        # Test basic tensor operations
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        print("✅ Basic tensor operations working")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch setup error: {e}")
        return False

def print_next_steps():
    """Print suggested next steps"""
    print("\n🚀 Next Steps:")
    print("1. Ensure your training data is properly formatted")
    print("2. Run: python train_model.py --help to see training options")
    print("3. Start training: python train_model.py")
    print("4. Monitor training with Weights & Biases")
    print("5. Use trained model: python inference_demo.py --model_path <checkpoint>")

def main():
    """Main validation function"""
    print("🔍 Invoice Processing Setup Validation")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Data Structure", check_data_structure),
        ("Source Modules", check_src_modules),
        ("PyTorch Setup", test_torch_setup),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ Error in {check_name}: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! You're ready to start training!")
        print_next_steps()
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        
        if any(name == "Dependencies" and not result for name, result in results):
            print("\n💡 Try running: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 
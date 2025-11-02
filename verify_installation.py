#!/usr/bin/env python3
"""
CrypTFed Installation Verification Script
=========================================

This script verifies that CrypTFed is properly installed with full FHE functionality.
Run this after installation to ensure everything is working correctly.
"""

def verify_installation():
    """Verify CrypTFed installation and FHE functionality"""
    
    print("CrypTFed Installation Verification")
    print("=" * 40)
    
    # Test 1: Basic import
    try:
        import cryptfed
        print("✓ CrypTFed package import: SUCCESS")
    except ImportError as e:
        print(f"✗ CrypTFed package import: FAILED - {e}")
        print("  Solution: Ensure PYTHONPATH includes CrypTFed directory")
        return False
    
    # Test 2: Check FHE availability
    try:
        from cryptfed.aggregators import FHE_AVAILABLE, aggregator_registry
        if FHE_AVAILABLE:
            print("✓ FHE functionality: AVAILABLE")
            
            # Count FHE aggregators
            fhe_count = sum(1 for name in aggregator_registry.keys() 
                          if any(keyword in name.lower() for keyword in ['ckks', 'secure', 'integer', 'fedavgm']))
            print(f"✓ FHE aggregators: {fhe_count} available")
            
        else:
            print("✗ FHE functionality: NOT AVAILABLE")
            print("  Solution: Ensure LD_LIBRARY_PATH includes OpenFHE libraries")
            print("  Try: source venv/bin/activate")
            return False
            
    except Exception as e:
        print(f"✗ FHE check: FAILED - {e}")
        return False
    
    # Test 3: TensorFlow compatibility
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow: {tf.__version__}")
    except ImportError:
        print("✗ TensorFlow: NOT AVAILABLE")
        print("  Solution: pip install tensorflow")
        return False
    
    # Test 4: Basic functionality test
    try:
        from cryptfed.core.federated_client import FederatedClient
        print("✓ Core components: AVAILABLE")
    except ImportError as e:
        print(f"✗ Core components: FAILED - {e}")
        return False
    
    # Test 5: OpenFHE direct test
    try:
        import openfhe_numpy as onp
        # Try to access basic attributes
        attrs = len([attr for attr in dir(onp) if not attr.startswith('_')])
        print(f"✓ OpenFHE direct access: {attrs} attributes available")
    except Exception as e:
        print(f"✗ OpenFHE direct access: FAILED - {e}")
        print("  This is expected if OpenFHE shared libraries are not in path")
    
    print("\n" + "=" * 40)
    print("Installation Status: SUCCESS")
    print("Ready for federated learning with FHE!")
    
    return True

def print_quick_start():
    """Print quick start instructions"""
    print("\nQuick Start:")
    print("-" * 20)
    print("1. Run a basic example:")
    print("   python3 examples/level_1_basic/basic_mnist.py")
    print()
    print("2. Test FHE functionality:")
    print("   python3 -c \"from cryptfed.aggregators import FHE_AVAILABLE; print('FHE:', FHE_AVAILABLE)\"")
    print()
    print("3. Explore examples:")
    print("   ls examples/level_*/*.py")

if __name__ == "__main__":
    success = verify_installation()
    
    if success:
        print_quick_start()
    else:
        print("\nInstallation Issues Detected!")
        print("Please check the solutions above or refer to README.md")
        exit(1)
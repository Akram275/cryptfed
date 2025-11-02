#!/bin/bash
"""
TensorFlow Privacy Compatibility Fix
===================================

This script fixes the compatibility issue between TensorFlow 2.20.0 and tensorflow-privacy.
"""

echo "🔧 Fixing TensorFlow Privacy Compatibility Issue"
echo "================================================="
echo ""

echo "📋 Current versions:"
python3 -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')" 2>/dev/null || echo "TensorFlow: Not installed"
python3 -c "import tensorflow_privacy as tfp; print(f'TensorFlow Privacy: {tfp.__version__}')" 2>/dev/null || echo "TensorFlow Privacy: Not working"
echo ""

echo "🚨 Problem: TensorFlow 2.20.0 is incompatible with current tensorflow-privacy versions"
echo ""

echo "💡 Solution: Install compatible versions"
echo ""

echo "1️⃣ Uninstalling current versions..."
pip uninstall -y tensorflow tensorflow-privacy tensorflow-estimator

echo ""
echo "2️⃣ Installing compatible versions..."
pip install tensorflow==2.15.0 tensorflow-privacy==0.8.12

echo ""
echo "3️⃣ Verifying installation..."
python3 -c "
import tensorflow as tf
print(f'✅ TensorFlow: {tf.__version__}')

try:
    import tensorflow_privacy as tfp
    print(f'✅ TensorFlow Privacy: {tfp.__version__}')
    
    # Test DP optimizer import
    from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
    print('✅ DP Optimizers: Available')
    
    # Test privacy analysis import
    from tensorflow_privacy.privacy.analysis import rdp_accountant
    print('✅ Privacy Analysis: Available')
    
    print('')
    print('🎉 SUCCESS! tensorflow-privacy is now working correctly.')
    print('   You can now use DP-SGD in CrypTFed.')
    
except Exception as e:
    print(f'❌ Error: {e}')
    print('')
    print('🔄 Alternative solution:')
    print('   pip install tensorflow==2.13.0 tensorflow-privacy==0.8.10')
"

echo ""
echo "✅ Setup complete!"
echo ""
echo "🧪 Test your setup:"
echo "   cd /home/akram/cryptfed"
echo "   python3 examples/level_2_intermediate/dp_sgd_federated_learning.py"
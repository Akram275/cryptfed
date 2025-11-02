#!/bin/bash
# Set PYTHONPATH to include the CrypTFed project root
export PYTHONPATH="/home/akram/cryptfed:$PYTHONPATH"

# Set LD_LIBRARY_PATH to include OpenFHE libraries
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/akram/cryptfed/venv/lib/python3.10/site-packages/openfhe/lib"

# Now you can run any script without sys.path.append and with full FHE support
echo "PYTHONPATH set for CrypTFed"
echo "LD_LIBRARY_PATH set for OpenFHE libraries"
echo "Full FHE functionality enabled!"
echo ""
echo "You can now run scripts directly:"
echo "python examples/level_1_basic/basic_mnist.py"
echo "python examples/level_2_intermediate/threshold_fhe_mnist.py"
echo "python examples/level_3_advanced/byzantine_robustness_comparison.py"
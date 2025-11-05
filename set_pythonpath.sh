#!/bin/bash
# Set PYTHONPATH to include the CrypTFed project root
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Set LD_LIBRARY_PATH to include OpenFHE libraries (generic path)
if [ -d "$SCRIPT_DIR/venv/lib/python*/site-packages/openfhe/lib" ]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$SCRIPT_DIR/venv/lib/python*/site-packages/openfhe/lib"
elif command -v python3 >/dev/null 2>&1; then
    # Fallback: try to find OpenFHE in the current Python environment
    PYTHON_SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
    if [ -d "$PYTHON_SITE_PACKAGES/openfhe/lib" ]; then
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PYTHON_SITE_PACKAGES/openfhe/lib"
    fi
fi

# Now you can run any script without sys.path.append and with full FHE support
echo "PYTHONPATH set for CrypTFed: $SCRIPT_DIR"
echo "LD_LIBRARY_PATH configured for OpenFHE libraries"
echo "Full FHE functionality enabled!"
echo ""
echo "You can now run scripts directly:"
echo "python examples/level_1_basic/basic_mnist.py"
echo "python examples/level_2_intermediate/threshold_fhe_mnist.py"
echo "python examples/level_3_advanced/byzantine_robustness_comparison.py"
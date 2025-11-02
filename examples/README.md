# Examples Overview

Welcome to the CrypTFed federated learning library examples! This directory contains three progressive levels of examples designed to help you learn and use CrypTFed effectively.

## Directory Structure

```
examples/
├── level_1_basic/          # Simple, quick-start examples
├── level_2_intermediate/   # Feature-rich examples  
├── level_3_advanced/       # Full library capabilities
└── README.md              # This file
```

## Learning Path

### Level 1: Basic Examples (< 100 lines)
**Perfect for:** First-time users, quick prototyping, understanding core concepts

- **`basic_mnist.py`** - Simple MNIST federated learning with plaintext aggregation
- **`basic_cifar10.py`** - Basic CIFAR-10 FL with CNN model
- **`basic_tabular.py`** - Tabular data FL with synthetic dataset

**Key Concepts:** Model definition, data partitioning, basic federated training

### Level 2: Intermediate Examples 
**Perfect for:** Users wanting to explore FHE and advanced features

- **`threshold_fhe_mnist.py`** - Threshold FHE cryptography with logging and benchmarking
- **`fhe_schemes_comparison.py`** - Compare different FHE schemes (BFV, BGV, CKKS)

**Key Concepts:** Threshold cryptography, FHE schemes, comprehensive logging, benchmarking, non-IID data

### Level 3: Advanced Examples
**Perfect for:** Research, production deployments, comprehensive evaluations

- **`byzantine_robustness_comparison.py`** - Full Byzantine robustness analysis
- **`test_ACSIncome.py`** - Real-world tabular data (ACS Income) with threshold FHE
- **`UCI_Adult_example.py`** - UCI Adult dataset with Dirichlet partitioning
- **`CIFAR10_example.py`** - Production-ready CIFAR-10 with full features
- **`MNIST_robust_example.py`** - Byzantine attacks vs robust aggregators

**Key Concepts:** Byzantine robustness, attack simulation, robust aggregators, real datasets, production deployment

## Quick Start

1. **Start with Level 1** - Run `basic_mnist.py` to understand the fundamentals
2. **Progress to Level 2** - Explore `threshold_fhe_mnist.py` for FHE capabilities  
3. **Master Level 3** - Use `byzantine_robustness_comparison.py` for research

## Prerequisites

- Python 3.8+
- TensorFlow 2.x
- All CrypTFed dependencies (see main README)

## Running Examples

```bash
# Level 1 - Basic
python examples/level_1_basic/basic_mnist.py

# Level 2 - Intermediate  
python examples/level_2_intermediate/threshold_fhe_mnist.py

# Level 3 - Advanced
python examples/level_3_advanced/byzantine_robustness_comparison.py
```

## Expected Outputs

- **CSV files** with detailed benchmarks and metrics
- **Console logs** showing training progress and results
- **Model evaluations** with accuracy and loss metrics

## Customization

Each example is designed to be easily customizable:
- Modify `num_clients`, `num_rounds` for different scales
- Change `fhe_scheme` to test different cryptographic approaches  
- Adjust `aggregator_name` to test different aggregation strategies
- Update `attack_type` to simulate different Byzantine behaviors

## Tips

- **Start simple** - Begin with Level 1 examples to understand the core workflow
- **Check logs** - Enable detailed logging to understand what's happening
- **Monitor resources** - FHE operations can be memory and compute intensive
- **Use benchmarks** - Leverage the built-in benchmarking for performance analysis

## Need Help?

- Check the main project README for installation instructions
- Review the API documentation for detailed parameter descriptions  
- Look at the source code comments for implementation details

# Changelog

All notable changes to CrypTFed will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-28

### Initial Release

#### Added
- **Core Federated Learning Framework**
  - `CrypTFed` main orchestrator class
  - `FederatedServer` with zero-knowledge design
  - `FederatedClient` with Byzantine behavior simulation
  - Client sampling and round management

- **Homomorphic Encryption Support**
  - CKKS scheme for approximate arithmetic
  - BFV/BGV schemes for exact arithmetic
  - Threshold cryptography (multi-party decryption)
  - OpenFHE integration with openfhe-numpy

- **Byzantine-Robust Aggregation**
  - Krum and Multi-Krum algorithms
  - FLAME (clustering-based defense)
  - FoolsGold (Sybil-resistant aggregation)
  - Trimmed Mean (statistical robustness)
  - FedProx with proximal term

- **Attack Simulations**
  - Sign flipping attacks
  - Gradient ascent attacks
  - Random noise injection
  - Label shuffling attacks

- **Benchmarking System**
  - Comprehensive metrics collection
  - FHE performance tracking
  - Network bandwidth monitoring
  - Model quality assessment
  - Memory usage profiling
  - CSV export functionality

- **Example Suite**
  - Level 1: Basic MNIST, CIFAR-10, tabular data
  - Level 2: Threshold FHE, scheme comparisons
  - Level 3: Byzantine robustness, real datasets

- **Documentation**
  - Complete API documentation
  - Progressive example tutorials
  - Installation and setup guides

#### Features
- **Privacy-Preserving**: Server never sees plaintext models
- **Scalable**: Supports 5-50+ clients efficiently
- **Flexible**: Compatible with TensorFlow/Keras models
- **Research-Ready**: Comprehensive attack and defense evaluation
- **Production-Ready**: Professional error handling and logging

#### Supported Datasets
- MNIST (handwritten digits)
- CIFAR-10 (natural images)
- UCI Adult (census income)
- ACS Income (American Community Survey)
- Synthetic tabular data

#### Technical Specifications
- **Python**: 3.8+ support
- **TensorFlow**: 2.8+ compatibility
- **FHE**: OpenFHE backend
- **Cryptography**: Single-key and threshold modes
- **Performance**: Optimized for real-world deployment

### Project Cleanup
- Removed legacy DP-SGD integration attempts
- Cleaned up dependencies and package structure
- Added professional packaging and distribution setup
- Comprehensive code organization and documentation

---

## [Unreleased]

### Planned Features
- **Additional FHE Schemes**: TFHE, FHEW support
- **More Aggregators**: Secure aggregation variants
- **Enhanced Attacks**: Model poisoning, backdoor attacks
- **Differential Privacy**: DP-SGD integration (stable version)
- **Distributed Computing**: Multi-node deployment support
- **Advanced Benchmarks**: Latency, throughput analysis

### Known Issues
- Large models may require significant memory for FHE operations
- Threshold cryptography setup requires careful key management
- Performance scales with model size and client count

---

## Version Numbering

- **Major.Minor.Patch** (e.g., 1.0.0)
- **Major**: Breaking API changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, minor improvements

## Support

For questions about changes or upgrading:
- Check the [Migration Guide](docs/migration.md)
- Review [Breaking Changes](docs/breaking-changes.md)
- Open an [Issue](https://github.com/cryptfed/cryptfed/issues)
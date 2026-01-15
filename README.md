# CrypTFed: Privacy-Preserving Federated Learning with Homomorphic Encryption

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.8+](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://www.tensorflow.org/)

**CrypTFed** is a comprehensive federated learning library that combines **privacy-preserving cryptography** with **Byzantine-robust aggregation algorithms**. It enables secure, decentralized machine learning with homomorphic encryption (FHE) while defending against malicious participants.
This project was supported by the French ANR project ANR-22-CE39-0002 EQUIHID. 

## Key Features

- **Homomorphic Encryption**: Support for CKKS, BFV, and BGV schemes (single-key & threshold)
- **Byzantine Robustness**: Multiple aggregation algorithms (Krum, FLAME, FoolsGold, etc.)
- **High Performance**: Optimized for real-world federated learning scenarios
- **Comprehensive Benchmarking**: Built-in performance and security metrics
- **Easy Integration**: Simple API compatible with TensorFlow/Keras models
- **Rich Examples**: From basic tutorials to advanced research scenarios

## Architecture

```
┌─────────────────┬──────────────────┬─────────────────┐
│   Federated     │    Byzantine     │   Homomorphic   │
│   Learning      │    Robustness    │   Encryption    │
├─────────────────┼──────────────────┼─────────────────┤
│ • FedAvg        │ • Krum/MultiKrum │ • CKKS          │
│ • FedProx       │ • FLAME          │ • BFV/BGV       │
│ • Client        │ • FoolsGold      │ • Threshold FHE │
│   Sampling      │ • Trimmed Mean   │ • OpenFHE       │
└─────────────────┴──────────────────┴─────────────────┘
```

## Protocol State Machine

CrypTFed implements a **formal protocol state machine** that ensures correct execution of federated learning protocols. The state machine is **always enabled** and provides:

- **Formal Verification**: All entities (orchestrator, server, clients) follow validated state transitions
- **Protocol Correctness**: Operations are executed in the correct sequence, preventing protocol violations
- **Audit Trails**: Complete history of all state transitions with timestamps and metadata
- **Runtime Validation**: Invalid transitions are caught immediately before they can cause issues

### State Flow

**Orchestrator (14 states)**: `UNINITIALIZED` → `INITIALIZING` → `SESSION_READY` → [per round: `ROUND_STARTING` → `DISTRIBUTING_MODEL` → `WAITING_FOR_CLIENTS` → `COLLECTING_UPDATES` → `AGGREGATING` → `DECRYPTING_MODEL` → `EVALUATING` → `ROUND_COMPLETE`] → `TRAINING_COMPLETE` → `TERMINATED`

**Server (10 states)**: `IDLE` ↔ `BROADCASTING_MODEL` → `RECEIVING_UPDATES` → `AGGREGATING_UPDATES` → `MODEL_UPDATED` → `IDLE`

**Client (13 states)**: `IDLE` → `RECEIVING_MODEL` → `TRAINING` → `ENCRYPTING_UPDATE` → `SENDING_UPDATE` → `WAITING` → `IDLE`

### Usage

The state machine runs automatically - no configuration needed:

```python
# State machine is always active
orchestrator = CrypTFed(model_fn, clients, test_data)
final_model = orchestrator.run()

# Access audit trail
audit = orchestrator.protocol_coordinator.generate_audit_report()
import json
with open('audit.json', 'w') as f:
    json.dump(audit, f, indent=2)
```

For a complete demonstration, see [`protocol_state_machine_demo.py`](examples/protocol_state_machine_demo.py).

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.8+
- Git

### Install from Source (Only Option Currently)
```bash
# Clone the repository
git clone https://github.com/Akram275/cryptfed.git
cd cryptfed

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install CrypTFed with all dependencies
pip install -e .
```

**Important**: The virtual environment activation automatically sets up:
- `PYTHONPATH` for package imports
- `LD_LIBRARY_PATH` for OpenFHE shared libraries
- Full FHE functionality

### Alternative: Manual Environment Setup
If you need to set up the environment manually:
```bash
# Set Python path for imports
export PYTHONPATH="/path/to/cryptfed:$PYTHONPATH"

# Set library path for OpenFHE (adjust Python version as needed)
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/path/to/cryptfed/venv/lib/python3.10/site-packages/openfhe/lib"
```

### Verifying Installation
Test that FHE functionality is working:
```bash
python3 verify_installation.py
```

Or quick check:
```bash
python3 -c "from cryptfed.aggregators import FHE_AVAILABLE; print('FHE Available:', FHE_AVAILABLE)"
```

Expected output:
```
FHE Available: True
```

If you see `FHE Available: False`, the OpenFHE shared libraries may not be in your library path. Use the manual environment setup above.

### Troubleshooting
Common issues and solutions:

**Issue**: `ImportError: libOPENFHEbinfhe.so.1: cannot open shared object file`
**Solution**: Ensure `LD_LIBRARY_PATH` includes the OpenFHE library directory as shown above.

**Issue**: `ModuleNotFoundError: No module named 'cryptfed'`
**Solution**: Ensure `PYTHONPATH` includes the CrypTFed root directory.

**Issue**: FHE aggregators not available
**Solution**: Run `source venv/bin/activate` to properly set up the environment.

## Quick Start

### Setup Environment
```bash
cd /path/to/cryptfed
source venv/bin/activate  # Sets up PYTHONPATH and LD_LIBRARY_PATH automatically
```

### Basic Federated Learning with FHE
```python
import tensorflow as tf
from cryptfed import CrypTFed
from cryptfed.core.federated_client import FederatedClient

# Define your model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and distribute data (example with MNIST)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create federated clients
clients = []
for i in range(5):
    start_idx = i * 12000
    end_idx = start_idx + 12000
    
    client = FederatedClient(
        client_id=f"client_{i}",
        model_fn=create_model,
        x_train=x_train[start_idx:end_idx],
        y_train=y_train[start_idx:end_idx],
        local_epochs=1
    )
    clients.append(client)

# Configure federated learning with threshold FHE
orchestrator = CrypTFed(
    model_fn=create_model,
    clients=clients,
    test_data=(x_test, y_test),
    crypto_setting="threshold",          # Distributed trust
    fhe_scheme="threshold_ckks",         # CKKS with threshold
    threshold_parties=3,                 # Require 3 parties for decryption
    num_rounds=10
)

# Train with privacy guarantees
final_model = orchestrator.run()

# Export comprehensive benchmarks
orchestrator.evaluate_and_export("benchmarks.csv")
```

### Byzantine-Robust Federated Learning
```python
# Add Byzantine clients
byzantine_client = FederatedClient(
    client_id="attacker_1",
    model_fn=create_model,
    x_train=x_train[:1000],
    y_train=y_train[:1000],
    local_epochs=1,
    byzantine=True,                      # Enable Byzantine behavior
    attack_type="sign_flipping"          # Attack strategy
)

clients.append(byzantine_client)

# Use robust aggregation
orchestrator = CrypTFed(
    model_fn=create_model,
    clients=clients,
    test_data=(x_test, y_test),
    use_fhe=False,                       # Plaintext for robust aggregation
    aggregator_name="krum",              # Byzantine-robust aggregator
    aggregator_args={"f": 2},            # Tolerate up to 2 Byzantine clients
    num_rounds=10
)
```

## Examples

### Level 1: Basic Examples
- [`basic_mnist.py`](examples/level_1_basic/basic_mnist.py) - Simple MNIST federated learning
- [`basic_cifar10.py`](examples/level_1_basic/basic_cifar10.py) - CIFAR-10 with CNN
- [`basic_tabular.py`](examples/level_1_basic/basic_tabular.py) - Tabular data example

### Level 2: Intermediate Examples  
- [`threshold_fhe_mnist.py`](examples/level_2_intermediate/threshold_fhe_mnist.py) - Threshold cryptography
- [`fhe_schemes_comparison.py`](examples/level_2_intermediate/fhe_schemes_comparison.py) - Compare FHE schemes

### Level 3: Advanced Examples
- [`byzantine_robustness_comparison.py`](examples/level_3_advanced/byzantine_robustness_comparison.py) - Full robustness evaluation
- [`UCI_Adult_example.py`](examples/level_3_advanced/UCI_Adult_example.py) - Real-world tabular data

## Supported Algorithms

### FHE Schemes
- **CKKS**: Approximate arithmetic for real numbers
- **BFV/BGV**: Exact arithmetic for integers  
- **Threshold variants**: Distributed decryption requiring multiple parties

### Aggregation Methods
| Algorithm | Privacy | Byzantine Robust | Description |
|-----------|---------|------------------|-------------|
| FedAvg | FHE | | Standard federated averaging |
| Krum | | | Distance-based robust aggregation |
| FLAME | | | Clustering-based defense |
| FoolsGold | | | Sybil-resistant aggregation |
| Trimmed Mean | FHE | | Robust statistical aggregation |

### Attack Simulations
- **Sign Flipping**: Negates model updates
- **Gradient Ascent**: Reverses optimization direction
- **Random Noise**: Adds Gaussian noise to updates
- **Label Shuffling**: Corrupts training labels

## Benchmarking & Metrics

CrypTFed provides comprehensive benchmarking across multiple dimensions:

- **FHE Performance**: Encryption/decryption times, key generation
- **Network Overhead**: Ciphertext sizes, bandwidth usage  
- **Model Quality**: Accuracy, loss, convergence rates
- **System Resources**: Memory usage, computation time


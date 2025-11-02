# Level 3: Advanced Examples

Master-level examples showcasing the full power of CrypTFed: Byzantine robustness, real-world datasets, and production-ready federated learning.

## Examples in this Level

### 1. `byzantine_robustness_comparison.py` 
**Comprehensive Byzantine attack vs robust aggregator analysis**
- Tests all robust aggregators (Krum, Multi-Krum, FLAME, FoolsGold, etc.)
- ⚔️ Multiple Byzantine attack simulations
- Compares with threshold FHE baseline
- Full benchmark analysis and recommendations

```bash
python byzantine_robustness_comparison.py
```

### 2. `test_ACSIncome.py`
**Real-world tabular data federated learning with American Community Survey**
- Threshold FHE support
- Income prediction task

### 2. `test_ACSIncome.py`
**Real-world tabular data with state-based federation**
- ACS Income dataset (US Census Bureau)
- Geographic federation (different US states)
- Threshold FHE support
- Income prediction task

```bash
python test_ACSIncome.py
```

### 3. `UCI_Adult_example.py`
**UCI Adult dataset with Dirichlet partitioning**
- UCI Adult Income dataset
- Dirichlet non-IID distribution
- Configurable FHE schemes
- Production-ready setup

```bash
python UCI_Adult_example.py
```

### 4. `CIFAR10_example.py`
**Production-scale image classification**
- Full CIFAR-10 dataset
- Robust CNN architecture
- ⚖️ Multiple aggregation strategies
- Configurable for different scenarios

```bash
python CIFAR10_example.py
```

### 5. `MNIST_robust_example.py`
**Byzantine attack laboratory**
- Controlled Byzantine attack experiments
- Robust aggregator testing
- Comparative analysis
- Research-oriented setup

```bash
python MNIST_robust_example.py
```

## Advanced Learning Objectives

### Byzantine Robustness Mastery
Understanding when and how to use each robust aggregator:

```python
# For different threat models
aggregators = {
    "trimmed_mean": {"beta": 2},      # Basic outlier removal
    "krum": {"f": 4},                 # Single best update selection  
    "multi_krum": {"f": 4, "m": 8},   # Multiple good updates
    "flame": {"cluster_threshold": 0.6}, # Clustering-based
    "fools_gold": {"memory_size": 5},  # Historical analysis
}
```

### Production Deployment Patterns
Real-world considerations:

```python
# Production configuration
config = {
    "client_sampling_proportion": 0.3,    # Realistic participation
    "num_rounds": 100,                    # Longer training
    "enable_benchmarking": True,          # Monitor performance
    "crypto_setting": "threshold",        # Multi-party security
    "threshold_parties": 10,              # Higher security threshold
}
```

## Byzantine Attack Types

| Attack | Description | Impact | Best Defense |
|--------|-------------|--------|--------------|
| **Label Shuffling** | Corrupts training labels | High | Krum, Multi-Krum |
| **Sign Flipping** | Flips gradient directions | High | Trimmed Mean, FLAME |
| **Random Noise** | Adds noise to updates | Medium | FoolsGold, Median |
| **Gradient Ascent** | Maximizes loss instead of minimizing | High | Robust aggregators |

## Security vs Performance Trade-offs

### Plaintext Robust Aggregation
```python
# High robustness, no encryption
orchestrator = CrypTFed(
    use_fhe=False,
    aggregator_name="krum",  # Byzantine-robust
    aggregator_args={"f": num_byzantine_clients}
)
```

### Threshold FHE with Basic Aggregation  
```python
# High privacy, basic robustness
orchestrator = CrypTFed(
    use_fhe=True,
    crypto_setting="threshold", 
    aggregator_name="auto",  # Secure FedAvg
    threshold_parties=t
)
```

### Hybrid Approach (Decrypt → Robust Aggregate → Re-encrypt)
```python
# Both privacy and robustness (slower)
orchestrator = CrypTFed(
    use_fhe=True,
    aggregator_name="trimmed_mean",  # Requires plaintext
    # CrypTFed automatically handles decrypt/re-encrypt
)
```

## Advanced Benchmark Analysis

### Expected Output from Byzantine Comparison:
```
COMPREHENSIVE BYZANTINE ROBUSTNESS COMPARISON
Method                   Attack          Accuracy     Robustness
krum                    label_shuffling  0.8234      High
multi_krum              sign_flipping    0.8156      High  
flame                   random_noise     0.7834      Medium
trimmed_mean            label_shuffling  0.7756      Medium
plaintext_fedavg        label_shuffling  0.2341      Low
Threshold FHE (CKKS)    None (secure)    0.8445      Encrypted

Best robust aggregator: krum (0.8234)
Threshold FHE accuracy: 0.8445
Threshold FHE provides better accuracy but with privacy guarantees
```

## Production Considerations

### 1. **Scalability Testing**
```python
# Test with many clients
num_clients = 100
client_sampling_proportion = 0.1  # Only 10% participate per round

# Longer training
num_rounds = 500
local_epochs = 5
```

### 2. **Resource Monitoring**
```python
# Monitor system resources
import psutil
import time

def log_system_metrics():
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    # Log to benchmark manager
```

### 3. **Failure Handling**
```python
# Robust client handling
try:
    result = orchestrator.run()
except Exception as e:
    logger.error(f"Training failed: {e}")
    # Implement fallback strategy
```

## Research Applications

### Comparing Aggregation Strategies
Use these examples to research:
- **Robustness vs Accuracy trade-offs**
- **FHE overhead analysis** 
- **Attack effectiveness studies**
- **Non-IID impact on different aggregators**

### Publication-Ready Experiments
```python
# Research configuration
experiments = [
    {"attack": "label_shuffling", "aggregator": "krum", "f": 4},
    {"attack": "sign_flipping", "aggregator": "trimmed_mean", "beta": 2},
    {"attack": "random_noise", "aggregator": "flame", "threshold": 0.6},
    # ... systematic comparison
]

# Run systematic comparison
for exp in experiments:
    result = run_experiment(**exp)
    results.append(result)
```

## Advanced Customization

### Custom Attack Implementation
```python
# In FederatedClient
def _custom_backdoor_attack(self, weights):
    # Implement your custom attack
    modified_weights = []
    for w in weights:
        # Custom modification logic
        modified_weights.append(custom_modification(w))
    return modified_weights
```

### Custom Robust Aggregator
```python
# Extend BaseAggregator
class CustomRobustAggregator(BaseAggregator):
    @property 
    def requires_plaintext_updates(self) -> bool:
        return True
    
    def aggregate(self, updates, weights, **kwargs):
        # Your custom robust aggregation logic
        return robust_aggregate(updates)
```

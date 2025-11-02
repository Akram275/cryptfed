# Level 2: Intermediate Examples 

Explore advanced Synergia features including FHE encryption, threshold cryptography, and comprehensive benchmarking.

## 📚 Examples in this Level

### 1. `threshold_fhe_mnist.py`
**Threshold FHE with comprehensive logging**
- Threshold CKKS encryption (3-of-8 scheme)
- Non-IID data distribution  
- 📝 Comprehensive logging setup
- Benchmark manager integration
- Client sampling (75% participation)

```bash
python threshold_fhe_mnist.py
```

**Key Features:**
- Threshold cryptography requiring 3 clients to decrypt
- Non-IID data (each client has subset of digit classes)
- Detailed progress logging
- Benchmark CSV export

### 2. `fhe_schemes_comparison.py` 
**Compare different FHE schemes**
- 🔍 Tests BFV, BGV, CKKS, and Threshold CKKS
- ⚖️ Performance and accuracy comparison
- Automated benchmarking
- Identifies best-performing scheme

```bash
python fhe_schemes_comparison.py
```

**Key Features:**
- Side-by-side FHE scheme comparison
- Automated result analysis
- Performance metrics collection
- Recommendations based on results

## Learning Objectives

After completing Level 2, you'll understand:

1. **Threshold Cryptography**:
   ```python
   orchestrator = Synergia(
       crypto_setting="threshold",
       fhe_scheme="threshold_ckks", 
       threshold_parties=3,  # Need 3 clients to decrypt
       ...
   )
   ```

2. **Advanced Configuration**:
   ```python
   config = {
       "client_sampling_proportion": 0.75,  # Only 75% participate per round
       "aggregator_name": "auto",           # Scheme-specific aggregator
       "enable_benchmarking": True,         # Collect detailed metrics
   }
   ```

3. **Non-IID Data Simulation** - Realistic federated data distributions

4. **Comprehensive Logging** - Detailed progress tracking and debugging

5. **Benchmark Analysis** - Performance measurement and comparison

## FHE Schemes Overview

| Scheme | Best For | Precision | Performance |
|--------|----------|-----------|-------------|
| **CKKS** | Real numbers, neural networks | Approximate | Fast |
| **BFV** | Integers, exact computation | Exact | Medium |
| **BGV** | Integers, deep computation | Exact | Medium |
| **Threshold** | Multi-party security | Varies | Slower |

## Expected Outputs

### Threshold FHE Example:
```
Starting threshold FHE federated training...
Client mnist_client_0: 825 samples, classes: [0 1 2]
Client mnist_client_1: 742 samples, classes: [3 4 5]
...
Round 1: Sampled 6 of 8 clients.
Round 1/8 - Test Accuracy: 0.8234, Loss: 0.5123
...
Experiment completed! Benchmarks saved to mnist_threshold_ckks_benchmark.csv
```

### FHE Comparison:
```
FHE SCHEMES COMPARISON RESULTS
Scheme              Accuracy     Loss        
ckks               0.8456       0.4234      
threshold_ckks     0.8234       0.4567      
bfv                0.8123       0.4789      
bgv                0.8098       0.4823      

Best performing scheme: ckks (0.8456)
```

## Advanced Customization

### Custom Threshold Configuration:
```python
# More secure: need 5 out of 10 clients
config = {
    "crypto_setting": "threshold",
    "fhe_scheme": "threshold_bgv",
    "threshold_parties": 5,
    "num_clients": 10
}
```

### Non-IID Tuning:
```python
# More extreme non-IID (lower α = more heterogeneous)
def create_extreme_non_iid(x_train, y_train, num_clients):
    return create_non_iid_partition(x_train, y_train, num_clients, alpha=0.1)
```

### Advanced Logging:
```python
# Debug-level logging for troubleshooting
configure_logging(logging.DEBUG)

# Custom benchmark events
if orchestrator.server.benchmark_manager:
    orchestrator.server.benchmark_manager.log_event(
        'Custom', 'Special Metric', value, unit='custom'
    )
```

## Performance Tips

1. **FHE Optimization**:
   - Use smaller models for better FHE performance
   - Threshold schemes are slower than single-key
   - CKKS is generally fastest for neural networks

2. **Memory Management**:
   - FHE operations can be memory-intensive
   - Consider smaller batch sizes
   - Monitor system resources

3. **Debugging**:
   - Enable INFO logging to track progress
   - Use DEBUG for detailed troubleshooting
   - Check benchmark files for performance bottlenecks

## Performance Expectations

- **Threshold FHE**: 2-5x slower than plaintext
- **Single-key FHE**: 1.5-3x slower than plaintext  
- **Memory Usage**: 2-4x higher with FHE
- **Model Size**: Keep under 10K parameters for efficiency

## ➡️ Next Steps

Ready for Level 3? You should:
- Successfully run both examples
- Understand threshold cryptography concepts
- Know how to compare FHE schemes
- Be comfortable with advanced configuration

**Level 3 Preview**: Byzantine attacks, robust aggregators, production-scale datasets!
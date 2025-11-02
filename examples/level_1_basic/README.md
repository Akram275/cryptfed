# Level 1: Basic Examples

Simple, straightforward examples to get you started with CrypTFed federated learning.

## Examples in this Level

### 1. `basic_mnist.py` 
**Quick MNIST federated learning setup**
- MNIST digit classification
- Simple data splitting across 5 clients
- Basic MLP model  
- Plaintext federated averaging
- ~40 lines of code

```bash
python basic_mnist.py
```

### 2. `basic_cifar10.py`
**Image classification with CNN**
- CIFAR-10 dataset
- Simple CNN architecture
- 4 clients with equal data splits
- Quick demonstration of image FL

```bash
python basic_cifar10.py
```

### 3. `basic_tabular.py`
**Structured data federated learning**
- Synthetic tabular dataset
- Simple neural network
- 6 clients setup
- Binary classification task

```bash
python basic_tabular.py
```

## Learning Objectives

After running these examples, you should understand:

1. **Basic CrypTFed workflow**:
   ```python
   # Create clients
   clients = [FederatedClient(...) for i in range(num_clients)]
   
   # Create orchestrator  
   orchestrator = CrypTFed(model_fn, clients, test_data, ...)
   
   # Run federated training
   final_model = orchestrator.run()
   ```

2. **Model definition** - How to create TensorFlow/Keras models compatible with CrypTFed

3. **Data partitioning** - Simple ways to split datasets across clients

4. **Basic configuration** - Essential parameters for federated learning

## Quick Start

1. **Pick an example** that matches your data type (images vs tabular)
2. **Run it directly** - no configuration needed
3. **Modify the parameters** to experiment:
   - Change `num_clients` for different federation sizes
   - Adjust `num_rounds` for longer/shorter training
   - Modify `local_epochs` for more/less local training

## Expected Results

- **Console output** showing training progress
- **Final accuracy** printed at the end
- **Execution time** under 5 minutes for all examples

## Customization Examples

```python
# More clients
num_clients = 10

# Longer training  
num_rounds = 10
local_epochs = 3

# Different model
def create_larger_model():
    return models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'), 
        layers.Dense(10, activation='softmax')
    ])
```

## Next Steps

Once comfortable with Level 1:
- **Level 2**: Explore FHE encryption and advanced features
- **Level 3**: Byzantine robustness and production scenarios

## Success Criteria

You've mastered Level 1 when you can:
- Run all three examples successfully
- Modify client numbers and training rounds
- Understand the basic CrypTFed workflow
- Ready to explore encryption and advanced aggregation!
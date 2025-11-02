"""
Level 1 Example: Basic MNIST Federated Learning
================================================

This is the simplest possible federated learning example using CrypTFed.
- Uses MNIST dataset with simple data splitting
- Basic MLP model
- Plaintext federated averaging
- Minimal configuration (< 50 lines of actual code)
"""
import os
import random
import numpy as np

# Set deterministic environment variables BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF messages except errors
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA device detection
os.environ['PYTHONHASHSEED'] = '42'  # Make Python hash deterministic
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # TensorFlow deterministic operations

# Set seeds before TensorFlow import
random.seed(42)
np.random.seed(42)

import tensorflow as tf
from tensorflow.keras import layers, models

# Set TensorFlow seeds and deterministic operations
tf.random.set_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except AttributeError:
    pass  # Already set via environment variable

# Import CrypTFed (now installed as package)
from cryptfed import CrypTFed
from cryptfed.core.federated_client import FederatedClient

# Additional TensorFlow warning suppression
tf.get_logger().setLevel('ERROR')

def create_simple_model():
    """Simple MLP for MNIST with deterministic initialization."""
    # Ensure deterministic model initialization
    tf.random.set_seed(42)
    
    model = models.Sequential([
        layers.Input(shape=(28, 28)),  # Use Input layer instead of input_shape
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_initializer='glorot_uniform'),
        layers.Dense(10, activation='softmax', kernel_initializer='glorot_uniform')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_deterministic_model():
    """Wrapper to ensure all model creations are deterministic."""
    # Reset seeds before each model creation
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    return create_simple_model()

if __name__ == "__main__":
    # Ensure complete deterministic setup
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
    
    # Simple data split across 5 clients
    num_clients = 5
    samples_per_client = len(x_train) // num_clients
    
    clients = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        
        client = FederatedClient(
            client_id=f"client_{i}",
            model_fn=create_deterministic_model,
            x_train=x_train[start_idx:end_idx],
            y_train=y_train[start_idx:end_idx],
            local_epochs=1,
            deterministic_seed=SEED
        )
        clients.append(client)

    # Run federated learning (Encrypted with default CKKS scheme)
    # Create orchestrator
    orchestrator_fhe = CrypTFed(
        model_fn=create_deterministic_model,
        clients=clients,
        test_data=(x_test, y_test),
        use_fhe=True,               # Default scheme is single_key CKKS
        crypto_setting="threshold",
        fhe_scheme="threshold_ckks",
        aggregator_name="auto",
        num_rounds=2
    )

    orchestrator_pt = CrypTFed(
        model_fn=create_deterministic_model,
        clients=clients,
        test_data=(x_test, y_test),
        use_fhe=False,              # Use plaintext (non-encrypted) federated learning
        num_rounds=1
    )

    print("Starting basic federated learning...")
    final_model_fhe = orchestrator_fhe.run()
    final_model_pt = orchestrator_pt.run()
    
    # Get actual slot size used by the FHE scheme
    actual_slot_size = orchestrator_fhe.get_slot_size()
    print(f"FHE scheme slot size: {actual_slot_size}")
    
    # Evaluate both models
    loss_fhe, accuracy_fhe = final_model_fhe.evaluate(x_test, y_test, verbose=0)
    loss_pt, accuracy_pt = final_model_pt.evaluate(x_test, y_test, verbose=0)
    print(f"FHE model test accuracy: {accuracy_fhe:.4f}")
    print(f"Plaintext model test accuracy: {accuracy_pt:.4f}")

    # Analyze differences tensor by tensor - focused on chunking patterns
    fhe_weights = final_model_fhe.get_weights()
    pt_weights = final_model_pt.get_weights()
    
    print(f"\nDetailed chunking analysis:")
    print(f"Total tensors: {len(fhe_weights)}")
    
    # Focus on first tensor (784x128) to understand chunking pattern
    i = 0
    fhe_tensor, pt_tensor = fhe_weights[i], pt_weights[i]
    diff = np.abs(fhe_tensor - pt_tensor)
    
    print(f"\nTensor {i} (Dense layer weights): shape={fhe_tensor.shape}")
    print(f"Flattened size: {fhe_tensor.size}")
    
    # Analyze flattened differences to see chunking patterns
    flat_diff = diff.flatten()
    
    # Use actual slot size and some common multiples/divisors
    if actual_slot_size:
        chunk_sizes = [actual_slot_size // 4, actual_slot_size // 2, actual_slot_size, actual_slot_size * 2]
        chunk_sizes = [size for size in chunk_sizes if size > 0]  # Remove any zero or negative sizes
        print(f"Using actual slot size {actual_slot_size} and its multiples/divisors: {chunk_sizes}")
    else:
        chunk_sizes = [4096, 8192, 16384, 32768]  # Fallback to common FHE slot counts
        print(f"FHE not used, using common slot sizes: {chunk_sizes}")
    
    for chunk_size in chunk_sizes:
        print(f"\nAnalyzing with chunk_size={chunk_size}:")
        
        if len(flat_diff) <= chunk_size:
            print(f"  Tensor fits in one chunk ({len(flat_diff)} <= {chunk_size})")
            continue
            
        num_chunks = (len(flat_diff) + chunk_size - 1) // chunk_size
        print(f"  Would need {num_chunks} chunks")

        for chunk_idx in range(num_chunks):  # Check all chunks
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(flat_diff))
            chunk_diff = flat_diff[start_idx:end_idx]
            
            print(f"  Chunk {chunk_idx} (indices {start_idx}:{end_idx}):")
            print(f"    Size: {len(chunk_diff)}, Max diff: {np.max(chunk_diff):.6f}, Mean diff: {np.mean(chunk_diff):.6f}")
            
            # Check if chunk has padding-like pattern (higher differences at end)
            if len(chunk_diff) == chunk_size:
                last_10_pct = chunk_diff[-int(0.1*chunk_size):]
                first_90_pct = chunk_diff[:-int(0.1*chunk_size)]
                print(f"    Last 10% mean diff: {np.mean(last_10_pct):.6f}")
                print(f"    First 90% mean diff: {np.mean(first_90_pct):.6f}")
                
                if np.mean(last_10_pct) > 2 * np.mean(first_90_pct):
                    print(f"    *** POTENTIAL PADDING CORRUPTION DETECTED ***")
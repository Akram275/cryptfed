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
        layers.Input(shape=(28, 28)),   
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

    # Create orchestrator
    orchestrator_fhe = CrypTFed(
        model_fn=create_deterministic_model,
        clients=clients,
        test_data=(x_test, y_test),
        use_fhe=True,               # Default scheme is single_key CKKS
        crypto_setting="single_key",
        fhe_scheme="bgv",
        num_rounds=2
    )

    orchestrator_pt = CrypTFed(
        model_fn=create_deterministic_model,
        clients=clients,
        test_data=(x_test, y_test),
        use_fhe=False,              # Use plaintext (non-encrypted) federated learning
        num_rounds=2
    )

    print("Starting basic federated learning...")
    final_model_fhe = orchestrator_fhe.run()
    final_model_pt = orchestrator_pt.run()
    # Evaluate final model
    pt_loss, pt_accuracy = final_model_pt.evaluate(x_test, y_test, verbose=0)
    print(f"Plaintext test accuracy: {pt_accuracy:.4f}")
    fhe_loss, fhe_accuracy = final_model_fhe.evaluate(x_test, y_test, verbose=0)
    print(f"FHE test accuracy: {fhe_accuracy:.4f}")
       

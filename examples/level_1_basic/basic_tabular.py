"""
Level 1 Example: Basic Tabular Data FL
======================================

Simple federated learning example with synthetic tabular data.
- Synthetic tabular dataset
- Simple neural network
- Plaintext federated averaging
- Demonstrates FL on structured data
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF messages except errors
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA device detection
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import CrypTFed (now installed as package)
from cryptfed import CrypTFed
from cryptfed.core.federated_client import FederatedClient

# Additional TensorFlow warning suppression
tf.get_logger().setLevel('ERROR')

def create_tabular_model(input_dim):
    """Simple neural network for tabular data."""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),  # Use Input layer instead of input_shape
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Generate synthetic tabular data
    X, y = make_classification(
        n_samples=50000, n_features=20, n_informative=15, 
        n_redundant=5, n_classes=2, random_state=42
    )
    
    # Split and normalize
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Simple data split across 6 clients
    num_clients = 6
    samples_per_client = len(X_train) // num_clients
    
    clients = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        
        client = FederatedClient(
            client_id=f"tabular_client_{i}",
            model_fn=lambda: create_tabular_model(X_train.shape[1]),
            x_train=X_train[start_idx:end_idx],
            y_train=y_train[start_idx:end_idx],
            local_epochs=1
        )
        clients.append(client)

    # Run federated learning (encrypted with default CKKS scheme)
    # Create orchestrator  
    orchestrator = CrypTFed(
        model_fn=lambda: create_tabular_model(X_train.shape[1]),
        clients=clients,
        test_data=(X_test, y_test),
        use_fhe=True,  # The default scheme is single_key CKKS
        fhe_scheme="bfv",
        num_rounds=8
    )
    print("Starting tabular data federated learning...")
    final_model = orchestrator.run()
    # Evaluate final model
    loss, accuracy = final_model.evaluate(X_test, y_test, verbose=0)
    print(f"Final test accuracy: {accuracy:.4f}")

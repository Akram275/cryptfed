"""
Level 1 Example: Basic CIFAR-10 Federated Learning
==================================================

Simple federated learning example with CIFAR-10 using a basic CNN.
- CIFAR-10 dataset with simple data splitting
- Basic CNN model
- Encrypted federated averaging
- Quick demonstration of image classification FL
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF messages except errors
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA device detection
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Import CrypTFed 
from cryptfed import CrypTFed
from cryptfed.core.federated_client import FederatedClient

# Additional TensorFlow warning suppression
tf.get_logger().setLevel('ERROR')

def create_simple_cnn():
    """Simple CNN for CIFAR-10."""
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)), 
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
    y_train, y_test = y_train.flatten(), y_test.flatten()  # Flatten labels
    
    # Simple data split across 4 clients (smaller dataset)
    num_clients = 4
    samples_per_client = len(x_train) // num_clients
    
    clients = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        
        client = FederatedClient(
            client_id=f"cifar_client_{i}",
            model_fn=create_simple_cnn,
            x_train=x_train[start_idx:end_idx],
            y_train=y_train[start_idx:end_idx],
            local_epochs=1
        )
        clients.append(client)
    
    # Create orchestrator
    orchestrator = CrypTFed(
        model_fn=create_simple_cnn,
        clients=clients,
        test_data=(x_test, y_test),
        use_fhe=True,  # The default scheme is single_key CKKS
        fhe_scheme="bgv",
        aggregator_name="auto",
        num_rounds=5
    )
    
    print("Starting CIFAR-10 federated learning...")
    final_model = orchestrator.run()
    
    # Evaluate final model
    loss, accuracy = final_model.evaluate(x_test, y_test, verbose=0)
    print(f"Final test accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
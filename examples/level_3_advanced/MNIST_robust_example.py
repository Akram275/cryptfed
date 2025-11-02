"""
MNIST Federated Learning Example with Robust Aggregators

This example demonstrates federated learning on the MNIST dataset using various
robust aggregation algorithms. It simulates both honest and Byzantine clients
to test the robustness of different aggregators.

Features:
- Non-IID data distribution using Dirichlet partitioning
- Configurable Byzantine attacks (label flipping, noise injection, zero updates)
- Multiple robust aggregators for comparison
- Comprehensive evaluation and visualization
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import logging
from typing import List, Tuple

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF messages except errors
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA device detection
tf.get_logger().setLevel('ERROR')

import sys

# Import CrypTFed (now installed as package)
from cryptfed import CrypTFed, configure_logging
from cryptfed.core.federated_client import FederatedClient

# Configure logging to see progress
configure_logging(logging.INFO)

def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """Create a simple CNN model for MNIST classification."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def load_and_preprocess_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess the MNIST dataset."""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dimension for CNN
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    print(f"Training data: {x_train.shape}, Test data: {x_test.shape}")
    return x_train, y_train, x_test, y_test

def dirichlet_partition(data: np.ndarray, labels: np.ndarray, num_clients: int, alpha: float = 0.5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data using Dirichlet distribution to simulate non-IID federated setting.
    
    Args:
        data: Input data array
        labels: Corresponding labels
        num_clients: Number of clients to partition data for
        alpha: Dirichlet concentration parameter (lower = more non-IID)
    
    Returns:
        List of (data, labels) tuples for each client
    """
    print(f"Partitioning data for {num_clients} clients using Dirichlet(α={alpha})...")
    
    num_classes = len(np.unique(labels))
    label_distribution = np.random.dirichlet([alpha] * num_classes, num_clients)
    
    client_datasets = []
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    for client_id in range(num_clients):
        client_data = []
        client_labels = []
        
        for class_id in range(num_classes):
            # Calculate number of samples for this class for this client
            num_samples = int(label_distribution[client_id, class_id] * len(class_indices[class_id]))
            
            if num_samples > 0:
                # Randomly sample from this class
                selected_indices = np.random.choice(
                    class_indices[class_id], 
                    size=min(num_samples, len(class_indices[class_id])), 
                    replace=False
                )
                
                client_data.append(data[selected_indices])
                client_labels.append(labels[selected_indices])
                
                # Remove selected indices to avoid overlap
                class_indices[class_id] = np.setdiff1d(class_indices[class_id], selected_indices)
        
        if client_data:  # Ensure client has some data
            client_x = np.concatenate(client_data)
            client_y = np.concatenate(client_labels)
            
            # Shuffle client data
            perm = np.random.permutation(len(client_x))
            client_datasets.append((client_x[perm], client_y[perm]))
            
            print(f"  Client {client_id}: {len(client_x)} samples, classes: {np.unique(client_y)}")
        else:
            # Fallback: give at least some data to each client
            fallback_indices = np.random.choice(len(data), size=100, replace=False)
            client_datasets.append((data[fallback_indices], labels[fallback_indices]))
            print(f"  Client {client_id}: 100 samples (fallback)")
    
    return client_datasets

# Note: ByzantineClient class removed - using built-in Byzantine functionality in FederatedClient

def run_mnist_experiment(aggregator_name: str, aggregator_args: dict = {}, 
                        num_clients: int = 20, byzantine_clients: int = 4,
                        attack_type: str = "label_flipping", num_rounds: int = 15):
    """
    Run a single MNIST federated learning experiment.
    
    Args:
        aggregator_name: Name of the aggregator to use
        aggregator_args: Arguments for the aggregator
        num_clients: Total number of clients
        byzantine_clients: Number of Byzantine clients
        attack_type: Type of Byzantine attack
        num_rounds: Number of federated rounds
    
    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*60}")
    print(f"Running experiment with {aggregator_name} aggregator")
    print(f"Clients: {num_clients} ({byzantine_clients} Byzantine)")
    print(f"Attack: {attack_type}")
    print(f"Available attacks: sign_flipping, random_noise, gradient_ascent, label_shuffling")
    print(f"{'='*60}")
    
    # Load and partition data
    x_train, y_train, x_test, y_test = load_and_preprocess_mnist()
    client_datasets = dirichlet_partition(x_train, y_train, num_clients, alpha=0.5)
    
    # Create model function
    model_fn = lambda: create_cnn_model()
    
    # Create clients (mix of honest and Byzantine)
    clients = []
    for i in range(num_clients):
        x_client, y_client = client_datasets[i]
        
        if i < byzantine_clients:
            # Byzantine client using built-in functionality
            client = FederatedClient(
                client_id=f"byzantine_client_{i}",
                model_fn=model_fn,
                x_train=x_client,
                y_train=y_client,
                local_epochs=1,
                byzantine=True,
                attack_type=attack_type,
                attack_args={"scale": 1.0}  # For random_noise attack
            )
        else:
            # Honest client
            client = FederatedClient(
                client_id=f"honest_client_{i}",
                model_fn=model_fn,
                x_train=x_client,
                y_train=y_client,
                local_epochs=1,
                byzantine=False
            )
        
        clients.append(client)
    
    # Create and run federated learning experiment
    orchestrator = CrypTFed(
        model_fn=model_fn,
        clients=clients,
        test_data=(x_test, y_test),
        crypto_setting="single_key",  # Plaintext FL
        use_fhe=False,
        aggregator_name=aggregator_name,
        aggregator_args=aggregator_args,
        num_rounds=num_rounds,
        client_sampling_proportion=1.0  # All clients participate
    )
    
    # Run the experiment
    final_model = orchestrator.run()
    
    # Final evaluation
    loss, accuracy = final_model.evaluate(x_test, y_test, verbose=0)
    
    print(f"\nFinal Results for {aggregator_name}:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {loss:.4f}")
    
    return {
        'aggregator': aggregator_name,
        'accuracy': accuracy,
        'loss': loss,
        'byzantine_clients': byzantine_clients,
        'attack_type': attack_type
    }

def compare_aggregators():
    """Compare multiple aggregators on the same MNIST setup."""
    
    # Test configuration
    config = {
        'num_clients': 20,
        'byzantine_clients': 4,
        'attack_type': 'label_shuffling',  # Available: sign_flipping, random_noise, gradient_ascent, label_shuffling
        'num_rounds': 10
    }
    
    # Aggregators to test
    aggregators_to_test = [
        ("plaintext_fedavg", {}),
        ("trimmed_mean", {"beta": 2}),
        ("median", {}),
        ("krum", {"f": 4}),
        ("multi_krum", {"f": 4, "m": 12}),
        ("flame", {"cluster_threshold": 0.7}),
        ("fools_gold", {"memory_size": 5}),
    ]
    
    results = []
    
    print("Starting MNIST Robust Aggregators Comparison")
    print(f"Configuration: {config}")
    
    for aggregator_name, aggregator_args in aggregators_to_test:
        try:
            result = run_mnist_experiment(
                aggregator_name=aggregator_name,
                aggregator_args=aggregator_args,
                **config
            )
            results.append(result)
        except Exception as e:
            print(f"Error with {aggregator_name}: {e}")
            continue
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("AGGREGATOR COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"{'Aggregator':<20} {'Accuracy':<12} {'Loss':<12} {'Robustness'}")
    print("-" * 80)
    
    for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        robustness = "High" if result['accuracy'] > 0.85 else "Medium" if result['accuracy'] > 0.7 else "Low"
        print(f"{result['aggregator']:<20} {result['accuracy']:<12.4f} {result['loss']:<12.4f} {robustness}")
    
    return results

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run the comparison
    results = compare_aggregators()
    
    print(f"\nMNIST robust aggregation experiment completed!")
    print(f"Best performing aggregator: {max(results, key=lambda x: x['accuracy'])['aggregator']}")
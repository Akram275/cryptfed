"""
Level 3 Example: Advanced Byzantine-Robust Federated Learning
=============================================================

Advanced example demonstrating the full capabilities of CrypTFed:
- Byzantine clients with multiple attack types
- Robust aggregators comparison in plaintext mode
- Threshold FHE comparison with standard FedAvg
- Comprehensive benchmark analysis
- Real-world dataset (CIFAR-10) with non-IID distribution
- Full logging and metrics collection
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import logging
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Suppress CUDA warnings and TensorFlow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF messages except errors
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA device detection

# Import CrypTFed (now installed as package)
from cryptfed import CrypTFed, configure_logging
from cryptfed.core.federated_client import FederatedClient

# Configure comprehensive logging
configure_logging(logging.INFO)
logger = logging.getLogger(__name__)

def create_robust_cnn():
    """Create a robust CNN model for CIFAR-10."""
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),  # Use Input layer instead of input_shape
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

def create_non_iid_cifar(x_train, y_train, num_clients, alpha=0.5):
    """Create non-IID CIFAR-10 distribution using Dirichlet partitioning."""
    logger.info(f"Creating non-IID CIFAR-10 partition with Dirichlet(α={alpha})")
    
    num_classes = 10
    label_distribution = np.random.dirichlet([alpha] * num_classes, num_clients)
    
    client_datasets = []
    class_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
    
    for client_id in range(num_clients):
        client_data = []
        client_labels = []
        
        for class_id in range(num_classes):
            num_samples = int(label_distribution[client_id, class_id] * len(class_indices[class_id]))
            
            if num_samples > 0:
                selected_indices = np.random.choice(
                    class_indices[class_id], 
                    size=min(num_samples, len(class_indices[class_id])), 
                    replace=False
                )
                
                client_data.append(x_train[selected_indices])
                client_labels.append(y_train[selected_indices])
                
                class_indices[class_id] = np.setdiff1d(class_indices[class_id], selected_indices)
        
        if client_data:
            client_x = np.concatenate(client_data)
            client_y = np.concatenate(client_labels)
            
            perm = np.random.permutation(len(client_x))
            client_datasets.append((client_x[perm], client_y[perm]))
            
            logger.info(f"Client {client_id}: {len(client_x)} samples, classes: {np.unique(client_y)}")
    
    return client_datasets

def run_byzantine_experiment(aggregator_name, aggregator_args, attack_type, num_byzantine=4):
    """Run experiment with Byzantine clients and robust aggregation."""
    logger.info(f"Testing {aggregator_name} against {attack_type} attacks")
    
    # Load CIFAR-10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()
    
    # Use subset for faster execution
    subset_size = 8000
    indices = np.random.choice(len(x_train), subset_size, replace=False)
    x_train_subset = x_train[indices]
    y_train_subset = y_train[indices]
    
    # Create non-IID distribution
    num_clients = 12
    client_datasets = create_non_iid_cifar(x_train_subset, y_train_subset, num_clients)
    
    # Create clients (mix of honest and Byzantine)
    clients = []
    for i, (x_client, y_client) in enumerate(client_datasets):
        is_byzantine = i < num_byzantine
        
        client = FederatedClient(
            client_id=f"client_{i}{'_byzantine' if is_byzantine else '_honest'}",
            model_fn=create_robust_cnn,
            x_train=x_client,
            y_train=y_client,
            local_epochs=1,
            byzantine=is_byzantine,
            attack_type=attack_type if is_byzantine else None,
            attack_args={"scale": 2.0} if attack_type == "random_noise" else {}
        )
        clients.append(client)
    
    # Run plaintext federated learning with robust aggregation
    orchestrator = CrypTFed(
        model_fn=create_robust_cnn,
        clients=clients,
        test_data=(x_test, y_test),
        use_fhe=False,  # Plaintext for robust aggregation
        aggregator_name=aggregator_name,
        aggregator_args=aggregator_args,
        num_rounds=10,
        client_sampling_proportion=0.8,
        enable_benchmarking=True
    )
    
    final_model = orchestrator.run()
    loss, accuracy = final_model.evaluate(x_test, y_test, verbose=0)
    
    # Export results
    filename = f"byzantine_{aggregator_name}_{attack_type}_results.csv"
    orchestrator.evaluate_and_export(filename)
    
    return {
        "aggregator": aggregator_name,
        "attack": attack_type,
        "accuracy": accuracy,
        "loss": loss,
        "filename": filename
    }

def run_threshold_fhe_baseline():
    """Run threshold FHE experiment as a secure baseline."""
    logger.info("Running threshold FHE baseline (secure but non-robust)")
    
    # Load CIFAR-10 
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()
    
    # Smaller model for FHE efficiency
    def create_fhe_model():
        model = models.Sequential([
            layers.Input(shape=(32, 32, 3)),  # Use Input layer instead of input_shape
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'), 
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    # Use subset for faster FHE
    subset_size = 3000
    indices = np.random.choice(len(x_train), subset_size, replace=False)
    x_train_subset = x_train[indices]
    y_train_subset = y_train[indices]
    
    # Create non-IID distribution
    num_clients = 6
    client_datasets = create_non_iid_cifar(x_train_subset, y_train_subset, num_clients)
    
    clients = []
    for i, (x_client, y_client) in enumerate(client_datasets):
        client = FederatedClient(
            client_id=f"fhe_client_{i}",
            model_fn=create_fhe_model,
            x_train=x_client,
            y_train=y_client,
            local_epochs=1
        )
        clients.append(client)
    
    # Run threshold FHE
    orchestrator = CrypTFed(
        model_fn=create_fhe_model,
        clients=clients,
        test_data=(x_test, y_test),
        crypto_setting="threshold",
        fhe_scheme="threshold_ckks",
        threshold_parties=3,
        use_fhe=True,
        aggregator_name="auto",  # Uses secure FedAvg
        num_rounds=6,
        enable_benchmarking=True
    )
    
    final_model = orchestrator.run()
    loss, accuracy = final_model.evaluate(x_test, y_test, verbose=0)
    
    filename = "threshold_fhe_baseline.csv"
    orchestrator.evaluate_and_export(filename)
    
    return {
        "method": "Threshold FHE (CKKS)",
        "accuracy": accuracy,
        "loss": loss,
        "filename": filename
    }

def main():
    logger.info("Starting Level 3: Advanced Byzantine-Robust FL Comparison")
    
    # Test different robust aggregators against various attacks
    robust_tests = [
        # Aggregator, args, attack_type
        ("trimmed_mean", {"beta": 2}, "label_shuffling"),
        ("trimmed_mean", {"beta": 2}, "sign_flipping"),
        ("krum", {"f": 4}, "random_noise"),
        ("multi_krum", {"f": 4, "m": 8}, "label_shuffling"),
        ("flame", {"cluster_threshold": 0.6}, "sign_flipping"),
        ("fools_gold", {"memory_size": 5}, "random_noise"),
        ("plaintext_fedavg", {}, "label_shuffling"),  # Vulnerable baseline
    ]
    
    # Run Byzantine robustness experiments
    byzantine_results = []
    logger.info(f"Testing {len(robust_tests)} robust aggregator configurations...")
    
    for aggregator, args, attack in robust_tests:
        try:
            result = run_byzantine_experiment(aggregator, args, attack)
            byzantine_results.append(result)
            logger.info(f"{aggregator} vs {attack}: {result['accuracy']:.4f}")
        except Exception as e:
            logger.error(f"Error with {aggregator} vs {attack}: {e}")
    
    # Run threshold FHE baseline
    try:
        fhe_result = run_threshold_fhe_baseline()
        logger.info(f"Threshold FHE baseline: {fhe_result['accuracy']:.4f}")
    except Exception as e:
        logger.error(f"Error with threshold FHE: {e}")
        fhe_result = None
    
    # Print comprehensive comparison
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE BYZANTINE ROBUSTNESS COMPARISON")
    logger.info("="*80)
    logger.info(f"{'Method':<25} {'Attack':<15} {'Accuracy':<12} {'Robustness'}")
    logger.info("-" * 80)
    
    # Sort by accuracy
    for result in sorted(byzantine_results, key=lambda x: x['accuracy'], reverse=True):
        robustness = "High" if result['accuracy'] > 0.6 else "Medium" if result['accuracy'] > 0.4 else "Low"
        logger.info(f"{result['aggregator']:<25} {result['attack']:<15} {result['accuracy']:<12.4f} {robustness}")
    
    if fhe_result:
        logger.info(f"{fhe_result['method']:<25} {'None (secure)':<15} {fhe_result['accuracy']:<12.4f} Encrypted")
    
    # Analysis and recommendations
    if byzantine_results:
        best_robust = max(byzantine_results, key=lambda x: x['accuracy'])
        logger.info(f"\nBest robust aggregator: {best_robust['aggregator']} ({best_robust['accuracy']:.4f})")
        
        if fhe_result:
            logger.info(f"Threshold FHE accuracy: {fhe_result['accuracy']:.4f}")
            if fhe_result['accuracy'] > best_robust['accuracy']:
                logger.info("Threshold FHE provides better accuracy but with privacy guarantees")
            else:
                logger.info("Robust aggregation provides better accuracy in plaintext")
    
    logger.info("\nAdvanced Byzantine robustness comparison completed!")
    logger.info("📁 Check individual CSV files for detailed benchmark data")

if __name__ == "__main__":
    # Set seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    main()
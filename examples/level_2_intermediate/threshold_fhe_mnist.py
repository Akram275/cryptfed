"""
Level 2 Example: MNIST with Threshold FHE and Comprehensive Logging
==================================================================

Intermediate example demonstrating threshold FHE using the proven working pattern:
- Threshold FHE cryptography (CKKS/BFV schemes)
- Critical configuration for threshold success:
  * Client sampling proportion of 50% (not 100%)
  * Single local epoch per round (not multiple)
  * Minimal ResNet architecture (4 filters)
  * Gradient clipping for numerical stability
- Comprehensive logging setup
- Custom aggregator configuration  
- Benchmark manager capabilities

Based on successful new_test.py threshold pattern.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import logging
import os
import sys

# Suppress CUDA warnings and TensorFlow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF messages except errors
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA device detection

# Import CrypTFed (now installed as package)
from cryptfed import CrypTFed, configure_logging
from cryptfed.core.federated_client import FederatedClient

# Configure comprehensive logging
configure_logging(logging.INFO)
logger = logging.getLogger(__name__)

def create_mini_resnet_mnist():
    """Minimal ResNet model based on the working threshold pattern, adapted for MNIST."""
    inputs = layers.Input(shape=(28, 28, 1))
    # Reshape to have channel dimension
    x = layers.Reshape((28, 28, 1))(inputs)
    
    num_filters = 4  # Very small like new_test.py
    x = layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    res_input = x
    x = layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(num_filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, res_input])
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=x)

    # Use gradient clipping for numerical stability (critical for threshold FHE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=1.0)  # Match new_test.py exactly

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',  # Keep sparse labels
                  metrics=['accuracy'])
    return model

def create_simple_iid_partition(x_train, y_train, num_clients):
    """Create a simple IID partition for better FHE convergence."""
    logger.info(f"Creating simple IID partition for {num_clients} clients")
    
    # Simple random split - better for demonstrating FHE capabilities
    total_samples = len(x_train)
    samples_per_client = total_samples // num_clients
    
    # Shuffle data first
    indices = np.random.permutation(total_samples)
    x_shuffled = x_train[indices]
    y_shuffled = y_train[indices]
    
    client_datasets = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client if i < num_clients - 1 else total_samples
        
        client_x = x_shuffled[start_idx:end_idx]
        client_y = y_shuffled[start_idx:end_idx]
        client_datasets.append((client_x, client_y))
        
        unique_classes = np.unique(client_y)
        logger.info(f"Client {i}: {len(client_x)} samples, all {len(unique_classes)} classes present")
    
    return client_datasets

def main():
    logger.info("Starting Level 2: FHE MNIST Example")
    
    # Configuration - Match working new_test.py exactly for threshold FHE
    config = {
        "num_clients": 10,  # Match new_test.py
        "num_rounds": 20,   # Match new_test.py
        "local_epochs": 1,  # CRITICAL: Match new_test.py (was 5, causing issues)
        "client_sampling_proportion": 0.5,  # CRITICAL: Only 50% like new_test.py (was 1.0)
        
        # Threshold FHE settings - exact match to new_test.py
        "crypto_setting": "threshold",
        "fhe_scheme": "threshold_ckks",  # or "threshold_bfv" 
        "threshold_parties": 7,  # Match new_test.py exactly
        
        # Aggregator settings
        "aggregator_name": "auto",
        "aggregator_args": {}
    }
    
    logger.info(f"Configuration: {config}")
    
    # Load and preprocess MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Create simple IID data distribution (better for FHE demonstration)
    client_datasets = create_simple_iid_partition(x_train, y_train, config["num_clients"])
    
    # Create clients
    clients = []
    for i, (x_client, y_client) in enumerate(client_datasets):
        client = FederatedClient(
            client_id=f"mnist_client_{i}",
            model_fn=create_mini_resnet_mnist,
            x_train=x_client,
            y_train=y_client,
            local_epochs=config["local_epochs"],
            local_lr=0.001,  # Explicit local learning rate like new_test.py
            verbose=0
        )
        clients.append(client)
    
    logger.info(f"Created {len(clients)} clients with IID data distribution")
    
    # Create CrypTFed orchestrator with threshold FHE
    orchestrator = CrypTFed(
        model_fn=create_mini_resnet_mnist,
        clients=clients,
        test_data=(x_test, y_test),
        crypto_setting=config["crypto_setting"],
        fhe_scheme=config["fhe_scheme"],
        threshold_parties=config["threshold_parties"],
        aggregator_name=config["aggregator_name"],
        aggregator_args=config["aggregator_args"],
        use_fhe=True,
        num_rounds=config["num_rounds"],
        client_sampling_proportion=config["client_sampling_proportion"],
        enable_benchmarking=True
    )
    
    logger.info("Starting FHE federated training...")
    final_model = orchestrator.run()
    
    # Export comprehensive benchmarks
    benchmark_filename = f"mnist_{config['fhe_scheme']}_benchmark.csv"
    orchestrator.evaluate_and_export(benchmark_filename)
    
    logger.info(f"Experiment completed! Benchmarks saved to {benchmark_filename}")
    
    # Final evaluation
    loss, accuracy = final_model.evaluate(x_test, y_test, verbose=0)
    logger.info(f"Final Results - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
    return final_model
if __name__ == "__main__":
    final_model = main()
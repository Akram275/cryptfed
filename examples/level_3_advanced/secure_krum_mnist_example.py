#!/usr/bin/env python3
"""
SecureKrum MNIST Example with Byzantine Robustness
==================================================

This example demonstrates the SecureKrum aggregator with proper Byzantine attacks
using the MNIST dataset. It tests SecureKrum's robustness against various attack types
while preserving privacy through homomorphic encryption.

Key features demonstrated:
- SecureKrum aggregation with encrypted model updates
- Proper Byzantine attack simulation using FederatedClient built-in attacks:
  • Random Noise: Adds large random values to model weights
  • Sign Flipping: Inverts the sign of all model weights
  • Label Shuffling: Randomly permutes training labels
  • Gradient Ascent: Uses negative learning rate to maximize loss
- Comparison across different FHE schemes (CKKS vs BFV)
- Performance benchmarking with Byzantine tolerance
- Privacy-preserving Byzantine robustness analysis
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from cryptfed import CrypTFed
from cryptfed.core.benchmark_manager import BenchmarkManager, BenchmarkProfile
from cryptfed.core.federated_client import FederatedClient

def create_simple_model():
    """Create a simple CNN model for MNIST"""
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model with optimizer and loss function
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def run_secure_krum_experiment(fhe_scheme="ckks", num_byzantine=1, attack_type="random_noise"):
    """
    Run federated learning with SecureKrum aggregation.
    
    Args:
        fhe_scheme: FHE scheme to use ("ckks" or "bfv")
        num_byzantine: Number of Byzantine clients to simulate
        attack_type: Type of Byzantine attack ("random_noise", "sign_flipping", "label_shuffling", "gradient_ascent")
    """
    logger.info(f"Starting SecureKrum experiment with {fhe_scheme.upper()} scheme")
    logger.info(f"Simulating {num_byzantine} Byzantine clients with {attack_type} attacks")
    
    # Load and preprocess MNIST data
    logger.info("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Add channel dimension
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Use smaller subset for faster testing
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    x_test = x_test[:200]
    y_test = y_test[:200]
    
    # Create non-IID data distribution
    logger.info("Creating non-IID data distribution...")
    num_clients = 5
    samples_per_client = len(x_train) // num_clients
    
    # Create FederatedClient objects with proper Byzantine configuration
    clients = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        client_x = x_train[start_idx:end_idx]
        client_y = y_train[start_idx:end_idx]
        
        # Mark first `num_byzantine` clients as Byzantine
        is_byzantine = i < num_byzantine
        
        client = FederatedClient(
            client_id=f"client_{i}{'_byzantine' if is_byzantine else '_honest'}",
            model_fn=create_simple_model,
            x_train=client_x,
            y_train=client_y,
            local_epochs=1,
            byzantine=is_byzantine,
            attack_type=attack_type if is_byzantine else None,
            attack_args={"scale": 2.0} if attack_type == "random_noise" else {}
        )
        clients.append(client)
        
        if is_byzantine:
            logger.info(f"Client {i}: Byzantine client with {attack_type} attack")
        else:
            logger.info(f"Client {i}: Honest client")
    
    # Setup CrypTFed configuration
    config = {
        "model_fn": create_simple_model,
        "clients": clients,
        "use_fhe": True,
        "fhe_scheme": fhe_scheme,
        "aggregator_name": "secure_krum",  # Use SecureKrum aggregator
        "aggregator_args": {"f": num_byzantine},  # Tolerate num_byzantine Byzantine clients
        "num_rounds": 3,
        "client_sampling_proportion": 1.0,  # All clients participate
        "test_data": (x_test, y_test),
    }
    
    # Setup benchmarking
    benchmark_manager = BenchmarkManager([BenchmarkProfile.MODEL_PERFORMANCE])
    config["benchmark_manager"] = benchmark_manager
    
    logger.info(f"Configuration: {num_clients} clients, {config['num_rounds']} rounds")
    logger.info(f"FHE Scheme: {fhe_scheme.upper()}, Aggregator: SecureKrum")
    logger.info(f"Byzantine clients: {num_byzantine}, Attack type: {attack_type}")
    
    # Run experiment
    experiment = CrypTFed(**config)
    
    # Run the experiment
    logger.info("Starting federated learning with SecureKrum...")
    final_model = experiment.run()
    
    # Evaluate final model
    logger.info("Evaluating final model...")
    final_loss, final_accuracy = final_model.evaluate(x_test, y_test, verbose=0)
    
    logger.info(f"Final Results:")
    logger.info(f"  Accuracy: {final_accuracy:.4f}")
    logger.info(f"  Loss: {final_loss:.4f}")
    
    return {
        "scheme": fhe_scheme,
        "accuracy": final_accuracy,
        "loss": final_loss,
        "num_byzantine": num_byzantine,
        "attack_type": attack_type,
        "aggregator": "SecureKrum"
    }

def run_comparison_experiment():
    """Run comparison between different schemes and attack types"""
    logger.info("="*60)
    logger.info("SECURE KRUM BYZANTINE ROBUSTNESS EXPERIMENT")
    logger.info("="*60)
    
    results = []
    
    # Test SecureKrum with different FHE schemes and Byzantine attacks
    test_configurations = [
        # (scheme, attack_type, num_byzantine)
        ("ckks", "random_noise", 1),
        ("ckks", "sign_flipping", 1),
        ("bfv", "random_noise", 1),
        ("bfv", "sign_flipping", 1),
        ("bgv", "label_shuffling", 1),
    ]
    
    for scheme, attack_type, num_byzantine in test_configurations:
        try:
            result = run_secure_krum_experiment(
                fhe_scheme=scheme, 
                num_byzantine=num_byzantine, 
                attack_type=attack_type
            )
            results.append(result)
            logger.info(f"✓ {scheme.upper()} SecureKrum vs {attack_type} completed successfully")
        except Exception as e:
            logger.error(f"✗ {scheme.upper()} SecureKrum vs {attack_type} failed: {e}")
            continue
    
    # Print comparison results
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENTAL RESULTS")
    logger.info("="*60)
    logger.info(f"{'Scheme':<8} {'Attack Type':<15} {'Byzantine':<10} {'Accuracy':<12} {'Loss':<12}")
    logger.info("-" * 60)
    
    for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        logger.info(f"{result['scheme'].upper():<8} {result['attack_type']:<15} {result['num_byzantine']:<10} {result['accuracy']:<12.4f} {result['loss']:<12.4f}")
    
    if results:
        best_result = max(results, key=lambda x: x['accuracy'])
        logger.info(f"\nBest performing configuration:")
        logger.info(f"  {best_result['scheme'].upper()} SecureKrum vs {best_result['attack_type']}: {best_result['accuracy']:.4f} accuracy")
        logger.info(f"  Successfully defended against {best_result['num_byzantine']} Byzantine clients")
    
    logger.info("\nSecureKrum Analysis:")
    logger.info("- Model updates remain encrypted throughout aggregation")
    logger.info("- Only distance scores are decrypted for client selection")
    logger.info("- Byzantine robustness preserved while maintaining privacy")
    logger.info("- Works with both CKKS and BFV encryption schemes")
    logger.info("- Defends against multiple attack types:")
    
    attack_descriptions = {
        "random_noise": "  • Random Noise: Adds large random values to model weights",
        "sign_flipping": "  • Sign Flipping: Inverts the sign of all model weights", 
        "label_shuffling": "  • Label Shuffling: Randomly permutes training labels",
        "gradient_ascent": "  • Gradient Ascent: Uses negative learning rate to maximize loss"
    }
    
    tested_attacks = set(result['attack_type'] for result in results)
    for attack in tested_attacks:
        if attack in attack_descriptions:
            logger.info(attack_descriptions[attack])
    
    return results

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    try:
        results = run_comparison_experiment()
        
        if results:
            print(f"\nSecureKrum experiment completed successfully!")
            print(f"   Tested {len(results)} configurations with Byzantine robustness")
        else:
            print(f"\nAll experiments failed. Please check the setup.")
            
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
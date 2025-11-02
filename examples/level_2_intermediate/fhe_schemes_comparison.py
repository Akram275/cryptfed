"""
Level 2 Example: CIFAR-10 with Different FHE Schemes Comparison
===============================================================

Intermediate example demonstrating:
- Comparison of different FHE schemes (BFV vs BGV vs CKKS)
- Both single-key and threshold variants
- Simple dense model architecture (to test if architecture matters for weight aggregation)
- Detailed benchmark analysis and comparison
- Uses proven Level 3 configuration (except model architecture)

Testing hypothesis: Does model architecture affect encrypted weight aggregation?
- Simple flatten + dense model (vs Mini-ResNet)
- Sequential data distribution (proven to work)
- 50% client sampling per round (critical for threshold schemes)
- 7 threshold parties for threshold schemes
- Gradient clipping for numerical stability
- Sparse categorical crossentropy (no one-hot encoding)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import logging
import sys
import os

# Suppress CUDA warnings and TensorFlow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF messages except errors
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA device detection

# Import CrypTFed (now installed as package)
from cryptfed import CrypTFed, configure_logging
from cryptfed.core.federated_client import FederatedClient

# Configure logging
configure_logging(logging.INFO)
logger = logging.getLogger(__name__)

def create_simple_dense_model():
    """
    Create a simple flatten + dense model (the one that was failing before).
    This will help us understand what the real difference is.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Use same optimizer settings as the working version
    optimizer = tf.keras.optimizers.Adam(clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def run_fhe_experiment(fhe_scheme, aggregator_name="auto"):
    """Run a single FHE experiment with given scheme."""
    logger.info(f"Starting experiment with {fhe_scheme} scheme")
    
    # Load CIFAR-10 (use same preprocessing as Level 3)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Keep y_train and y_test as sparse labels (don't flatten)
    # Shuffle data like in Level 3
    p = np.random.permutation(len(x_train))
    x_train, y_train = x_train[p], y_train[p]
    
    # Use proven Level 3 configuration
    num_clients = 10
    samples_per_client = 5000
    
    # Use sequential data distribution (not random) - proven to work in Level 3
    clients = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = start_idx + samples_per_client
        
        client = FederatedClient(
            client_id=f"cifar_client_{i}",
            model_fn=create_simple_dense_model,
            x_train=x_train[start_idx:end_idx],
            y_train=y_train[start_idx:end_idx],
            local_epochs=1,  # Single epoch like Level 3
            local_lr=0.001,  # Explicit learning rate like Level 3
            chunking_strategy='flatten'
        )
        clients.append(client)
    
    # Configure based on FHE scheme - use proven Level 3 settings
    crypto_setting = "threshold" if "threshold" in fhe_scheme else "single_key"
    threshold_parties = 7 if crypto_setting == "threshold" else 1  # Use 7 parties like Level 3
    
    # Run experiment with Level 3 proven configuration
    orchestrator = CrypTFed(
        model_fn=create_simple_dense_model,
        clients=clients,
        test_data=(x_test, y_test),
        crypto_setting=crypto_setting,
        fhe_scheme=fhe_scheme,
        threshold_parties=threshold_parties,
        aggregator_name=aggregator_name,
        use_fhe=True,
        num_rounds=10,  # Reasonable number of rounds
        client_sampling_proportion=0.5,  # CRITICAL: 50% sampling like Level 3
        enable_benchmarking=True
    )
    
    final_model = orchestrator.run()
    
    # Export results
    filename = f"cifar_comparison_{fhe_scheme.replace('_', '-')}.csv"
    orchestrator.evaluate_and_export(filename)
    
    # Get final accuracy
    loss, accuracy = final_model.evaluate(x_test, y_test, verbose=0)
    logger.info(f"{fhe_scheme} - Final accuracy: {accuracy:.4f}")
    
    return {"scheme": fhe_scheme, "accuracy": accuracy, "loss": loss, "filename": filename}

def main():
    logger.info("Starting Level 2: FHE Schemes Comparison")
    logger.info("Using proven Level 3 configuration for better results")
    
    # FHE schemes to compare (focus on schemes that work well)
    schemes_to_test = [
        "ckks",
        "bfv", 
        "bgv",
        "threshold_bfv",
        "threshold_bgv", 
        "threshold_ckks"
    ]
    
    results = []
    
    logger.info(f"Testing FHE schemes: {schemes_to_test}")
    logger.info("Configuration: Mini-ResNet, 10 clients, 50% sampling, 7 threshold parties")
    
    for scheme in schemes_to_test:
        try:
            logger.info(f"="*60)
            logger.info(f"Starting {scheme} experiment...")
            result = run_fhe_experiment(scheme)
            results.append(result)
            logger.info(f"Completed {scheme}: Accuracy={result['accuracy']:.4f}")
        except Exception as e:
            logger.error(f"Error with {scheme}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            continue
    
    # Print comparison results
    logger.info("\n" + "="*60)
    logger.info("FHE SCHEMES COMPARISON RESULTS")
    logger.info("="*60)
    logger.info(f"{'Scheme':<20} {'Accuracy':<12} {'Loss':<12}")
    logger.info("-" * 60)
    
    for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        logger.info(f"{result['scheme']:<20} {result['accuracy']:<12.4f} {result['loss']:<12.4f}")
    
    if results:
        best_scheme = max(results, key=lambda x: x['accuracy'])
        logger.info(f"\nBest performing scheme: {best_scheme['scheme']} ({best_scheme['accuracy']:.4f})")
        
        # Performance analysis
        threshold_results = [r for r in results if 'threshold' in r['scheme']]
        single_key_results = [r for r in results if 'threshold' not in r['scheme']]
        
        if threshold_results:
            avg_threshold = np.mean([r['accuracy'] for r in threshold_results])
            logger.info(f"Average threshold schemes accuracy: {avg_threshold:.4f}")
        
        if single_key_results:
            avg_single = np.mean([r['accuracy'] for r in single_key_results])
            logger.info(f"Average single-key schemes accuracy: {avg_single:.4f}")
    else:
        logger.error("No experiments completed successfully!")
    
    logger.info("\nFHE comparison experiment completed!")
    logger.info("Note: Results should now be much better using Level 3 proven configuration")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    main()
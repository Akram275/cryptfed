"""
Example 5: Quality-Weighted Aggregation with Modular Payloads

This example demonstrates custom aggregation using CrypTFed orchestrator:
- Modular payloads (encrypted model + plaintext accuracy)
- Custom quality-weighted aggregator (higher accuracy = higher weight)
- FHE computation graphs for validation
- Full integration with CrypTFed orchestrator

This showcases how to extend CrypTFed with custom aggregation logic
while maintaining the simple, clean API of the orchestrator.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from cryptfed import CrypTFed
from cryptfed.core.federated_client import FederatedClient
from cryptfed.core.payload import PayloadBuilder
from cryptfed.core.fhe_graph import FHEGraphBuilder
from cryptfed.aggregators.modular_aggregator import CustomAggregatorBuilder
from cryptfed.aggregators import aggregator_registry  # To register custom aggregator

tf.get_logger().setLevel('ERROR')


def create_simple_cnn():
    """Simple CNN for CIFAR-10"""
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


def quality_weighted_aggregation(payloads, cc=None, **kwargs):
    """
    Custom aggregation: Weight by client accuracy.
    Clients with higher accuracy get more influence.
    
    This function is compatible with CrypTFed's modular aggregator system.
    """
    print("\n  Custom Quality-Weighted Aggregation:")
    
    # Extract accuracy weights from payloads
    accuracies = []
    for payload in payloads:
        acc_item = payload.get_item("local_accuracy")
        accuracy = acc_item.data if acc_item else 0.5
        accuracies.append(accuracy)
        print(f"    {payload.client_id}: accuracy={accuracy:.4f}")
    
    # Normalize to sum to 1
    total_acc = sum(accuracies)
    weights = [acc / total_acc for acc in accuracies]
    
    print(f"  Normalized weights: {[f'{w:.3f}' for w in weights]}")
    
    # Get encrypted chunks from first client
    first_chunks = payloads[0].get_item("model_update").data
    num_chunks = len(first_chunks)
    
    print(f"  Aggregating {num_chunks} encrypted chunks...")
    
    # Aggregate each chunk with quality weights
    aggregated_chunks = []
    for chunk_idx in range(num_chunks):
        weighted_chunk = None
        
        for payload_idx, payload in enumerate(payloads):
            chunks = payload.get_item("model_update").data
            chunk = chunks[chunk_idx]
            weight = weights[payload_idx]
            
            if weighted_chunk is None:
                weighted_chunk = chunk * weight
            else:
                weighted_chunk = weighted_chunk + (chunk * weight)
        
        aggregated_chunks.append(weighted_chunk)
    
    return aggregated_chunks


def main():
    """Demonstrate custom quality-weighted aggregation with CrypTFed orchestrator"""
    print("\n" + "="*70)
    print("CUSTOM Quality-Weighted Aggregation Demo".center(70))
    print("Using CrypTFed Orchestrator + Modular Payloads".center(70))
    print("="*70 + "\n")
    
    # Configuration
    NUM_CLIENTS = 2
    NUM_ROUNDS = 3
    LOCAL_EPOCHS = 1
    SAMPLES_PER_CLIENT = 800
    
    print(f"Configuration:")
    print(f"  Clients: {NUM_CLIENTS}")
    print(f"  Rounds: {NUM_ROUNDS}")
    print(f"  Samples per client: {SAMPLES_PER_CLIENT}")
    print(f"  Custom aggregation: Quality-weighted (accuracy-based)")
    
    # Load CIFAR-10
    print("\nLoading CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()
    
    x_test = x_test[:500]
    y_test = y_test[:500]
    
    # Create clients with different data quality (simulated by data distribution)
    print(f"\nPreparing {NUM_CLIENTS} clients with varying data quality...")
    client_data = []
    for i in range(NUM_CLIENTS):
        # Client 0 gets better data (lower class indices, easier to learn)
        # Client 1 gets harder data (higher class indices)
        if i == 0:
            # Filter for classes 0-4 (easier)
            mask = y_train < 5
            x_filtered = x_train[mask][:SAMPLES_PER_CLIENT]
            y_filtered = y_train[mask][:SAMPLES_PER_CLIENT]
        else:
            # Filter for classes 5-9 (harder)
            mask = y_train >= 5
            x_filtered = x_train[mask][:SAMPLES_PER_CLIENT]
            y_filtered = y_train[mask][:SAMPLES_PER_CLIENT]
        
        # Split into train/val
        split = int(0.8 * len(x_filtered))
        
        client_data.append((
            x_filtered[:split], y_filtered[:split],
            x_filtered[split:], y_filtered[split:]
        ))
        print(f"  client_{i}: {len(x_filtered[:split])} train, {len(x_filtered[split:])} val")
    
    # Create custom aggregator with FHE graph validation
    print("\nCreating custom quality-weighted aggregator...")
    aggregator_class = CustomAggregatorBuilder.from_function(
        quality_weighted_aggregation,
        requires_plaintext=False,
        name="quality_weighted"
    )
    print(f"  Aggregator: {aggregator_class.__class__.__name__}")
    print(f"  Requires plaintext: {aggregator_class.requires_plaintext_updates}")
    
    # Register custom aggregator in registry for CrypTFed to use
    aggregator_registry['quality_weighted'] = aggregator_class.__class__
    
    # Validate aggregation graph
    print("\nValidating FHE aggregation graph...")
    graph = FHEGraphBuilder("quality_weighted_agg")
    graph.weighted_sum(["update_0", "update_1"], [0.6, 0.4], "aggregated")
    built_graph = graph.build()
    valid, errors = built_graph.validate()
    print(f"  Graph valid: {valid}")
    if valid:
        print(f"  Multiplicative depth: {built_graph.estimate_depth()}")
    
    # Create FederatedClient instances with payload mode enabled
    print("\nCreating FederatedClient instances with payload mode...")
    clients = []
    for i in range(NUM_CLIENTS):
        x_tr, y_tr, x_val, y_val = client_data[i]
        client = FederatedClient(
            client_id=f"client_{i}",
            x_train=x_tr,
            y_train=y_tr,
            model_fn=create_simple_cnn,
            local_epochs=LOCAL_EPOCHS,
            use_payload_mode=True,  # Enable payload mode
            validation_data=(x_val, y_val)  # Needed for local accuracy
        )
        clients.append(client)
    
    # Initialize CrypTFed orchestrator with custom aggregator
    print("\nInitializing CrypTFed orchestrator...")
    print("  Mode: Payload-based with custom aggregation")
    cryptfed = CrypTFed(
        model_fn=create_simple_cnn,
        clients=clients,
        test_data=(x_test, y_test),
        use_fhe=True,
        fhe_scheme='ckks',
        aggregator_name='quality_weighted',  # Use registered custom aggregator
        num_rounds=NUM_ROUNDS
    )
    
    # Federated learning rounds
    print("\n" + "="*70)
    print("Starting Custom Quality-Weighted FL with CrypTFed")
    print("="*70)
    
    # Run training
    fhe_model = cryptfed.run()
    
    # Final results
    print("\n" + "="*70)
    print("Final Results")
    print("="*70)
    loss, acc = fhe_model.evaluate(x_test, y_test, verbose=0)
    print(f"Final test accuracy: {acc:.4f}")
    print(f"Final test loss: {loss:.4f}")
    
    print("\nWhat Was Demonstrated:")
    print("  - CrypTFed orchestrator with custom quality-weighted aggregation")
    print("  - Modular payloads: encrypted model + plaintext accuracy")
    print("  - Clients with higher accuracy got higher aggregation weight")
    print("  - Full FHE encryption of model weights during aggregation")
    print("  - Backward compatibility: level_1_basic examples still work")
    
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()

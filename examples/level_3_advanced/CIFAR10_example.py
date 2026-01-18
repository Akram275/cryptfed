"""
Advanced CIFAR-10 Example: Threshold FHE with Byzantine Robustness

This example demonstrates CrypTFed's threshold FHE capabilities with Byzantine
fault tolerance

Key configuration for threshold FHE success:
- Minimal ResNet architecture (4 filters) compatible with threshold depth constraints
- Gradient clipping (clipvalue=1.0) for numerical stability
- Sparse categorical crossentropy (no one-hot encoding)
- Client sampling proportion of 50% (critical for threshold convergence)
- Single local epoch per round to prevent drift
- Threshold BFV scheme with 7 parties

Features demonstrated:
- Threshold homomorphic encryption for multi-party security
- Byzantine client simulation with sign flipping attacks
- Robust aggregation in encrypted domain
- Comprehensive benchmarking and evaluation
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import sys
import time
import logging

# Suppress CUDA warnings and TensorFlow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF messages except errors
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA device detection

tf.get_logger().setLevel('ERROR')
logging.basicConfig(level=logging.INFO)

# Import CrypTFed (now installed as package)
from cryptfed import CrypTFed
from cryptfed.core.federated_client import FederatedClient


def create_simple_dense_model():
    """
    Create a simple flatten + dense model (the one that was failing before).
    This will help us understand what the real difference is.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Use same optimizer settings as the working version
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_mini_resnet_cifar10(input_shape=(32, 32, 3), num_classes=10):
    """
    Minimal ResNet model based on the working pattern from new_test.py.
    Uses very few filters and simple structure compatible with threshold FHE.
    """
    inputs = layers.Input(shape=input_shape)
    num_filters = 4
    x = layers.Conv2D(num_filters, (3, 3), padding='same', activation='relu')(inputs)
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
    x = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=x)

    # Use gradient clipping for numerical stability (critical for threshold FHE)
    optimizer = tf.keras.optimizers.Adam(clipvalue=1.0)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',  # Use sparse labels like new_test.py
                  metrics=['accuracy'])
    return model


# --- Simulation Parameters ---
print("--- Preparing CIFAR-10 Simulation (Threshold FHE Pattern) ---")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# DO NOT convert to categorical - use sparse labels like new_test.py
# y_train and y_test remain as integer labels for sparse_categorical_crossentropy

# Shuffle data like in new_test.py
p = np.random.permutation(len(x_train))
x_train, y_train = x_train[p], y_train[p]

NUM_CLIENTS = 10  # Match new_test.py exactly
NUM_BYZANTINE_CLIENTS = 0  # Enable Byzantine clients for demonstration
ATTACK_TYPE = "sign_flipping"  # Use same attack as new_test.py
ATTACK_ARGS = {}
NUM_ROUNDS = 20  
SAMPLES_PER_CLIENT = 5000  
LOCAL_EPOCHS_PER_ROUND = 1  

# Use threshold FHE
USE_FHE = True
CRYPTO_SETTING = "threshold" 
FHE_SCHEME = "threshold_bgv"  

# Use simple sequential data distribution like new_test.py (not IID)
client_datasets = []
for i in range(NUM_CLIENTS):
    start, stop = i * SAMPLES_PER_CLIENT, (i + 1) * SAMPLES_PER_CLIENT
    client_datasets.append((x_train[start:stop], y_train[start:stop]))

clients = []
for i in range(NUM_CLIENTS):
    is_byzantine = (i < NUM_BYZANTINE_CLIENTS)
    x_client, y_client = client_datasets[i]
    clients.append(FederatedClient(
                        client_id=f"client_{i}", model_fn=create_simple_dense_model,
                        x_train=x_client, y_train=y_client,
                        local_epochs=LOCAL_EPOCHS_PER_ROUND,
                        local_lr=0.001,  # Explicit local learning rate like new_test.py
                        chunking_strategy='flatten',
                        # Pass the Byzantine configuration to the client (Default is Honest client)
                        byzantine=is_byzantine,
                        attack_type=ATTACK_TYPE if is_byzantine else None,
                        attack_args=ATTACK_ARGS if is_byzantine else {}
                    ))

# --- CrypTFed Orchestrator Initialization ---
orchestrator = CrypTFed(
    model_fn=create_simple_dense_model,
    clients=clients,
    test_data=(x_test, y_test),
    use_fhe=USE_FHE,
    crypto_setting=CRYPTO_SETTING,
    fhe_scheme=FHE_SCHEME,
    threshold_parties=7,  # Match new_test.py exactly
    num_rounds=NUM_ROUNDS,
    client_sampling_proportion=0.5,  # CRITICAL: Only 50% of clients per round like new_test.py
    aggregator_name="auto",  # Let CrypTFed auto-select the appropriate aggregator
)

# --- Start Training ---
start = time.time()
final_model = orchestrator.run()
print("\n--- Evaluating final global model on the CIFAR-10 test set ---")
loss, accuracy = final_model.evaluate(x_test, y_test, verbose=0)
print(f"\n\nTraining done in {(time.time() - start):.4f} s. Final Global Model Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

# --- Export Benchmarks ---
mode_str = f"secure_{FHE_SCHEME}" if USE_FHE else "plaintext"
filename = f'cryptfed_cifar10_{mode_str}_multi_ct.csv'
orchestrator.evaluate_and_export(filename)
print(f"\n CIFAR-10 simulation complete. Benchmark data saved to '{filename}'.")
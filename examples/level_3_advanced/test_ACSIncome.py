import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import sys
import logging

# Suppress CUDA warnings and TensorFlow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF messages except errors
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA device detection

# Import CrypTFed (now installed as package)
import cryptfed

cryptfed.configure_logging(logging.INFO) 

try:
    from folktables import ACSDataSource, ACSEmployment
except ImportError:
    print("Please install the 'folktables' library by running: pip install folktables")
    exit()

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Import CrypTFed components
from cryptfed import CrypTFed
from cryptfed.core.federated_client import FederatedClient

# --- Step 2: Define the Model ---
# A simple multi-layer perceptron (MLP) is well-suited for tabular data.
def create_mlp_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Binary classification (predicting >$50K income)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# --- Step 3: Data Loading and Preparation ---
print("--- Preparing ACSIncome (Folktables) Simulation ---")

# Each client will be a different US state, creating a non-IID scenario.
states = ['CA', 'TX', 'FL', 'NY', 'PA']
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')

client_datasets = []
print("Downloading and preprocessing data for each state...")
for state in states:
    # Download data for the state
    acs_data = data_source.get_data(states=[state], download=True)

    # Define features and the target task (predicting income > $50K)
    features, label, _ = ACSEmployment.df_to_numpy(acs_data)

    # Normalize features for better training
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    client_datasets.append((features, label))
    print(f"  Loaded data for state: {state} ({len(features)} samples)")

# Use one state's data as a held-out test set
test_state = 'IL'
print(f"Downloading held-out test data for state: {test_state}")
acs_test_data = data_source.get_data(states=[test_state], download=True)

x_test, y_test, _ = ACSEmployment.df_to_numpy(acs_test_data)
# -----------------------

x_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)

# --- Step 4: Simulation Parameters ---
NUM_CLIENTS = len(states)
NUM_ROUNDS = 10
LOCAL_EPOCHS_PER_ROUND = 1

USE_FHE = True
FHE_SCHEME = "threshold_ckks"
CRYPTO_SETTING = "threshold" # "single_key" or "threshold"
AGGREGATOR = "auto" # "auto" or "trimmed_mean"


# --- Client and Model Initialization ---
# Get the input shape from the first client's data
input_shape = (client_datasets[0][0].shape[1],)
model_fn = lambda: create_mlp_model(input_shape)

clients = []
for i, state in enumerate(states):
    x_client, y_client = client_datasets[i]
    clients.append(FederatedClient(
        client_id=f"client_{state}",
        model_fn=model_fn,
        x_train=x_client,
        y_train=y_client,
        chunking_strategy='flatten',
        local_epochs=LOCAL_EPOCHS_PER_ROUND,
    ))

# --- CrypTFed Orchestrator Initialization ---
orchestrator = CrypTFed(
    model_fn=model_fn,
    clients=clients,
    test_data=(x_test, y_test),
    crypto_setting=CRYPTO_SETTING,
    aggregator_name=AGGREGATOR,
    use_fhe=USE_FHE,
    fhe_scheme=FHE_SCHEME,
    num_rounds=NUM_ROUNDS,
)

# --- Start Training ---
final_model = orchestrator.run()

# --- Final Evaluation & Export ---
mode_str = f"{CRYPTO_SETTING}_{FHE_SCHEME}" if USE_FHE else "plaintext"
filename = f'cryptfed_folktables_{mode_str}_{AGGREGATOR}.csv'
orchestrator.evaluate_and_export(filename)

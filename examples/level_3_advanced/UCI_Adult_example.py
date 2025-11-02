import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import sys
import random
import logging
from datetime import datetime
from pathlib import Path
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# Suppress CUDA warnings and TensorFlow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF messages except errors
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA device detection

# Import CrypTFed (now installed as package)
from cryptfed import CrypTFed
from cryptfed.core.federated_client import FederatedClient
from cryptfed.core.benchmark_manager import BenchmarkManager, BenchmarkProfile

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(f"benchmark_results_adult_{TIMESTAMP}")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Data Loading and Preprocessing ---
def load_and_preprocess_adult_data():
    """Fetches, cleans, and preprocesses the UCI Adult dataset."""
    print("--- Loading and Preprocessing UCI Adult Dataset ---")
    adult = fetch_ucirepo(id=2)
    X = adult.data.features
    y = adult.data.targets

    # Clean the target variable
    y['income'] = y['income'].str.strip().replace({'<=50K.': '<=50K', '>50K.': '>50K'})
    y = y['income'].apply(lambda x: 1 if x == '>50K' else 0).to_numpy()

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    # Create a preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Split data before fitting the preprocessor
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Fit the pipeline on the training data and transform both sets
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    x_train_processed = pipeline.fit_transform(x_train).toarray()
    x_test_processed = pipeline.transform(x_test).toarray()
    
    print(f"Dataset processed. Train shape: {x_train_processed.shape}, Test shape: {x_test_processed.shape}")
    return x_train_processed, y_train, x_test_processed, y_test

# --- Dirichlet Partitioning ---
def dirichlet_partition(data, labels, num_clients, alpha=0.5):
    """
    Partitions a dataset among clients using a Dirichlet distribution to simulate non-IID data.
    """
    print(f"--- Partitioning data for {num_clients} clients with Dirichlet (alpha={alpha}) ---")
    num_samples = len(data)
    num_classes = len(np.unique(labels))
    
    # Create a list of indices for each class
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    # Proportions of each class for each client, drawn from a Dirichlet distribution
    client_proportions = np.random.dirichlet([alpha] * num_classes, num_clients)
    
    client_data_indices = [[] for _ in range(num_clients)]
    
    for c in range(num_classes):
        class_c_indices = class_indices[c]
        np.random.shuffle(class_c_indices)
        
        # Calculate the number of samples of class c for each client
        proportions_c = client_proportions[:, c]
        samples_per_client_c = (proportions_c / proportions_c.sum() * len(class_c_indices)).astype(int)
        
        # Ensure all samples are distributed
        diff = len(class_c_indices) - samples_per_client_c.sum()
        for i in range(diff):
            samples_per_client_c[i % num_clients] += 1
            
        # Distribute indices
        current_pos = 0
        for client_id in range(num_clients):
            num_samples_for_client = samples_per_client_c[client_id]
            client_data_indices[client_id].extend(class_c_indices[current_pos : current_pos + num_samples_for_client])
            current_pos += num_samples_for_client
            
    client_data = [(data[indices], labels[indices]) for indices in client_data_indices]
    return client_data

# --- Model Definition (MLP) ---
def create_mlp(input_shape):
    """Defines a simple MLP for the tabular data."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# --- Main Experiment ---
if __name__ == "__main__":
    # --- Configuration ---
    TOTAL_CLIENTS = 10
    DIRICHLET_ALPHA = 100.0  # Lower alpha = more non-IID
    NUM_ROUNDS = 50
    
    # 1. Load and preprocess data
    x_train, y_train, x_test, y_test = load_and_preprocess_adult_data()
    
    # 2. Partition data among clients
    client_datasets = dirichlet_partition(x_train, y_train, TOTAL_CLIENTS, alpha=DIRICHLET_ALPHA)
    
    # 3. Create model function
    input_shape = (x_train.shape[1],)
    model_fn = lambda: create_mlp(input_shape)
    
    # 4. Create FederatedClient instances
    clients = []
    for i, (x_client, y_client) in enumerate(client_datasets):
        clients.append(FederatedClient(
            client_id=f"client_{i}",
            model_fn=model_fn,
            x_train=x_client,
            y_train=y_client,
            local_epochs=1,
            local_lr=0.001
        ))
        
    # 5. Configure and run the CrypTFed experiment
    experiment_config = {
        "model_fn": model_fn,
        "clients": clients,
        "client_sampling_proportion": 0.5, # 50% of clients participate per round
        "test_data": (x_test, y_test),
        "num_rounds": NUM_ROUNDS,
        "use_fhe": True,  # Start with FHE enabled
        "crypto_setting": "threshold",
        "fhe_scheme": "threshold_bfv",
        "threshold_parties": 3,
        "aggregator_name": "auto",
    }
    
    benchmark_manager = BenchmarkManager([BenchmarkProfile.MODEL_PERFORMANCE])
    experiment_config["benchmark_manager"] = benchmark_manager
    
    print("\n--- Running Federated Learning Experiment on UCI Adult ---")
    experiment = CrypTFed(**experiment_config)
    final_model = experiment.run()
    
    # 6. Evaluate and export results
    csv_path = OUTPUT_DIR / 'adult_experiment_benchmarks.csv'
    print(f'\nExporting and Visualizing Results to {csv_path}')
    experiment.evaluate_and_export(csv_path)
    
    benchmark_manager.plot_round_comparison(
        metric_name="Model Accuracy",
        plot_type="line",
        save_path=OUTPUT_DIR / "adult_accuracy_over_rounds.png"
    )
    
    print(f"\nAll benchmark results and visualizations have been saved to: {OUTPUT_DIR}")
    print("\nExperiment Summary:")
    print(f"- Dataset: UCI Adult")
    print(f"- Number of Clients: {TOTAL_CLIENTS}")
    print(f"- Data Distribution: Dirichlet (alpha={DIRICHLET_ALPHA})")
    print(f"- Number of Rounds: {NUM_ROUNDS}")

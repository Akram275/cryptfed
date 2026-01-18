"""
Example 8: Fairness-Aware FL with ACSIncome and SPD weighting

This example demonstrates:
- Tabular data processing with ACSIncome (Folktables)
- Statistical Parity Difference (SPD) calculation as a local fairness metric
- Custom aggregation weights as a function of local fairness (SPD closer to 0 = higher weight)
- Threshold FHE (CKKS) with 10 clients
- Integration with CrypTFed's modular payload system and custom metrics hook

The goal is to promote a more "fair" global model by giving more weight 
to clients whose local sub-populations show less gender-based prediction bias.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import logging
import random

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from cryptfed import CrypTFed, configure_logging
from cryptfed.core.federated_client import FederatedClient
from cryptfed.aggregators.modular_aggregator import CustomAggregatorBuilder
from cryptfed.aggregators import aggregator_registry

# Configure logging
configure_logging(logging.INFO)
logger = logging.getLogger(__name__)

try:
    from folktables import ACSDataSource, ACSIncome
except ImportError:
    print("Please install 'folktables' to run this example: pip install folktables")
    sys.exit(1)

# --- Configuration ---
NUM_CLIENTS = 10
NUM_ROUNDS = 20
LOCAL_EPOCHS = 3 # Increased local epochs
FHE_SCHEME = "threshold_ckks"
CRYPTO_SETTING = "threshold"
# The number of parties involved in the threshold ceremony
THRESHOLD_PARTIES = 5 

# --- Model Definition ---
def create_mlp_model(input_shape):
    """Simple MLP for tabular binary classification"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Fairness Metric: Statistical Parity Difference (SPD) ---
def calculate_spd(model, validation_data):
    """
    Calculates Statistical Parity Difference (SPD) between Male/Female groups.
    SPD = |P(Y_pred=1 | Male) - P(Y_pred=1 | Female)|
    """
    x_val, y_val, sex_attr = validation_data
    
    # Generate predictions
    y_probs = model.predict(x_val, verbose=0)
    y_pred = (y_probs > 0.5).astype(int).flatten()
    
    # Sex attribute in ACSIncome (SEX): 1 = Male, 2 = Female
    # Find indices
    male_indices = (sex_attr == 1)
    female_indices = (sex_attr == 2)
    
    # Calculate acceptance rates
    prob_male = np.mean(y_pred[male_indices]) if np.any(male_indices) else 0
    prob_female = np.mean(y_pred[female_indices]) if np.any(female_indices) else 0
    
    spd = abs(prob_male - prob_female)
    
    return {
        "local_spd": float(spd),
        "acceptance_male": float(prob_male),
        "acceptance_female": float(prob_female)
    }

# --- Fairness-Aware Aggregation Function ---
def fairness_spd_weighted_aggregation(payloads, cc=None, **kwargs):
    """
    Custom aggregation function that weights clients based on their SPD.
    Weight = exp(-5 * SPD)
    We want weights to be higher for SPD closer to 0.
    """
    print("\n  Fairness-Aware Aggregation (SPD-Based):")
    
    spds = []
    for payload in payloads:
        spd_item = payload.get_item("local_spd")
        spd = spd_item.data if spd_item else 1.0 # Penalize if missing
        spds.append(spd)
        print(f"    {payload.client_id}: SPD={spd:.4f}")
    
    # Map SPD to score (exponential decay)
    # SPD=0 -> score=1.0, SPD=0.2 -> score=0.36, SPD=0.5 -> score=0.08
    scores = [np.exp(-5 * s) for s in spds]
    
    # Normalize scores to sum to 1
    total_score = sum(scores)
    weights = [s / total_score for s in scores]
    
    print(f"  Derived Weights: {[f'{w:.3f}' for w in weights]}")
    
    # Handle FHE aggregation across chunks
    # We use simple weighted sum on encrypted model chunks
    first_chunks = payloads[0].get_item("model_update").data
    num_chunks = len(first_chunks)
    
    aggregated_chunks = []
    for chunk_idx in range(num_chunks):
        weighted_chunk = None
        for i, payload in enumerate(payloads):
            chunk = payload.get_item("model_update").data[chunk_idx]
            # Convert NumPy float to standard Python float for OpenFHE compatibility
            w = float(weights[i])
            
            if weighted_chunk is None:
                weighted_chunk = chunk * w
            else:
                weighted_chunk = weighted_chunk + (chunk * w)
        aggregated_chunks.append(weighted_chunk)
    
    return aggregated_chunks

# --- Main Execution ---
def main():
    print("\n" + "="*80)
    print("FAIRNESS-AWARE FEDERATED LEARNING DEMO".center(80))
    print("Dataset: ACSIncome | Metric: Statistical Parity Difference (SPD)".center(80))
    print("Security: Threshold CKKS (5-of-10)".center(80))
    print("="*80 + "\n")

    # Step 1: Load ACSIncome Data
    # Each client will be a different state or random split of multiple states
    # For this demo, we'll use a few large states and split them
    states = ['CA', 'TX', 'NY', 'FL', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI']
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    
    print(f"Initializaing {NUM_CLIENTS} clients with ACSIncome data...")
    client_configs = []
    
    # We'll use ACSIncome task: AGEP, COW, SCHL, MAR, OCCP, POBP, RELP, WKHP, SEX, RAC1P
    # Label is PINCP (Income > $50k)
    
    # Sex attribute index in the returned numpy array
    # ACSIncome features: AGEP(0), COW(1), SCHL(2), MAR(3), OCCP(4), POBP(5), RELP(6), WKHP(7), SEX(8), RAC1P(9)
    SEX_INDEX = 8 
    
    global_x_test = []
    global_y_test = []
    
    for i, state in enumerate(states):
        print(f"  Downloading/Loading state: {state}...")
        try:
            acs_data = data_source.get_data(states=[state], download=True)
            x, y, _ = ACSIncome.df_to_numpy(acs_data)
        except Exception as e:
            print(f"Error loading {state}, using CA again. Error: {e}")
            acs_data = data_source.get_data(states=['CA'], download=True)
            x, y, _ = ACSIncome.df_to_numpy(acs_data)
        
        # Shuffle everything to ensure representative slices
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x, y = x[indices], y[indices]
        
        # Convert y to float32 (0.0 / 1.0)
        y = y.astype('float32')

        # Keep SEX column for SPD calculation before normalization
        # Folktables SEX: 1=Male, 2=Female
        sex_col = x[:, SEX_INDEX]
        
        # Normalize features
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        x_std[x_std == 0] = 1.0 # Prevent div by zero
        x_norm = (x - x_mean) / x_std
        
        # Split into train/validation (80/20)
        split = int(0.8 * len(x_norm))
        # Limit to 1500 samples per client for better training
        max_samples = 1500
        train_slice = min(split, max_samples)
        
        x_train, y_train = x_norm[:train_slice], y[:train_slice]
        x_val, y_val, sex_val = x_norm[split:split+300], y[split:split+300], sex_col[split:split+300]
        
        client_configs.append({
            'client_id': f"client_{state}",
            'x_train': x_train,
            'y_train': y_train,
            'val_data': (x_val, y_val, sex_val)
        })
        
        # Add to global test set
        global_x_test.append(x_norm[split+300:split+500])
        global_y_test.append(y[split+300:split+500])

    x_test = np.concatenate(global_x_test)
    y_test = np.concatenate(global_y_test)
    input_shape = (x_test.shape[1],)
    
    # Step 2: Define Model and Aggregator
    model_fn = lambda: create_mlp_model(input_shape)
    
    # Register custom aggregator
    print("\nConfiguring custom SPD-weighted aggregator...")
    fair_agg_class = CustomAggregatorBuilder.from_function(
        fairness_spd_weighted_aggregation,
        requires_plaintext=False,
        name="spd_weighted"
    )
    aggregator_registry['spd_weighted'] = fair_agg_class.__class__
    
    # Step 3: Initialize Clients
    print("\nInitializing clients with payload mode and fairness metrics...")
    clients = []
    for cfg in client_configs:
        client = FederatedClient(
            client_id=cfg['client_id'],
            model_fn=model_fn,
            x_train=cfg['x_train'],
            y_train=cfg['y_train'],
            local_epochs=LOCAL_EPOCHS,
            use_payload_mode=True,
            validation_data=cfg['val_data'],
            custom_metrics_fn=calculate_spd # The hook we added earlier
        )
        clients.append(client)
    
    # Step 4: Initialize CrypTFed Orchestrator
    print("\nInitializing CrypTFed in Threshold mode...")
    cryptfed = CrypTFed(
        model_fn=model_fn,
        clients=clients,
        test_data=(x_test, y_test),
        use_fhe=True,
        fhe_scheme=FHE_SCHEME,
        crypto_setting=CRYPTO_SETTING,
        threshold_parties=THRESHOLD_PARTIES,
        aggregator_name='spd_weighted',
        num_rounds=NUM_ROUNDS
    )
    
    # Step 5: Execute Training
    print("\n" + "~"*80)
    print("Starting Federated Training (SPD-Aware Threshold CKKS)")
    print("~"*80 + "\n")
    
    final_model = cryptfed.run()
    
    # Final Evaluation
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    loss, acc = final_model.evaluate(x_test, y_test, verbose=0)
    print(f"Global Model Accuracy: {acc:.4f}")
    
    # Evaluate SPD on global test set
    # Need to get sex column for full test set if we wanted full fairness check
    # But for demo, the print logs during round show the per-client weighting
    
    print("\nWhat happened:")
    print("  1. Clients computed true local income predictions bias (SPD) locally.")
    print("  2. Client-side fairness metrics were attached to encrypted model weights in a modular payload.")
    print("  3. The server, without seeing any model weights, derived aggregation influence based on these fairness scores.")
    print("  4. More 'fair' clients had significantly more influence on the global model parameters.")
    print("  5. Threshold cryptography ensured no single party could decrypt individual model updates.")
    print("="*80)

if __name__ == "__main__":
    main()

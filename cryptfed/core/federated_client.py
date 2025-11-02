from typing import List, Callable, Any, Tuple, Dict
import numpy as np
import time
from openfhe import *
import openfhe_numpy as onp
import logging
import tensorflow as tf
import random
import os

class FederatedClient:
    """
    Represents a client in the FL system, with optional Byzantine behavior and
    full support for single-key and threshold FHE protocols.
    """
    def __init__(self, client_id: str, model_fn: Callable, x_train: np.ndarray, y_train: np.ndarray, local_epochs: int,
                 local_lr: float = 0.001,
                 verbose: int = 0,
                 chunking_strategy: str = 'flatten',
                 byzantine: bool = False, attack_type: str = None, attack_args: Dict = {},
                 deterministic_seed: int = None):
        self.client_id = client_id
        self.deterministic_seed = deterministic_seed

        # Ensure the dataset size is divisible by the batch size to prevent warnings.
        batch_size = 32
        num_samples = len(x_train)
        samples_to_keep = (num_samples // batch_size) * batch_size
        self.x_train = x_train[:samples_to_keep]
        self.y_train = y_train[:samples_to_keep]

        self.weight = len(self.x_train)
        self.model = model_fn()
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.verbose = verbose
        self.chunking_strategy = chunking_strategy
        self.encryption_service = None
        self.byzantine = byzantine
        self.attack_type = attack_type
        self.attack_args = attack_args

        # State for Threshold Cryptography
        self.secret_key_share = None

        if self.byzantine:
            logging.getLogger(__name__).warning(f"ATTENTION: Client {self.client_id} is a BYZANTINE client, attack: {self.attack_type}")

    def set_deterministic_training(self, seed=42):
        """Ensure deterministic training across all random sources."""
        # Set Python random seed
        random.seed(seed)
        
        # Set NumPy random seed
        np.random.seed(seed)
        
        # Set TensorFlow random seed
        tf.random.set_seed(seed)
        
        # Enable deterministic operations in TensorFlow (if not already set)
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            # For older TensorFlow versions or if already enabled
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
        
        # Set environment variables for determinism
        os.environ['PYTHONHASHSEED'] = str(seed)

    def connect_to_fhe_service(self, fhe_manager):
        """Connect to FHE service - legacy method for compatibility."""
        self.connect_to_server({}, fhe_manager)

    def connect_to_server(self, public_context: dict, fhe_manager: Any):
        self.encryption_service = fhe_manager
        secure_mode = "secure (FHE)" if fhe_manager else "non-secure (plaintext)"
        logging.getLogger(__name__).info(f"Client {self.client_id} connected in {secure_mode} mode.")

    # --- Methods for Threshold FHE Protocol ---
    def generate_secret_key_share(self, cc: Any) -> Any:
        """Generates a key pair share and returns the SECRET part for setup."""
        logging.getLogger(__name__).info(f"Client {self.client_id} generating secret key share...")
        keys = cc.KeyGen()
        self.secret_key_share = keys.secretKey
        return self.secret_key_share

    def get_decryption_share(self, cc: Any, ciphertext: Any) -> Any:
        """Performs partial decryption on a ciphertext using its secret key share."""
        if not self.secret_key_share:
            raise RuntimeError("Client does not have a secret key share for decryption.")

        raw_ciphertext = ciphertext.data if isinstance(ciphertext, onp.FHETensor) else ciphertext
        return cc.MultipartyDecryptMain([raw_ciphertext], self.secret_key_share)[0]

    # --- UPDATED METHOD FOR SINGLE-KEY DECRYPTION ---
    def decrypt_model_chunks(self, encrypted_chunks: List[Any]) -> List[np.ndarray]:
        """
        Decrypts the global model chunks using the single-key manager.
        Returns a list of padded, decrypted chunks.
        """
        if not self.encryption_service:
            raise PermissionError("Client cannot decrypt without an FHE service.")

        # The FHE manager's decrypt method returns a list of padded chunks.
        # The client simply passes this result up to the orchestrator.
        return self.encryption_service.decrypt(encrypted_chunks)
    # ----------------------------------------------------

    def _flatten_weights(self, weights: List[np.ndarray]) -> np.ndarray:
        return np.concatenate([w.flatten() for w in weights])

    # --- Byzantine Attack Methods ---
    def _sign_flipping_attack(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        logging.getLogger(__name__).warning(f"Client {self.client_id} is performing a SIGN-FLIPPING attack.")
        return [-w for w in weights]

    def _random_noise_attack(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        scale = self.attack_args.get("scale", 1.0)
        logging.getLogger(__name__).warning(f"Client {self.client_id} is performing a RANDOM NOISE attack with scale {scale}.")
        return [w + np.random.normal(loc=0, scale=scale, size=w.shape).astype(w.dtype) for w in weights]

    def _gradient_ascent_attack(self):
        logging.getLogger(__name__).warning(f"Client {self.client_id} is performing a GRADIENT ASCENT attack.")
        original_lr = self.model.optimizer.learning_rate.numpy()
        self.model.optimizer.learning_rate.assign(-abs(original_lr))
        return original_lr

    def _label_shuffling_attack(self, y_train):
        logging.getLogger(__name__).warning(f"Client {self.client_id} is performing a LABEL SHUFFLING attack.")
        y_shuffled = np.copy(y_train)
        np.random.shuffle(y_shuffled)
        return y_shuffled

    def execute_training_round(self, global_model_weights: List[np.ndarray]) -> Tuple[Any, float]:
        # Set deterministic training only if a seed is provided
        if self.deterministic_seed is not None:
            self.set_deterministic_training(self.deterministic_seed)
        
        #print(f"--- Client {self.client_id} starting training round ---")
        self.model.set_weights(global_model_weights)

        original_lr = self.model.optimizer.learning_rate.numpy()
        self.model.optimizer.learning_rate.assign(self.local_lr)

        x_train_to_use, y_train_to_use = self.x_train, self.y_train

        if self.byzantine and self.attack_type == "gradient_ascent":
            self._gradient_ascent_attack()
        if self.byzantine and self.attack_type == "label_shuffling":
            y_train_to_use = self._label_shuffling_attack(self.y_train)

        start_time = time.time()
        batch_size = 32
        steps_per_epoch = len(x_train_to_use) // batch_size
        self.model.fit(x_train_to_use, y_train_to_use,
                       epochs=self.local_epochs,
                       batch_size=batch_size,
                       steps_per_epoch=steps_per_epoch,
                       shuffle=False,
                       verbose=self.verbose)
        local_train_time = time.time() - start_time

        self.model.optimizer.learning_rate.assign(original_lr)

        # Prefer an explicit client-side benchmark manager if provided, otherwise fall back to the
        # encryption service's benchmark manager (used in FHE mode).
        bm = getattr(self, 'benchmark_manager', None)
        if not bm and self.encryption_service:
            bm = getattr(self.encryption_service, 'benchmark_manager', None)
        if bm:
            bm.log_event(self.client_id, 'Local Train Time', local_train_time)

        local_weights = self.model.get_weights()

        if self.byzantine and self.attack_type == "sign_flipping":
            local_weights = self._sign_flipping_attack(local_weights)
        if self.byzantine and self.attack_type == "random_noise":
            local_weights = self._random_noise_attack(local_weights)

        if self.encryption_service:
            slot_count = self.encryption_service.slot_count
            if slot_count == 0: raise ValueError("FHE Manager slot_count is not set.")

            # Determine effective slot count based on scheme type
            # CKKS can only use N/2 slots due to complex number packing
            # BFV/BGV can use full N slots for integer data
            if hasattr(self.encryption_service, '__class__') and ('CKKS' in self.encryption_service.__class__.__name__):
                effective_slot_count = slot_count // 2
            else:
                effective_slot_count = slot_count

            weight_chunks = []
            if self.chunking_strategy == 'flatten':
                flat_local_weights = self._flatten_weights(local_weights)
                
                # For CKKS (non-threshold), use the original np.array_split approach as it was working
                # For BFV/BGV and threshold schemes (including threshold CKKS), use fixed-size chunks aligned with slot boundaries  
                if hasattr(self.encryption_service, '__class__') and ('CKKS' in self.encryption_service.__class__.__name__) and ('Threshold' not in self.encryption_service.__class__.__name__):
                    # Regular CKKS only: Use original chunking (was working before) - use effective_slot_count for proper slot utilization
                    num_chunks = int(np.ceil(len(flat_local_weights) / effective_slot_count))
                    weight_chunks = np.array_split(flat_local_weights, num_chunks)
                else:
                    # BFV/BGV (single-key and threshold) and threshold CKKS: Use fixed-size chunks aligned with slot boundaries
                    weight_chunks = []
                    # For threshold CKKS, use effective_slot_count; for BFV/BGV use full slot_count
                    chunk_size = effective_slot_count if (hasattr(self.encryption_service, '__class__') and 'CKKS' in self.encryption_service.__class__.__name__) else slot_count
                    for i in range(0, len(flat_local_weights), chunk_size):
                        chunk = flat_local_weights[i:i+chunk_size]
                        weight_chunks.append(chunk)
            elif self.chunking_strategy == 'per_layer':
                for i, layer_weights in enumerate(local_weights):
                    flat_layer = layer_weights.flatten()
                    if len(flat_layer) > slot_count:
                        raise ValueError(f"Layer {i} is too large ({len(flat_layer)} params) for FHE slot count ({slot_count}).")
                    weight_chunks.append(flat_layer)

            encrypted_chunks = self.encryption_service.encrypt(weight_chunks, self.client_id)

            # Track FHE bandwidth
            if bm:  # Using the benchmark manager from earlier in the method
                # Calculate approximate size based on the scheme type
                total_size = 0
                import sys

                for chunk in encrypted_chunks:
                    try:
                        if isinstance(chunk, onp.FHETensor):  # CKKS type (FHETensor)
                            # For CKKS tensors, use numpy array size
                            total_size += chunk.data.size * 16  # Approximate size per complex number
                        else:  # OpenFHE type or other
                            if hasattr(chunk, 'GetElements'):
                                elements = chunk.GetElements()
                                # If elements is a list, sum its components
                                if isinstance(elements, list):
                                    total_size += sum(elements) * 16
                                else:
                                    total_size += elements * 16
                            else:
                                # Fallback to system size
                                total_size += sys.getsizeof(chunk)
                    except Exception:
                        # Fallback if any estimation fails
                        total_size += sys.getsizeof(chunk)

                # Log with the correct metric name from BenchmarkProfile.FHE_BANDWIDTH
                bm.log_event(self.client_id, 'Network Transfer Size', total_size, unit='bytes')

            return (encrypted_chunks, self.weight)
        else:
            flat_local_weights = self._flatten_weights(local_weights)
            return (flat_local_weights, self.weight)

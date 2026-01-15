from typing import Callable, List, Any, Dict
import numpy as np
import time
import logging

from ..aggregators import aggregator_registry
from ..fhe import fhe_manager_registry
from .benchmark_manager import BenchmarkManager
from .federated_client import FederatedClient

try:
    from .protocol_state import ServerStateMachine, ServerState, ProtocolCoordinator
    STATE_MACHINE_AVAILABLE = True
except ImportError:
    STATE_MACHINE_AVAILABLE = False
    ServerStateMachine = None
    ServerState = None

class FederatedServer:
    """
    A stateful cryptographic worker. It holds an encrypted model and performs
    aggregation, but it NEVER decrypts. Decryption is orchestrated externally.
    The high-level wrapper Synergia orchestartes decryptions, either in single_key or threshold modes.
    """
    def __init__(self, model_fn: Callable,
                 crypto_setting: str = "single_key",
                 threshold_parties: int = 3,
                 aggregator_name: str = "auto", aggregator_args: Dict = {},
                 use_fhe: bool = True, fhe_scheme: str = "ckks",
                 enable_benchmarking: bool = True,
                 benchmark_manager: 'BenchmarkManager' = None,
                 x_test: np.ndarray = None,
                 y_test: np.ndarray = None,
                 protocol_coordinator: 'ProtocolCoordinator' = None):

        self.use_fhe = use_fhe
        self.crypto_setting = crypto_setting if self.use_fhe else "single_key"
        self.fhe_scheme = fhe_scheme
        
        # Initialize state machine if coordinator provided
        if protocol_coordinator and STATE_MACHINE_AVAILABLE:
            self.state_machine = protocol_coordinator.register_server()
        else:
            self.state_machine = None

        if self.crypto_setting == "threshold" and "threshold" not in self.fhe_scheme:
            raise ValueError(f"Crypto setting is 'threshold' but FHE scheme '{self.fhe_scheme}' is not a threshold scheme.")
        if self.crypto_setting == "single_key" and "threshold" in self.fhe_scheme:
            raise ValueError(f"Crypto setting is 'single_key' but FHE scheme '{self.fhe_scheme}' is a threshold scheme.")

        self.threshold_parties = threshold_parties
        self.key_holders = []  # Will store ALL clients with secret shares
        self.all_clients = []  # Store reference to all clients
        # Allow an external BenchmarkManager to be supplied (useful for tests/plots)
        if benchmark_manager is not None:
            self.benchmark_manager = benchmark_manager
        else:
            self.benchmark_manager = BenchmarkManager() if enable_benchmarking else None

        # Store test data for metrics collection
        self.x_test = x_test
        self.y_test = y_test
        self._model_for_eval = model_fn() if x_test is not None else None
        
        self.fhe_manager = None
        if self.use_fhe:
            if self.fhe_scheme not in fhe_manager_registry:
                raise ValueError(f"Unknown FHE scheme: '{self.fhe_scheme}'.")
            fhe_manager_class = fhe_manager_registry.get(self.fhe_scheme)
            fhe_manager_init_args = {'benchmark_manager': self.benchmark_manager}
            if self.fhe_scheme in ["threshold_bfv", "threshold_bgv", "threshold_ckks"]:
                fhe_manager_init_args['threshold'] = self.threshold_parties
            self.fhe_manager = fhe_manager_class(**fhe_manager_init_args)

        if aggregator_name == "auto":
            if not self.use_fhe: aggregator_name = "plaintext_fedavg"
            elif "ckks" in self.fhe_scheme: aggregator_name = "ckks_fedavg"
            else: aggregator_name = "integer_fedavg"

        aggregator_class = aggregator_registry.get(aggregator_name)
        
        # If FHE aggregator is not available, fall back to plaintext
        if aggregator_class is None:
            print(f"Warning: Aggregator '{aggregator_name}' not available. Falling back to plaintext_fedavg.")
            self.use_fhe = False
            aggregator_name = "plaintext_fedavg"
            aggregator_class = aggregator_registry.get(aggregator_name)
            
        self.aggregator = aggregator_class(benchmark_manager=self.benchmark_manager, **aggregator_args)
        if self.use_fhe :
            logging.getLogger(__name__).info(f"Server initialized in '{self.crypto_setting}' mode. Scheme: {self.fhe_scheme}. Aggregator: {aggregator_class.__name__}.")
        else :
            logging.getLogger(__name__).info(f"Server initialized in Plaintext trusted mode. Aggregator: FedAvg.")
        self._model_template = model_fn()
        self.encrypted_global_model = None # The server's main state

    def init_session(self, clients: List[FederatedClient] = []):
        # State: UNINITIALIZED -> INITIALIZING_CRYPTO or IDLE
        if self.state_machine:
            if self.use_fhe:
                self.state_machine.transition_to(ServerState.INITIALIZING_CRYPTO)
            else:
                self.state_machine.transition_to(ServerState.IDLE)
                return
        
        if not self.use_fhe: return
        
        if self.crypto_setting == "single_key":
            self.fhe_manager.generate_crypto_context_and_keys()
        elif self.crypto_setting == "threshold":
            if len(clients) < self.threshold_parties:
                raise ValueError(f"Not enough clients ({len(clients)}) for threshold ({self.threshold_parties})")

            cc = self.fhe_manager.generate_crypto_context()
            self.key_holders = clients[:self.threshold_parties]
            logging.getLogger(__name__).info(f"Selected {len(self.key_holders)} clients as key holders.")

            secret_key_shares = [client.generate_secret_key_share(cc) for client in self.key_holders]
            self.fhe_manager.collaborative_keygen(secret_key_shares)
        
        # State: INITIALIZING_CRYPTO -> CRYPTO_READY
        if self.state_machine:
            self.state_machine.transition_to(ServerState.CRYPTO_READY)
    
    def initialize_encrypted_global_model(self):
        initial_weights_list = self._model_template.get_weights()
        if not self.use_fhe:
            self.encrypted_global_model = initial_weights_list
            # In plaintext mode, no state transitions needed - model init is implicit
            return

        logging.getLogger(__name__).info("Server is encrypting the initial global model...")
        initial_weights_flat = np.concatenate([w.flatten() for w in initial_weights_list])
        slot_count = self.fhe_manager.slot_count
        
        # Use the same chunking strategy as clients for consistency
        # For threshold CKKS, use effective slot count; for others use full slot count
        is_threshold_ckks = (hasattr(self.fhe_manager, '__class__') and 
                            'Threshold' in self.fhe_manager.__class__.__name__ and 
                            'CKKS' in self.fhe_manager.__class__.__name__)
        chunk_size = slot_count // 2 if is_threshold_ckks else slot_count
        
        # Create fixed-size chunks like the clients do
        weight_chunks = []
        for i in range(0, len(initial_weights_flat), chunk_size):
            chunk = initial_weights_flat[i:i+chunk_size]
            weight_chunks.append(chunk)
            
        self.encrypted_global_model = self.fhe_manager.encrypt(weight_chunks, "server_init")
        
        # State: CRYPTO_READY -> MODEL_INITIALIZED -> IDLE
        if self.state_machine:
            self.state_machine.transition_to(ServerState.MODEL_INITIALIZED)
            self.state_machine.transition_to(ServerState.IDLE)
        
        # Create fixed-size chunks like the clients do
        weight_chunks = []
        for i in range(0, len(initial_weights_flat), chunk_size):
            chunk = initial_weights_flat[i:i+chunk_size]
            weight_chunks.append(chunk)
            
        self.encrypted_global_model = self.fhe_manager.encrypt(weight_chunks, "server_init")

    def encrypt_and_set_model(self, plaintext_vector: np.ndarray):
        """Encrypts a plaintext vector and sets it as the global model."""
        if not self.use_fhe:
            # In plaintext mode, just reshape and set
            temp_model = self._model_template
            current_pos = 0
            new_weights = []
            for layer in temp_model.get_weights():
                shape, size = layer.shape, layer.size
                new_weights.append(plaintext_vector[current_pos : current_pos + size].reshape(shape))
                current_pos += size
            self.encrypted_global_model = new_weights
            return

        logging.getLogger(__name__).info("Server is re-encrypting the new global model...")
        slot_count = self.fhe_manager.slot_count
        num_chunks = int(np.ceil(len(plaintext_vector) / slot_count))
        weight_chunks = np.array_split(plaintext_vector, num_chunks)
        self.encrypted_global_model = self.fhe_manager.encrypt(weight_chunks, "server_re_encrypt")

    def get_encrypted_global_model(self, for_broadcast: bool = False) -> Any:
        """
        A simple getter for the server's encrypted state.
        
        Args:
            for_broadcast: If True, transition to BROADCASTING_MODEL state
        """
        # State: IDLE -> BROADCASTING_MODEL (only when actually broadcasting to clients)
        if for_broadcast and self.state_machine and self.state_machine.current_state == ServerState.IDLE:
            self.state_machine.transition_to(ServerState.BROADCASTING_MODEL)
        return self.encrypted_global_model

    def aggregate_and_update(self, client_payloads: List[tuple]):
        """Securely aggregates client models and updates the internal encrypted global model state."""
        if not client_payloads: return
        
        # State: BROADCASTING_MODEL/IDLE -> RECEIVING_UPDATES -> AGGREGATING_UPDATES
        if self.state_machine:
            current = self.state_machine.current_state
            if current == ServerState.BROADCASTING_MODEL:
                self.state_machine.transition_to(ServerState.RECEIVING_UPDATES)
            elif current == ServerState.IDLE:
                self.state_machine.transition_to(ServerState.RECEIVING_UPDATES)
            # Now transition to aggregating
            if self.state_machine.current_state == ServerState.RECEIVING_UPDATES:
                self.state_machine.transition_to(ServerState.AGGREGATING_UPDATES)
        
        updates, weights = zip(*client_payloads)

        new_encrypted_model = None
        if not self.use_fhe:
            aggregated_vector = self.aggregator.aggregate(list(updates), list(weights))
            temp_model = self._model_template
            current_pos = 0
            new_weights = []
            for layer in temp_model.get_weights():
                shape, size = layer.shape, layer.size
                new_weights.append(aggregated_vector[current_pos : current_pos + size].reshape(shape))
                current_pos += size
            new_encrypted_model = new_weights
        else:
            num_chunks = len(updates[0])
            aggregated_chunks = []
            #print(f"Server starting aggregation across {num_chunks} chunks...")
            for i in range(num_chunks):
                chunks_for_this_index = [client_update[i] for client_update in updates]
                agg_kwargs = {}
                if "integer" in self.aggregator.__class__.__name__.lower():
                    agg_kwargs['cc'] = self.fhe_manager.cc
                    agg_kwargs['vec_len'] = self.fhe_manager.slot_count
                elif "securekrum" in self.aggregator.__class__.__name__.lower():
                    agg_kwargs['cc'] = self.fhe_manager.cc
                    agg_kwargs['fhe_manager'] = self.fhe_manager
                aggregated_chunk = self.aggregator.aggregate(chunks_for_this_index, list(weights), **agg_kwargs)
                aggregated_chunks.append(aggregated_chunk)
            new_encrypted_model = aggregated_chunks

        self.encrypted_global_model = new_encrypted_model
        
        # State: AGGREGATING_UPDATES -> MODEL_UPDATED -> IDLE
        if self.state_machine:
            self.state_machine.transition_to(ServerState.MODEL_UPDATED)
            self.state_machine.transition_to(ServerState.IDLE)
        
        # Track memory usage if benchmarking is enabled
        if self.benchmark_manager:
            import psutil
            process = psutil.Process()
            memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
            self.benchmark_manager.log_event('server', 'Peak Memory Usage', memory_usage, unit='MB')
            
            # If we have test data, evaluate model metrics
            if self.x_test is not None and self._model_for_eval is not None:
                # We'll evaluate metrics after decryption in the high-level orchestrator
                # This is just placeholder to show where metrics would be collected
                pass

    def get_public_context(self) -> dict:
        if not self.use_fhe: return None
        if self.crypto_setting == "threshold":
            if not self.fhe_manager.cc: raise ValueError("CryptoContext not generated.")
            return {"cc": self.fhe_manager.cc}
        else:
            return self.fhe_manager.get_public_context()

    def export_benchmarks(self, filepath: str):
        if self.benchmark_manager: self.benchmark_manager.export_to_csv(filepath)

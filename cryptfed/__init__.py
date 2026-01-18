import numpy as np
import copy
import time
from tqdm import tqdm
import contextlib
import io
import logging
import random
from .core.federated_server import FederatedServer
from .core.protocol_state import (
    ProtocolCoordinator, 
    OrchestratorState, 
    ClientState,
    StateTransitionError, 
    ProtocolViolationError
)

def configure_logging(level=logging.INFO):
    """Configure logging for the Synergia package.
    
    Args:
        level: The logging level to set. Can be logging.DEBUG, logging.INFO,
              logging.WARNING, logging.ERROR, or logging.CRITICAL
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Set CrypTFed logger level
    logging.getLogger('cryptfed').setLevel(level)
from .core.federated_client import FederatedClient
from typing import List, Callable, Tuple, Dict, Any

class CrypTFed:
    """
    The main orchestrator for a CrypTFed federated learning experiment.
    This class handles the setup, execution, and all decryption ceremonies,
    ensuring the server remains zero-knowledge.
    
    The protocol state machine is always enabled to ensure formal protocol
    execution, provide audit trails, and validate state transitions.
    """
    def __init__(self,
                 model_fn: Callable,
                 clients: List[FederatedClient],
                 test_data: Tuple[np.ndarray, np.ndarray],
                 crypto_setting: str = "single_key",
                 threshold_parties: int = 3,
                 aggregator_name: str = "auto", aggregator_args: Dict = {},
                 use_fhe: bool = True, fhe_scheme: str = "ckks",
                 num_rounds: int = 3,
                 client_sampling_proportion: float = 1.0,
                 enable_benchmarking: bool = True,
                 benchmark_manager: object = None):

        self.model_fn = model_fn
        self.num_rounds = num_rounds
        self.clients = clients
        self.x_test, self.y_test = test_data
        self.client_sampling_proportion = client_sampling_proportion
        
        # Initialize protocol state machine (always enabled for proper protocol execution)
        self.protocol_coordinator = ProtocolCoordinator()
        self.state_machine = self.protocol_coordinator.register_orchestrator()
        logging.getLogger(__name__).info("Protocol state machine initialized")

        self.server = FederatedServer(
            model_fn=model_fn, crypto_setting=crypto_setting,
            threshold_parties=threshold_parties, aggregator_name=aggregator_name,
            aggregator_args=aggregator_args, use_fhe=use_fhe, fhe_scheme=fhe_scheme,
            enable_benchmarking=enable_benchmarking, benchmark_manager=benchmark_manager,
            protocol_coordinator=self.protocol_coordinator
        )

    def _reshape_vector_to_model(self, flat_vector: np.ndarray) -> List[np.ndarray]:
        """Helper to reshape a flat vector into the model's layer structure."""
        temp_model = self.model_fn()
        current_pos = 0
        reshaped_weights = []
        for layer in temp_model.get_weights():
            shape, size = layer.shape, layer.size
            reshaped_weights.append(flat_vector[current_pos : current_pos + size].reshape(shape))
            current_pos += size
        return reshaped_weights

    def _decrypt_and_reconstruct(self, encrypted_model: List[Any], clients_for_decryption: List[FederatedClient]) -> np.ndarray:
        """
        The main decryption and reconstruction entry point, managed by the orchestrator.
        """
        decrypted_padded_chunks = None

        if self.server.crypto_setting == "single_key":
            #print("Orchestrator asking a client to decrypt the global model...")
            decrypted_padded_chunks = self.clients[0].decrypt_model_chunks(encrypted_model)
        elif self.server.crypto_setting == "threshold":
            #print("Orchestrator starting collaborative decryption...")
            decrypted_padded_chunks = []
            
            for chunk in encrypted_model:
                dec_shares = [client.get_decryption_share(self.server.fhe_manager.cc, chunk) for client in clients_for_decryption]
                if "ckks" in self.server.fhe_scheme:
                    raw_chunk = chunk.data
                else:
                    raw_chunk = chunk
                dec_shares.insert(0, raw_chunk)
                decrypted_chunk = self.server.fhe_manager.merge_decryption_shares(dec_shares)
                decrypted_padded_chunks.append(decrypted_chunk)

        # --- FIXED RECONSTRUCTION LOGIC (properly handles chunking/padding) ---
        # The old logic just concatenated chunks and trimmed, but this doesn't handle
        # padding within chunks correctly. We need to properly reconstruct by 
        # respecting slot boundaries and only taking the actual data from each chunk.
        original_layers = self.model_fn().get_weights()
        model_vec_len = sum(w.size for w in original_layers)
        
        # Get slot count to understand chunking
        slot_count = self.server.fhe_manager.slot_count
        
        # Handle reconstruction differently for CKKS vs BFV/BGV (including threshold versions)
        is_ckks = hasattr(self.server.fhe_manager, '__class__') and ('CKKS' in self.server.fhe_manager.__class__.__name__)
        is_threshold = hasattr(self.server.fhe_manager, '__class__') and ('Threshold' in self.server.fhe_manager.__class__.__name__)
        
        if is_ckks and not is_threshold:
            # Regular CKKS: Use original simple concatenation (was working before)
            plaintext_result = np.concatenate(decrypted_padded_chunks)[:model_vec_len]
        else:
            # BFV/BGV (single-key and threshold) AND threshold CKKS: Use fixed-size chunk reconstruction
            if slot_count == 0 or len(decrypted_padded_chunks) == 1:
                # Fallback to simple concatenation if no chunking info or single chunk
                plaintext_result = np.concatenate(decrypted_padded_chunks)[:model_vec_len]
            else:
                # Proper reconstruction respecting chunk boundaries
                reconstructed_values = []
                remaining_elements = model_vec_len
                
                # For threshold CKKS, use effective slot count; for BFV/BGV use full slot count
                chunk_size = slot_count // 2 if (is_ckks and is_threshold) else slot_count
                
                for i, chunk in enumerate(decrypted_padded_chunks):
                    # Determine how many valid elements are in this chunk
                    elements_in_chunk = min(chunk_size, remaining_elements)
                    
                    if elements_in_chunk > 0:
                        # Take only the valid elements, ignore padding
                        valid_elements = chunk[:elements_in_chunk]
                        reconstructed_values.extend(valid_elements)
                        remaining_elements -= elements_in_chunk
                    
                    if remaining_elements <= 0:
                        break
                
                plaintext_result = np.array(reconstructed_values[:model_vec_len])
        # ------------------------------------
        # Final de-quantization for integer schemes
        if "integer" in self.server.aggregator.__class__.__name__.lower():
            if hasattr(self.server.aggregator, 'total_weight'):

                total_weight = self.server.aggregator.total_weight
                result = plaintext_result / total_weight
                return result   
            else:
                # Single model decryption - de-quantization already done by FHE manager
                return plaintext_result
        else:
            return plaintext_result

    def run(self):
        """Executes the entire federated learning simulation with a single static tqdm bar."""
        logger = logging.getLogger(__name__)
        
        # State: UNINITIALIZED -> INITIALIZING
        if self.state_machine:
            self.state_machine.transition_to(OrchestratorState.INITIALIZING, 
                                            {"num_clients": len(self.clients), "num_rounds": self.num_rounds})
        
        logger.info("Initializing federated session...")
        self.server.init_session(self.clients)
        public_context = self.server.get_public_context()

        logger.info("Connecting clients to server...")
        for client in self.clients:
            # Register client in protocol coordinator BEFORE connecting
            if self.protocol_coordinator:
                client.state_machine = self.protocol_coordinator.register_client(client.client_id)
            
            fhe_manager = self.server.fhe_manager if self.server.use_fhe else None
            client.connect_to_server(public_context, fhe_manager)
            # Ensure clients can log benchmarks even in plaintext mode by exposing the server's manager
            if self.server.benchmark_manager:
                setattr(client, 'benchmark_manager', self.server.benchmark_manager)

        logger.info("Initializing global model...")
        self.server.initialize_encrypted_global_model()
        
        # State: INITIALIZING -> SESSION_READY
        if self.state_machine:
            self.state_machine.transition_to(OrchestratorState.SESSION_READY, 
                                            {"crypto_setting": self.server.crypto_setting})

        n_clients = len(self.clients)
        avg_clients_per_round = int(n_clients * self.client_sampling_proportion)
        total_steps = self.num_rounds * (avg_clients_per_round + 1)  # clients + aggregation

        with tqdm(total=total_steps, desc="Federated Training", ncols=100) as pbar:
            for r in range(self.num_rounds):
                round_num = r + 1
                
                # State: SESSION_READY/ROUND_COMPLETE -> ROUND_STARTING
                if self.state_machine:
                    self.state_machine.transition_to(OrchestratorState.ROUND_STARTING, 
                                                    {"round": round_num})
                
                if self.server.benchmark_manager:
                    self.server.benchmark_manager.set_round(round_num)

                # --- Client Sampling ---
                num_participating_clients = max(1, int(n_clients * self.client_sampling_proportion))
                participating_clients = random.sample(self.clients, num_participating_clients)
                logger.info(f"Round {round_num}: Sampled {len(participating_clients)} of {n_clients} clients.")
                # ---------------------

                # State: ROUND_STARTING -> DISTRIBUTING_MODEL
                if self.state_machine:
                    self.state_machine.transition_to(OrchestratorState.DISTRIBUTING_MODEL, 
                                                    {"num_participating": len(participating_clients)})
                
                pbar.set_postfix_str(f" Redistribute model")
                encrypted_model = self.server.get_encrypted_global_model(for_broadcast=True)

                if self.server.use_fhe:
                    clients_for_decryption = self.clients
                    if self.server.crypto_setting == "threshold":
                        clients_for_decryption = random.sample(self.server.key_holders, self.server.threshold_parties)
                    plaintext_vector = self._decrypt_and_reconstruct(encrypted_model, clients_for_decryption)
                else:
                    plaintext_vector = np.concatenate([w.flatten() for w in encrypted_model])
                plaintext_weights = self._reshape_vector_to_model(plaintext_vector)

                # State: DISTRIBUTING_MODEL -> WAITING_FOR_CLIENTS
                if self.state_machine:
                    self.state_machine.transition_to(OrchestratorState.WAITING_FOR_CLIENTS, 
                                                    {"sampled_clients": [c.client_id for c in participating_clients]})
                
                client_payloads = []

                # Capture and silence prints inside client training
                for client_idx, client in enumerate(participating_clients, start=1):
                    with contextlib.redirect_stdout(io.StringIO()):
                        payload = client.execute_training_round(plaintext_weights)
                    client_payloads.append(payload)
                    pbar.update(1)
                    pbar.set_postfix_str(f"Round {round_num}/{self.num_rounds} - Client {client_idx}/{num_participating_clients}")

                # Detect payload mode: check if first payload is a ClientPayload object
                from .core.payload import ClientPayload
                is_payload_mode = isinstance(client_payloads[0], ClientPayload)

                # State: WAITING_FOR_CLIENTS -> COLLECTING_UPDATES
                if self.state_machine:
                    self.state_machine.transition_to(OrchestratorState.COLLECTING_UPDATES, 
                                                    {"num_updates": len(client_payloads)})
                
                # Aggregation phase
                start = time.time()

                # State: COLLECTING_UPDATES -> AGGREGATING
                if self.state_machine:
                    self.state_machine.transition_to(OrchestratorState.AGGREGATING, 
                                                    {"aggregator": self.server.aggregator.__class__.__name__})

                if self.server.use_fhe:
                    if self.server.aggregator.requires_plaintext_updates:
                        # SCENARIO A: FHE is ON, but aggregator needs plaintext (e.g., TrimmedMean)
                        pbar.set_postfix_str(f"Round {round_num}/{self.num_rounds} - Decrypting for Aggregation")
                        
                        plaintext_updates = []
                        weights = []
                        
                        for payload in client_payloads:
                            if is_payload_mode:
                                encrypted_update = payload.get_item("model_update").data
                                weight = payload.weight
                            else:
                                encrypted_update, weight = payload
                                
                            clients_for_decryption = random.sample(self.server.key_holders, self.server.threshold_parties)
                            decrypted_vector = self._decrypt_and_reconstruct(encrypted_update, clients_for_decryption)
                            plaintext_updates.append(decrypted_vector)
                            weights.append(weight)
                        
                        pbar.set_postfix_str(f"Round {round_num}/{self.num_rounds} - Plaintext Aggregation")
                        new_model_vector = self.server.aggregator.aggregate(plaintext_updates, weights)

                        self.server.encrypt_and_set_model(new_model_vector)
                    else:
                        # SCENARIO B: FHE is ON, and aggregation is homomorphic
                        # Check if aggregator can handle payloads directly
                        if is_payload_mode and hasattr(self.server.aggregator, 'aggregate_payloads'):
                            # Use payload-aware aggregation
                            self.server.aggregate_payloads_and_update(client_payloads)
                        else:
                            # Legacy aggregation (convert payloads to tuples if needed)
                            if is_payload_mode:
                                legacy_payloads = []
                                for p in client_payloads:
                                    model_data = p.get_item("model_update").data
                                    legacy_payloads.append((model_data, p.weight))
                                self.server.aggregate_and_update(legacy_payloads)
                            else:
                                self.server.aggregate_and_update(client_payloads)
                else:
                    # SCENARIO C: FHE is OFF, aggregation is on plaintext updates
                    if is_payload_mode and hasattr(self.server.aggregator, 'aggregate_payloads'):
                        self.server.aggregate_payloads_and_update(client_payloads)
                    elif is_payload_mode:
                        # Convert to legacy format
                        legacy_payloads = []
                        for p in client_payloads:
                            model_data = p.get_item("model_update").data
                            legacy_payloads.append((model_data, p.weight))
                        self.server.aggregate_and_update(legacy_payloads)
                    else:
                        self.server.aggregate_and_update(client_payloads)

                end = time.time()
                pbar.update(1)
                pbar.set_postfix_str(f"Round {round_num}/{self.num_rounds} - Aggregation {end - start:.4f}s")
                
                # State: AGGREGATING -> DECRYPTING_MODEL (if FHE) or EVALUATING (if plaintext)
                if self.state_machine:
                    if self.server.use_fhe:
                        self.state_machine.transition_to(OrchestratorState.DECRYPTING_MODEL)
                    else:
                        self.state_machine.transition_to(OrchestratorState.EVALUATING)
                
                # Evaluate model after each round if test data is available
                if self.x_test is not None and self.y_test is not None:
                    # State: DECRYPTING_MODEL -> EVALUATING (if was in decrypt state)
                    if self.state_machine and self.state_machine.current_state == OrchestratorState.DECRYPTING_MODEL:
                        self.state_machine.transition_to(OrchestratorState.EVALUATING)
                    
                    # Get the updated model weights AFTER aggregation
                    eval_encrypted_model = self.server.get_encrypted_global_model()
                    if self.server.use_fhe:
                        clients_for_decryption = self.clients
                        if self.server.crypto_setting == "threshold":
                            clients_for_decryption = random.sample(self.server.key_holders, self.server.threshold_parties)
                        eval_plaintext_vector = self._decrypt_and_reconstruct(eval_encrypted_model, clients_for_decryption)
                    else:
                        eval_plaintext_vector = np.concatenate([w.flatten() for w in eval_encrypted_model])
                    eval_plaintext_weights = self._reshape_vector_to_model(eval_plaintext_vector)
                    
                    eval_model = self.model_fn()
                    eval_model.set_weights(eval_plaintext_weights)  # Use UPDATED weights
                    loss, accuracy = eval_model.evaluate(self.x_test, self.y_test, verbose=0)
                    
                    if self.server.benchmark_manager:
                        self.server.benchmark_manager.log_event('Evaluation', 'Model Accuracy', accuracy * 100, unit='%')
                        self.server.benchmark_manager.log_event('Evaluation', 'Model Loss', loss, unit='loss')
                        
                    logger.info(f"Round {round_num} - Test Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
                
                # State: EVALUATING -> ROUND_COMPLETE
                if self.state_machine:
                    self.state_machine.transition_to(OrchestratorState.ROUND_COMPLETE, 
                                                    {"round": round_num, "total_rounds": self.num_rounds})
                
                # Transition all participating clients back to IDLE for next round
                for client in participating_clients:
                    if hasattr(client, 'state_machine') and client.state_machine:
                        if client.state_machine.current_state == ClientState.WAITING:
                            client.state_machine.transition_to(ClientState.IDLE)

        # State: ROUND_COMPLETE -> TRAINING_COMPLETE
        if self.state_machine:
            self.state_machine.transition_to(OrchestratorState.TRAINING_COMPLETE, 
                                            {"total_rounds": self.num_rounds})
        
        logger.info("\nFederated training finished.")
        final_model = self.model_fn()
        final_model.set_weights(plaintext_weights)
        return final_model

    def evaluate_and_export(self, filename: str):
        logger = logging.getLogger(__name__)
        logger.info("\n--- Evaluating final global model on the test set ---")
        final_encrypted_model = self.server.get_encrypted_global_model()
        final_plaintext_vector = None
        if self.server.use_fhe:
            clients_for_decryption = self.clients
            if self.server.crypto_setting == "threshold":
                clients_for_decryption = random.sample(self.server.key_holders, self.server.threshold_parties)
            final_plaintext_vector = self._decrypt_and_reconstruct(final_encrypted_model, clients_for_decryption)
        else:
            final_plaintext_vector = np.concatenate([w.flatten() for w in final_encrypted_model])

        final_model = self.model_fn()
        final_model.set_weights(self._reshape_vector_to_model(final_plaintext_vector))

        loss, accuracy = final_model.evaluate(self.x_test, self.y_test, verbose=2)
        logger.info(f"Final Global Model Accuracy: {accuracy:.4f}")

        if self.server.benchmark_manager:
            self.server.benchmark_manager.log_event('Evaluation', 'Final Accuracy', accuracy)
            self.server.export_benchmarks(filepath=filename)
            logger.info(f"\n Simulation complete. Benchmark data saved to '{filename}'.")

    def get_slot_size(self):
        """Get the slot size of the FHE scheme being used.
        
        Returns:
            int: The slot count/size of the FHE scheme, or None if not using FHE
        """
        if self.server.use_fhe and hasattr(self.server, 'fhe_manager'):
            return self.server.fhe_manager.slot_count
        return None

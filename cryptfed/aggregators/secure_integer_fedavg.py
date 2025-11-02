from .base_aggregator import BaseAggregator
from typing import List, Any
import time
import numpy as np

class SecureIntegerFedAvg(BaseAggregator):
    """
    Implements a cryptographically robust weighted average for integer schemes
    by encoding scalar weights as full plaintext vectors for multiplication.
    """
    def __init__(self, benchmark_manager: Any = None):
        super().__init__(benchmark_manager)
        # Must match BFV/BGV manager quantization scale for consistency
        self.weight_quantization_scale = float(2**20)
        

    @property
    def requires_plaintext_updates(self) -> bool:
        return False

    def aggregate(self, encrypted_updates: List[Any], weights: List[int], cc: Any = None, vec_len: int = 0) -> Any:
        if not encrypted_updates: raise ValueError("Cannot aggregate empty list.")
        if cc is None: raise ValueError("CryptoContext 'cc' is required.")
        if vec_len == 0: raise ValueError("Vector length 'vec_len' is required for this aggregator.")

        start_time = time.time()

        # For integer schemes, use actual integer weights without normalization
        # The normalization will be handled during decryption
        integer_weights = [int(w) for w in weights]
        total_weight = sum(integer_weights)
        if total_weight == 0:
            raise ValueError("Total weight cannot be zero")
        
        # Create plaintext vectors for integer client weights
        scalar_as_vector_0 = [integer_weights[0]] * vec_len
        ptx_weight_0 = cc.MakePackedPlaintext(scalar_as_vector_0)
        aggregated_sum = cc.EvalMult(encrypted_updates[0], ptx_weight_0)

        # 2. Loop for the rest of the updates.
        for i in range(1, len(encrypted_updates)):
            scalar_as_vector_i = [integer_weights[i]] * vec_len
            ptx_weight_i = cc.MakePackedPlaintext(scalar_as_vector_i)
            weighted_update = cc.EvalMult(encrypted_updates[i], ptx_weight_i)
            aggregated_sum = cc.EvalAdd(aggregated_sum, weighted_update)
        
        # Store total weight for normalization during decryption
        self.total_weight = total_weight
        
        end_time = time.time()

        # Record aggregation performance metrics
        self.aggregation_time = end_time - start_time
        self.encrypted_operations_count = len(encrypted_updates) + 1  # Number of multiplications + additions
        
        return aggregated_sum
        # -----------------------------------
        duration = time.time() - start_time
        if self.benchmark_manager:
            self.benchmark_manager.log_event('Server', 'Integer Weighted Avg Time', duration)

        #print(f"Integer weighted avg finished in {duration:.4f}s.")
        return aggregated_sum

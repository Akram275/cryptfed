from .base_aggregator import BaseAggregator
from typing import List, Any
import time
import openfhe_numpy as onp
import numpy as np

class SecureCkksFedAvg(BaseAggregator):
    """
    Implements the canonical Federated Averaging for the CKKS scheme.
    """
    @property
    def requires_plaintext_updates(self) -> bool:
        return False

    def aggregate(self, encrypted_updates: List[onp.array], weights: List[float], **kwargs) -> onp.array:
        """
        Performs a homomorphic weighted average. Ignores extra kwargs.
        """
        if not encrypted_updates:
            raise ValueError("Cannot aggregate an empty list.")

        start_time = time.time()

        total_weight = sum(weights)
        if total_weight == 0:
            return encrypted_updates[0]

        normalized_weights = [w / total_weight for w in weights]

        aggregated_result = encrypted_updates[0] * normalized_weights[0]

        for i in range(1, len(encrypted_updates)):
            weighted_update = encrypted_updates[i] * normalized_weights[i]
            aggregated_result += weighted_update

        duration = time.time() - start_time
        if self.benchmark_manager:
            self.benchmark_manager.log_event('Server', 'CKKS Weighted Aggregation Time', duration)

        return aggregated_result

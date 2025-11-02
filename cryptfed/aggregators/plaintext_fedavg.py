from .base_aggregator import BaseAggregator
from typing import List
import numpy as np
import time

class PlaintextFedAvg(BaseAggregator):
    """
    Implements a non-secure, plaintext Federated Averaging algorithm.

    This aggregator performs a weighted average directly on unencrypted NumPy arrays.
    It serves as a performance baseline to measure the overhead of FHE.
    """
    @property
    def requires_plaintext_updates(self) -> bool:
        return True # This aggregator only works on plaintext data.

    def aggregate(self, updates: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """
        Performs a simple weighted average on a list of plaintext model vectors.
        """
        if not updates or not weights:
            raise ValueError("Cannot aggregate empty lists.")
        if len(updates) != len(weights):
            raise ValueError("Number of updates must match the number of weights.")

        #print(f"Starting plaintext weighted aggregation of {len(updates)} updates...")
        start_time = time.time()

        total_weight = sum(weights)
        if total_weight == 0:
            return updates[0] # Avoid division by zero, although unlikely

        # Create a zero vector of the correct shape
        weighted_sum = np.zeros_like(updates[0], dtype=np.float64)

        # Calculate the weighted sum of all model vectors
        for update, weight in zip(updates, weights):
            weighted_sum += update * weight

        averaged_model = weighted_sum / total_weight

        duration = time.time() - start_time

        if self.benchmark_manager:
            self.benchmark_manager.log_event('Server', 'Plaintext Aggregation Time', duration, 'seconds')

        #print(f"Plaintext aggregation finished in {duration:.6f}s.")
        return averaged_model

from .base_aggregator import BaseAggregator
from typing import List, Any
import time
import openfhe_numpy as onp
import logging

class SecureFedAvg(BaseAggregator):
    """
    Implements Federated Averaging by homomorphically summing encrypted updates
    using the OpenFHE-NumPy API.
    """
    def aggregate(self, encrypted_updates: List[onp.array]) -> onp.array:
        """Homomorphically sums all encrypted onp.array updates."""
        if not encrypted_updates:
            raise ValueError("Cannot aggregate an empty list of updates.")

        logging.getLogger(__name__).info(f"Starting homomorphic aggregation of {len(encrypted_updates)} updates...")
        start_time = time.time()

        # Start with the first encrypted tensor
        aggregated_result = encrypted_updates[0]

        for i in range(1, len(encrypted_updates)):
            aggregated_result += encrypted_updates[i]

        duration = time.time() - start_time

        if self.benchmark_manager:
            self.benchmark_manager.log_event('Server', 'Aggregation Time', duration, 'seconds')

        logging.getLogger(__name__).info(f"Aggregation finished in {duration:.4f}s.")
        return aggregated_result

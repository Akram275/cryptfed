from .base_aggregator import BaseAggregator
from typing import List, Any
import numpy as np
import time
import logging

class FedTrimmedMean(BaseAggregator):
    """
    Implements Federated Trimmed Mean, a robust aggregation strategy.
    """
    def __init__(self, beta: int = 1, benchmark_manager: Any = None):
        super().__init__(benchmark_manager)
        if beta < 0:
            raise ValueError("Beta (number of clients to trim) cannot be negative.")
        self.beta = beta

    @property
    def requires_plaintext_updates(self) -> bool:
        return True

    def aggregate(self, updates: List[np.ndarray], weights: List[float], **kwargs) -> np.ndarray:
        """
        Performs trimmed mean aggregation. Ignores weights and extra kwargs.
        """
        if not updates:
            raise ValueError("Cannot aggregate an empty list of updates.")
        if 2 * self.beta >= len(updates):
            raise ValueError("Cannot trim all clients; 2 * beta must be less than the number of clients.")

        logging.getLogger(__name__).info(f"Starting trimmed mean aggregation of {len(updates)} updates, trimming {self.beta} outliers...")
        start_time = time.time()

        stacked_updates = np.vstack(updates)
        sorted_updates = np.sort(stacked_updates, axis=0)

        if self.beta > 0:
            trimmed_updates = sorted_updates[self.beta:-self.beta, :]
        else:
            trimmed_updates = sorted_updates

        new_global_model = np.mean(trimmed_updates, axis=0)

        duration = time.time() - start_time

        if self.benchmark_manager:
            self.benchmark_manager.log_event('Server', 'Trimmed Mean Aggregation Time', duration)

        logging.getLogger(__name__).info(f"Trimmed mean aggregation finished in {duration:.4f}s.")
        return new_global_model

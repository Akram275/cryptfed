from .base_aggregator import BaseAggregator
from typing import List, Any, Optional
import time
import openfhe_numpy as onp
import numpy as np
import logging

class FedAvgMomentum(BaseAggregator):
    """
    Implements the stateless calculation for one round of FedAvg with Momentum.
    """
    def __init__(self, mu: float = 0.9, benchmark_manager: Any = None):
        super().__init__(benchmark_manager)
        if not 0.0 <= mu < 1.0:
            raise ValueError("Momentum coefficient 'mu' must be in [0, 1).")
        self.mu = mu

    @property
    def requires_plaintext_updates(self) -> bool:
        return False

    def aggregate(self, encrypted_updates: List[onp.array], weights: List[float], prev_momentum_vector: Optional[onp.array] = None, **kwargs) -> onp.array:
        """
        Performs a homomorphic weighted sum and applies momentum. Ignores extra kwargs.
        """
        if not encrypted_updates:
            raise ValueError("Cannot aggregate an empty list.")

        logging.getLogger(__name__).info(f"Starting FedAvg with Momentum aggregation...")
        start_time = time.time()

        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        current_average = encrypted_updates[0] * normalized_weights[0]
        for i in range(1, len(encrypted_updates)):
            current_average += encrypted_updates[i] * normalized_weights[i]

        if prev_momentum_vector is None:
            new_momentum = current_average
        else:
            term1 = prev_momentum_vector * self.mu
            new_momentum = term1 + current_average

        duration = time.time() - start_time

        if self.benchmark_manager:
            self.benchmark_manager.log_event('Server', 'FedAvgM Aggregation Time', duration, 'seconds')

        logging.getLogger(__name__).info(f"FedAvgM aggregation finished in {duration:.4f}s.")
        return new_momentum

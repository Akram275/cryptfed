from .base_aggregator import BaseAggregator
from typing import List, Any
import numpy as np
import time
import logging

class FedProx(BaseAggregator):
    """
    Implements FedProx aggregation with server-side momentum.
    FedProx is designed to handle statistical heterogeneity (non-IID data)
    by adding a proximal term during local training and optional momentum during aggregation.
    
    Reference: "Federated Optimization in Heterogeneous Networks" by Li et al. (2020)
    
    Note: This aggregator can work with both encrypted and plaintext updates,
    but the momentum term requires access to the previous global model.
    """
    def __init__(self, momentum: float = 0.0, benchmark_manager: Any = None):
        """
        Args:
            momentum: Server-side momentum coefficient (0.0 = no momentum, typical range: 0.9-0.99)
        """
        super().__init__(benchmark_manager)
        self.momentum = momentum
        self.velocity = None  # Stores the momentum term

    @property
    def requires_plaintext_updates(self) -> bool:
        # FedProx can work with encrypted updates, but momentum requires plaintext access
        return self.momentum > 0.0

    def aggregate(self, updates: List[np.ndarray], weights: List[float], **kwargs) -> np.ndarray:
        """
        Performs FedProx aggregation with optional server-side momentum.
        """
        if not updates:
            raise ValueError("Cannot aggregate an empty list of updates.")

        logging.getLogger(__name__).info(f"Starting FedProx aggregation of {len(updates)} updates (momentum={self.momentum})...")
        start_time = time.time()

        # Weighted average of updates
        total_weight = sum(weights)
        if total_weight == 0:
            # Fallback to uniform weighting
            weights = [1.0] * len(updates)
            total_weight = len(updates)

        weighted_update = np.zeros_like(updates[0])
        for update, weight in zip(updates, weights):
            weighted_update += (weight / total_weight) * update

        # Apply server-side momentum if enabled
        if self.momentum > 0.0:
            if self.velocity is None:
                # Initialize velocity on first round
                self.velocity = np.zeros_like(weighted_update)
            
            # Update velocity: v = momentum * v + update
            self.velocity = self.momentum * self.velocity + weighted_update
            new_global_model = self.velocity
        else:
            new_global_model = weighted_update

        duration = time.time() - start_time

        if self.benchmark_manager:
            self.benchmark_manager.log_event('Server', 'FedProx Aggregation Time', duration)

        logging.getLogger(__name__).info(f"FedProx aggregation finished in {duration:.4f}s.")
        return new_global_model
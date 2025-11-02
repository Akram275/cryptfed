from .base_aggregator import BaseAggregator
from typing import List, Any
import numpy as np
import time
import logging

class FedMedian(BaseAggregator):
    """
    Implements Federated Median aggregation, a robust strategy against outliers.
    This aggregator computes the coordinate-wise median of client updates.
    
    Note: This requires plaintext updates as median computation is not easily 
    homomorphic.
    """
    def __init__(self, benchmark_manager: Any = None):
        super().__init__(benchmark_manager)

    @property
    def requires_plaintext_updates(self) -> bool:
        return True

    def aggregate(self, updates: List[np.ndarray], weights: List[float], **kwargs) -> np.ndarray:
        """
        Performs coordinate-wise median aggregation. Ignores weights.
        """
        if not updates:
            raise ValueError("Cannot aggregate an empty list of updates.")

        logging.getLogger(__name__).info(f"Starting median aggregation of {len(updates)} updates...")
        start_time = time.time()

        # Stack updates and compute coordinate-wise median
        stacked_updates = np.vstack(updates)
        new_global_model = np.median(stacked_updates, axis=0)

        duration = time.time() - start_time

        if self.benchmark_manager:
            self.benchmark_manager.log_event('Server', 'Median Aggregation Time', duration)

        logging.getLogger(__name__).info(f"Median aggregation finished in {duration:.4f}s.")
        return new_global_model
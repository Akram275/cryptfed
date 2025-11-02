from .base_aggregator import BaseAggregator
from typing import List, Any
import numpy as np
import time
import logging

class Krum(BaseAggregator):
    """
    Implements the Krum aggregation algorithm, designed to be robust against 
    Byzantine attacks. Krum selects the update that is closest to its neighbors.
    
    Reference: "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
    by Blanchard et al. (2017)
    
    Note: This requires plaintext updates for distance computations.
    """
    def __init__(self, f: int = 0, benchmark_manager: Any = None):
        """
        Args:
            f: Maximum number of Byzantine clients the algorithm should tolerate.
               If f=0, it will be automatically set to (n-1)//2 where n is the number of clients.
        """
        super().__init__(benchmark_manager)
        self.f = f

    @property
    def requires_plaintext_updates(self) -> bool:
        return True

    def aggregate(self, updates: List[np.ndarray], weights: List[float], **kwargs) -> np.ndarray:
        """
        Performs Krum aggregation by selecting the most central update.
        """
        if not updates:
            raise ValueError("Cannot aggregate an empty list of updates.")
        
        n = len(updates)
        if self.f == 0:
            f = (n - 1) // 2  # Default: tolerate up to half of the clients being Byzantine
        else:
            f = self.f
            
        if n <= 2 * f:
            raise ValueError(f"Krum requires n > 2f, but got n={n}, f={f}")

        logging.getLogger(__name__).info(f"Starting Krum aggregation of {n} updates, tolerating {f} Byzantine clients...")
        start_time = time.time()

        # Convert to numpy array for easier computation
        updates_array = np.array(updates)
        
        # Compute pairwise squared distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(updates_array[i] - updates_array[j]) ** 2
                distances[i, j] = dist
                distances[j, i] = dist

        # For each client, compute the score (sum of distances to n-f-2 closest neighbors)
        scores = np.zeros(n)
        for i in range(n):
            # Sort distances to other clients (excluding self)
            client_distances = np.sort(distances[i])
            # Sum distances to n-f-2 closest neighbors (excluding distance to self which is 0)
            scores[i] = np.sum(client_distances[1:n-f-1])

        # Select the client with the smallest score
        selected_idx = np.argmin(scores)
        new_global_model = updates[selected_idx]

        duration = time.time() - start_time

        if self.benchmark_manager:
            self.benchmark_manager.log_event('Server', 'Krum Aggregation Time', duration)
            self.benchmark_manager.log_event('Server', 'Krum Selected Client', selected_idx)

        logging.getLogger(__name__).info(f"Krum aggregation finished in {duration:.4f}s. Selected client {selected_idx}.")
        return new_global_model
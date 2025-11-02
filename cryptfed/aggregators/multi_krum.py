from .base_aggregator import BaseAggregator
from typing import List, Any
import numpy as np
import time
import logging

class MultiKrum(BaseAggregator):
    """
    Implements the Multi-Krum aggregation algorithm, an extension of Krum that
    averages the m best updates instead of selecting just one.
    
    Reference: "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
    by Blanchard et al. (2017)
    
    Note: This requires plaintext updates for distance computations.
    """
    def __init__(self, f: int = 0, m: int = 0, benchmark_manager: Any = None):
        """
        Args:
            f: Maximum number of Byzantine clients to tolerate.
            m: Number of best updates to average. If 0, defaults to n-f.
        """
        super().__init__(benchmark_manager)
        self.f = f
        self.m = m

    @property
    def requires_plaintext_updates(self) -> bool:
        return True

    def aggregate(self, updates: List[np.ndarray], weights: List[float], **kwargs) -> np.ndarray:
        """
        Performs Multi-Krum aggregation by averaging the m best updates.
        """
        if not updates:
            raise ValueError("Cannot aggregate an empty list of updates.")
        
        n = len(updates)
        if self.f == 0:
            f = (n - 1) // 2  # Default: tolerate up to half of the clients being Byzantine
        else:
            f = self.f
            
        if self.m == 0:
            m = n - f  # Default: use all non-Byzantine updates
        else:
            m = min(self.m, n - f)  # Ensure m doesn't exceed available good updates
            
        if n <= 2 * f:
            raise ValueError(f"Multi-Krum requires n > 2f, but got n={n}, f={f}")
        if m <= 0:
            raise ValueError(f"Multi-Krum requires m > 0, but got m={m}")

        logging.getLogger(__name__).info(f"Starting Multi-Krum aggregation of {n} updates, tolerating {f} Byzantine clients, averaging {m} best...")
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

        # Select the m clients with the smallest scores
        selected_indices = np.argsort(scores)[:m]
        selected_updates = [updates[i] for i in selected_indices]
        
        # Average the selected updates
        new_global_model = np.mean(selected_updates, axis=0)

        duration = time.time() - start_time

        if self.benchmark_manager:
            self.benchmark_manager.log_event('Server', 'Multi-Krum Aggregation Time', duration)
            self.benchmark_manager.log_event('Server', 'Multi-Krum Selected Clients', len(selected_indices))

        logging.getLogger(__name__).info(f"Multi-Krum aggregation finished in {duration:.4f}s. Averaged {m} best updates.")
        return new_global_model
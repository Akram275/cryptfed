from .base_aggregator import BaseAggregator
from typing import List, Any
import numpy as np
import time
import logging

class FlameByzantine(BaseAggregator):
    """
    Implements the FLAME (Federated Learning with Malicious Experts) aggregation algorithm.
    FLAME uses clustering to identify and filter out Byzantine updates.
    
    Reference: "FLAME: Taming Backdoors in Federated Learning" by Nguyen et al. (2022)
    
    Note: This requires plaintext updates for clustering operations.
    """
    def __init__(self, cluster_threshold: float = 0.5, min_cluster_size: int = 2, benchmark_manager: Any = None):
        """
        Args:
            cluster_threshold: Threshold for determining if updates belong to the same cluster
            min_cluster_size: Minimum number of updates required to form a valid cluster
        """
        super().__init__(benchmark_manager)
        self.cluster_threshold = cluster_threshold
        self.min_cluster_size = min_cluster_size

    @property
    def requires_plaintext_updates(self) -> bool:
        return True

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def _cluster_updates(self, updates: List[np.ndarray]) -> List[List[int]]:
        """Cluster updates based on cosine similarity."""
        n = len(updates)
        clusters = []
        assigned = [False] * n

        for i in range(n):
            if assigned[i]:
                continue
            
            # Start a new cluster with update i
            cluster = [i]
            assigned[i] = True
            
            # Find similar updates
            for j in range(i + 1, n):
                if assigned[j]:
                    continue
                
                similarity = self._cosine_similarity(updates[i], updates[j])
                if similarity >= self.cluster_threshold:
                    cluster.append(j)
                    assigned[j] = True
            
            clusters.append(cluster)
        
        return clusters

    def aggregate(self, updates: List[np.ndarray], weights: List[float], **kwargs) -> np.ndarray:
        """
        Performs FLAME aggregation by clustering updates and using the largest cluster.
        """
        if not updates:
            raise ValueError("Cannot aggregate an empty list of updates.")

        logging.getLogger(__name__).info(f"Starting FLAME aggregation of {len(updates)} updates...")
        start_time = time.time()

        # Cluster the updates
        clusters = self._cluster_updates(updates)
        
        # Filter clusters by minimum size
        valid_clusters = [cluster for cluster in clusters if len(cluster) >= self.min_cluster_size]
        
        if not valid_clusters:
            # Fallback: if no valid clusters, use all updates
            logging.getLogger(__name__).warning("No valid clusters found, falling back to simple average")
            new_global_model = np.mean(updates, axis=0)
        else:
            # Select the largest valid cluster
            largest_cluster = max(valid_clusters, key=len)
            cluster_updates = [updates[i] for i in largest_cluster]
            cluster_weights = [weights[i] for i in largest_cluster]
            
            # Weighted average of the selected cluster
            total_weight = sum(cluster_weights)
            if total_weight == 0:
                new_global_model = np.mean(cluster_updates, axis=0)
            else:
                new_global_model = np.zeros_like(cluster_updates[0])
                for update, weight in zip(cluster_updates, cluster_weights):
                    new_global_model += (weight / total_weight) * update

        duration = time.time() - start_time

        if self.benchmark_manager:
            self.benchmark_manager.log_event('Server', 'FLAME Aggregation Time', duration)
            self.benchmark_manager.log_event('Server', 'FLAME Clusters Found', len(clusters))
            if valid_clusters:
                self.benchmark_manager.log_event('Server', 'FLAME Largest Cluster Size', len(max(valid_clusters, key=len)))

        logging.getLogger(__name__).info(f"FLAME aggregation finished in {duration:.4f}s. Found {len(clusters)} clusters.")
        return new_global_model
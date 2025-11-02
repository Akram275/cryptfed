from .base_aggregator import BaseAggregator
from typing import List, Any
import numpy as np
import time
import logging

class FoolsGold(BaseAggregator):
    """
    Implements the FoolsGold aggregation algorithm for Byzantine-robust federated learning.
    FoolsGold uses historical similarity patterns to detect and down-weight Byzantine clients.
    
    Reference: "The Hidden Vulnerability of Distributed Learning in Byzantium" 
    by Fung et al. (2018)
    
    Note: This requires plaintext updates and maintains historical state.
    """
    def __init__(self, memory_size: int = 10, learning_rate: float = 1.0, benchmark_manager: Any = None):
        """
        Args:
            memory_size: Number of recent rounds to remember for similarity computation
            learning_rate: Learning rate for updating client reputation scores
        """
        super().__init__(benchmark_manager)
        self.memory_size = memory_size
        self.learning_rate = learning_rate
        self.client_history = {}  # Store recent updates for each client
        self.client_scores = {}   # Store reputation scores for each client

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

    def _update_client_history(self, client_ids: List[str], updates: List[np.ndarray]):
        """Update the historical record of client updates."""
        for client_id, update in zip(client_ids, updates):
            if client_id not in self.client_history:
                self.client_history[client_id] = []
            
            self.client_history[client_id].append(update.copy())
            
            # Maintain memory size limit
            if len(self.client_history[client_id]) > self.memory_size:
                self.client_history[client_id].pop(0)

    def _compute_similarity_scores(self, client_ids: List[str], current_updates: List[np.ndarray]) -> np.ndarray:
        """Compute pairwise similarity scores between current updates."""
        n = len(client_ids)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_similarity(current_updates[i], current_updates[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
        
        return similarity_matrix

    def aggregate(self, updates: List[np.ndarray], weights: List[float], client_ids: List[str] = None, **kwargs) -> np.ndarray:
        """
        Performs FoolsGold aggregation using historical similarity patterns.
        
        Args:
            client_ids: List of client identifiers (required for maintaining history)
        """
        if not updates:
            raise ValueError("Cannot aggregate an empty list of updates.")
        
        if client_ids is None:
            # Generate default client IDs if not provided
            client_ids = [f"client_{i}" for i in range(len(updates))]
        
        if len(client_ids) != len(updates):
            raise ValueError("Number of client_ids must match number of updates")

        logging.getLogger(__name__).info(f"Starting FoolsGold aggregation of {len(updates)} updates...")
        start_time = time.time()

        # Update client history
        self._update_client_history(client_ids, updates)
        
        # Compute current similarity scores
        similarity_matrix = self._compute_similarity_scores(client_ids, updates)
        
        # Initialize or update client reputation scores
        n = len(client_ids)
        current_scores = np.ones(n)  # Default score of 1.0
        
        for i, client_id in enumerate(client_ids):
            if client_id not in self.client_scores:
                self.client_scores[client_id] = 1.0
            
            # Penalize clients that are too similar to others (potential collusion)
            similarity_penalty = np.sum(similarity_matrix[i]) - 1.0  # Exclude self-similarity
            similarity_penalty = max(0, similarity_penalty)  # Only penalize positive similarities
            
            # Update score with learning rate
            self.client_scores[client_id] *= (1.0 - self.learning_rate * similarity_penalty)
            self.client_scores[client_id] = max(0.1, self.client_scores[client_id])  # Minimum score
            
            current_scores[i] = self.client_scores[client_id]
        
        # Normalize scores
        current_scores = current_scores / np.sum(current_scores)
        
        # Weighted aggregation using FoolsGold scores
        new_global_model = np.zeros_like(updates[0])
        for i, (update, score) in enumerate(zip(updates, current_scores)):
            new_global_model += score * update

        duration = time.time() - start_time

        if self.benchmark_manager:
            self.benchmark_manager.log_event('Server', 'FoolsGold Aggregation Time', duration)
            self.benchmark_manager.log_event('Server', 'FoolsGold Min Score', np.min(current_scores))
            self.benchmark_manager.log_event('Server', 'FoolsGold Max Score', np.max(current_scores))

        logging.getLogger(__name__).info(f"FoolsGold aggregation finished in {duration:.4f}s.")
        return new_global_model
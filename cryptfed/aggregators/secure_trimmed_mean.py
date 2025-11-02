from .base_aggregator import BaseAggregator
from typing import List, Any
import numpy as np
import logging

class SecureTrimmedMean(BaseAggregator):
    """
    Implements a secure, interactive coordinate-wise trimmed mean.

    This aggregator requires an interactive protocol with the clients to
    compute the trimmed mean of blinded (randomized) values. The server
    orchestrates this process but never sees the individual decrypted updates.
    """
    def __init__(self, beta: int = 1, benchmark_manager: Any = None):
        super().__init__(benchmark_manager)
        if beta < 0:
            raise ValueError("Beta (number of clients to trim) cannot be negative.")
        self.beta = beta

    @property
    def requires_plaintext_updates(self) -> bool:
        return False

    @property
    def is_interactive(self) -> bool:
        return True

    def aggregate(self, updates: List[Any], weights: List[float], **kwargs) -> np.ndarray:
        """
        This method is a placeholder. The actual aggregation is handled by
        an interactive protocol in the Synergia orchestrator.
        """
        raise NotImplementedError("SecureTrimmedMean requires an interactive protocol orchestrated by Synergia.")

from abc import ABC, abstractmethod
from typing import List, Any
import numpy as np

class BaseAggregator(ABC):
    """Abstract Base Class for all aggregation strategies."""
    def __init__(self, benchmark_manager: Any = None):
        self.benchmark_manager = benchmark_manager

    @property
    @abstractmethod
    def requires_plaintext_updates(self) -> bool:
        pass

    @property
    def is_interactive(self) -> bool:
        """Returns True if the aggregator requires an interactive protocol."""
        return False

    @abstractmethod
    def aggregate(self, updates: List[Any], weights: List[float], cc: Any = None, vec_len: int = 0) -> np.ndarray:
        """
        Aggregates a list of client updates using a unified interface.

        Args:
            updates (List[Any]): Client model vectors (encrypted or plaintext).
            weights (List[float]): Corresponding plaintext weights.
            cc (Any, optional): The FHE CryptoContext, required by some aggregators.
            vec_len (int, optional): The length of the model vector, required by some.
        """
        pass

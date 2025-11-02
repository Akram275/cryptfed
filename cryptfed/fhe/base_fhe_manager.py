from abc import ABC, abstractmethod
from typing import List, Any
import numpy as np

class BaseFHEManager(ABC):
    """Abstract Base Class for all FHE Managers."""
    def __init__(self, security_level: str = 'standard', benchmark_manager: Any = None):
        self.benchmark_manager = benchmark_manager
        self._public_context = None
        self._secret_key = None
        self.security_level = security_level
        self.slot_count = 0 # To be set by concrete class

    @abstractmethod
    def generate_crypto_context_and_keys(self) -> None:
        pass

    @abstractmethod
    def encrypt(self, data_chunks: List[np.ndarray], client_id: str) -> List[Any]:
        """Encrypts a list of numpy array chunks into a list of ciphertexts."""
        pass

    @abstractmethod
    def decrypt(self, ciphertext_chunks: List[Any], length: int = 0) -> np.ndarray:
        """Decrypts a list of ciphertexts and reconstructs the original vector."""
        pass

    def get_public_context(self) -> dict:
        if not self._public_context:
            raise ValueError("Cryptographic context has not been generated yet.")
        return self._public_context

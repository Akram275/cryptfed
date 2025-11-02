from .base_fhe_manager import BaseFHEManager
from typing import Any, List
import numpy as np
import time
from openfhe import *
import openfhe_numpy as onp

class CKKSManager(BaseFHEManager):
    def __init__(self, security_level: str = 'standard', benchmark_manager: Any = None):
        super().__init__(security_level, benchmark_manager)
        self.cc = None
        self.keys = None
        self.quantization_scale = 1.0

    def generate_crypto_context_and_keys(self) -> None:
        start_time = time.time()
        params = CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(2)
        params.SetScalingModSize(50)
        params.SetRingDim(32768)
        self.cc = GenCryptoContext(params)
        self.cc.Enable(PKESchemeFeature.PKE)
        self.cc.Enable(PKESchemeFeature.KEYSWITCH)
        self.cc.Enable(PKESchemeFeature.LEVELEDSHE)
        self.keys = self.cc.KeyGen()
        self.cc.EvalMultKeyGen(self.keys.secretKey)
        self._public_context = {"cc": self.cc, "pk": self.keys.publicKey}
        self._secret_key = self.keys.secretKey
        self.slot_count = self.cc.GetRingDimension() // 2
        duration = time.time() - start_time
        if self.benchmark_manager: self.benchmark_manager.log_event('FHE Setup (CKKS)', 'Setup Time', duration)
        print(f"CKKS context generated. Slot count: {self.slot_count}")

    def encrypt(self, data_chunks: List[np.ndarray], client_id: str) -> List[onp.array]:
        """Encrypts each chunk of the model into a separate ciphertext."""
        encrypted_chunks = []
        for chunk in data_chunks:
            encrypted_tensor = onp.array(
                cc=self.cc, data=chunk, public_key=self._public_context["pk"], fhe_type="C"
            )
            encrypted_chunks.append(encrypted_tensor)
        return encrypted_chunks

    def decrypt(self, ciphertext_chunks: List[onp.array], **kwargs) -> List[np.ndarray]:
        """
        Decrypts each ciphertext chunk and returns a list of padded numpy arrays.
        The server is responsible for concatenation and trimming.
        """
        decrypted_chunks = []
        for chunk in ciphertext_chunks:
            # The high-level API handles unpacking and returns a padded numpy array
            decrypted_chunk = chunk.decrypt(self._secret_key, unpack_type="original")
            decrypted_chunks.append(decrypted_chunk)
        return decrypted_chunks

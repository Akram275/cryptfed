from .base_fhe_manager import BaseFHEManager
from typing import Any, List
import numpy as np
import time
from openfhe import *
import math
import logging

class BGVManager(BaseFHEManager):
    def __init__(self, security_level: str = 'standard', benchmark_manager: Any = None):
        super().__init__(security_level, benchmark_manager)
        self.cc = None
        # Use same quantization scale as SecureIntegerFedAvg for consistency
        self.quantization_scale = float(2**16)

    def generate_crypto_context_and_keys(self) -> None:
        start_time = time.time()
        params = CCParamsBGVRNS()
        params.SetMultiplicativeDepth(2)
        p = 17180557313
        params.SetPlaintextModulus(p)
        self.cc = GenCryptoContext(params)
        self.slot_count = self.cc.GetRingDimension()
        self.cc.Enable(PKESchemeFeature.PKE)
        self.cc.Enable(PKESchemeFeature.KEYSWITCH)
        self.cc.Enable(PKESchemeFeature.LEVELEDSHE)
        keys = self.cc.KeyGen()
        self.cc.EvalMultKeyGen(keys.secretKey)
        self._public_context = {"pk": keys.publicKey}
        self._secret_key = keys.secretKey
        duration = time.time() - start_time
        if self.benchmark_manager:
            self.benchmark_manager.log_event('FHE Setup (BGV)', 'Setup Time', duration)
        logging.getLogger(__name__).info(f"BGV context generated. Slot count: {self.slot_count}, Plaintext Modulus: {p} ({np.round(math.log(p, 2))} bits)")

    def encrypt(self, data_chunks: List[np.ndarray], client_id: str) -> List[Any]:
        encrypted_chunks = []
        for chunk in data_chunks:
            quantized_data = (chunk * self.quantization_scale).round().astype(np.int64).tolist()
            if len(quantized_data) < self.slot_count:
                quantized_data.extend([0] * (self.slot_count - len(quantized_data)))
            plaintext = self.cc.MakePackedPlaintext(quantized_data)
            encrypted_chunks.append(self.cc.Encrypt(self._public_context["pk"], plaintext))
        return encrypted_chunks

    def decrypt(self, ciphertext_chunks: List[Any], **kwargs) -> List[np.ndarray]:
        """
        Decrypts each ciphertext chunk and returns a list of padded numpy arrays.
        """
        decrypted_chunks = []
        for chunk in ciphertext_chunks:
            decrypted_plaintext = self.cc.Decrypt(self._secret_key, chunk)
            decrypted_plaintext.SetLength(self.slot_count)
            int_vector = np.array(decrypted_plaintext.GetPackedValue(), dtype=np.int64)

            p = self.cc.GetPlaintextModulus()
            half_p = p // 2
            int_vector[int_vector > half_p] -= p

            float_vector = int_vector.astype(np.float64)
            dequantized_vector = float_vector / self.quantization_scale
            decrypted_chunks.append(dequantized_vector)
        return decrypted_chunks

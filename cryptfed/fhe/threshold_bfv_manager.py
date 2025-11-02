from .base_fhe_manager import BaseFHEManager
from typing import Any, List
import numpy as np
import math
import time
from openfhe import *
import logging

class ThresholdBFVManager(BaseFHEManager):
    def __init__(self, security_level: str = 'standard', benchmark_manager: Any = None, threshold: int = 1):
        super().__init__(security_level, benchmark_manager)
        self.cc = None
        self.joint_public_key = None
        # Use same quantization scale as single-key BFV for consistency
        # (reduced from 2^20 to 2^16 to avoid overflow issues)
        self.quantization_scale = float(2**16)
        self.threshold = threshold

    def generate_crypto_context(self) -> Any:
        params = CCParamsBFVRNS()
        params.SetMultiplicativeDepth(3)
        p = 17180557313  # A much larger prime
        params.SetPlaintextModulus(p)
        params.SetThresholdNumOfParties(self.threshold)
        self.cc = GenCryptoContext(params)
        self.slot_count = self.cc.GetRingDimension()
        self.cc.Enable(PKESchemeFeature.PKE)
        self.cc.Enable(PKESchemeFeature.KEYSWITCH)
        self.cc.Enable(PKESchemeFeature.LEVELEDSHE)
        self.cc.Enable(PKESchemeFeature.MULTIPARTY)
        logging.getLogger(__name__).info(f"Threshold BFV context generated. Slot count: {self.slot_count}, Plaintext Modulus: {p} ({np.round(math.log(p, 2))} bits)")
        return self.cc

    def generate_crypto_context_and_keys(self):
        raise NotImplementedError("Single-key generation is not supported in Threshold mode.")

    def collaborative_keygen(self, secret_key_shares: List[Any]):
        if not self.cc: raise RuntimeError("CryptoContext not generated.")

        # Step 1: Generate the joint public key from the secret shares.
        joint_key_pair = self.cc.MultipartyKeyGen(secret_key_shares)
        self.joint_public_key = joint_key_pair.publicKey

        logging.getLogger(__name__).info("Generating joint evaluation keys for relinearization...")
        for sk in secret_key_shares:
            self.cc.EvalMultKeyGen(sk)
        logging.getLogger(__name__).info("Joint evaluation keys generated successfully.")
        # ------------------------

    def encrypt(self, data_chunks: List[np.ndarray], client_id: str) -> List[Any]:
        if not self.joint_public_key: raise RuntimeError("Joint public key not generated.")
        encrypted_chunks = []
        for chunk in data_chunks:
            quantized_data = (chunk * self.quantization_scale).round().astype(np.int64).tolist()
            if len(quantized_data) < self.slot_count:
                quantized_data.extend([0] * (self.slot_count - len(quantized_data)))
            plaintext = self.cc.MakePackedPlaintext(quantized_data)
            encrypted_chunks.append(self.cc.Encrypt(self.joint_public_key, plaintext))
        return encrypted_chunks

    def merge_decryption_shares(self, decryption_shares: List[Any]) -> np.ndarray:
        final_plaintext = self.cc.MultipartyDecryptFusion(decryption_shares)
        final_plaintext.SetLength(self.slot_count)
        int_vector = np.array(final_plaintext.GetPackedValue(), dtype=np.int64)

        p = self.cc.GetPlaintextModulus()
        half_p = p // 2
        int_vector[int_vector > half_p] -= p

        float_vector = int_vector.astype(np.float64)
        return float_vector / self.quantization_scale

    def decrypt(self, ciphertext_chunks: List[Any], **kwargs) -> List[np.ndarray]:
        raise NotImplementedError("Single-party decryption is not supported in Threshold mode.")

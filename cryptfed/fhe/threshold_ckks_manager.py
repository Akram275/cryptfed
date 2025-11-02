from .base_fhe_manager import BaseFHEManager
from typing import Any, List, Tuple
import numpy as np
import time
from openfhe import *
import openfhe_numpy as onp
import logging

class ThresholdCKKSManager(BaseFHEManager):
    def __init__(self, security_level: str = 'standard', benchmark_manager: Any = None, threshold: int = 3):
        super().__init__(security_level, benchmark_manager)
        self.cc = None
        self.joint_public_key = None
        self.quantization_scale = 1.0
        self.threshold = threshold

    def generate_crypto_context(self) -> Any:
        params = CCParamsCKKSRNS()
        params.SetMultiplicativeDepth(2)  # Match regular CKKS parameters
        params.SetScalingModSize(50)
        params.SetRingDim(32768)
        self.cc = GenCryptoContext(params)
        self.slot_count = self.cc.GetRingDimension() // 2
        self.cc.Enable(PKESchemeFeature.PKE)
        self.cc.Enable(PKESchemeFeature.KEYSWITCH)
        self.cc.Enable(PKESchemeFeature.LEVELEDSHE)
        self.cc.Enable(PKESchemeFeature.MULTIPARTY)
        logging.getLogger(__name__).info(f"Threshold CKKS context generated. Slot count: {self.slot_count}")
        return self.cc

    def generate_crypto_context_and_keys(self):
        raise NotImplementedError("Single-key generation is not supported in Threshold mode.")

    def collaborative_keygen(self, secret_key_shares: List[Any]):
        if not self.cc:
            raise RuntimeError("CryptoContext not generated.")
        
        # Generate joint public key
        joint_key_pair = self.cc.MultipartyKeyGen(secret_key_shares)
        self.joint_public_key = joint_key_pair.publicKey
        
        # Generate evaluation keys for homomorphic operations.
        # NOTE: OpenFHE's EvalMultKeyGen registers evaluation keys into the
        # CryptoContext and returns None for the per-share call. Calling it
        # for each secret-share is sufficient (mirrors threshold BFV/BGV).
        logging.getLogger(__name__).info("Generating joint evaluation keys for CKKS relinearization...")
        for sk in secret_key_shares:
            # EvalMultKeyGen registers the key in the context for the given share
            # and does not return a usable eval-key object to be fused.
            self.cc.EvalMultKeyGen(sk)

        logging.getLogger(__name__).info("Joint evaluation keys generated successfully for threshold CKKS.")

    def encrypt(self, data_chunks: List[np.ndarray], client_id: str) -> List[Any]:
        if not self.joint_public_key: raise RuntimeError("Joint public key not generated.")
        encrypted_chunks = []
        for chunk in data_chunks:
            encrypted_tensor = onp.array(
                cc=self.cc, data=chunk, public_key=self.joint_public_key, fhe_type="C"
            )
            encrypted_chunks.append(encrypted_tensor)
        return encrypted_chunks

    def merge_decryption_shares(self, decryption_shares: List[Any]) -> np.ndarray:
        final_plaintext = self.cc.MultipartyDecryptFusion(decryption_shares)
        # For CKKS, ensure we're using the correct length (effective slot count)
        # Use effective slot count (slot_count // 2) to match encryption behavior
        effective_length = self.slot_count // 2
        final_plaintext.SetLength(effective_length)
        result = np.array(final_plaintext.GetRealPackedValue(), dtype=np.float64)
        
        # Debug: check for precision issues
        if len(result) != effective_length:
            logging.getLogger(__name__).warning(f"Expected {effective_length} values, got {len(result)}")
        
        return result

    def decrypt(self, ciphertext_chunks: List[Any], length: int = 0) -> np.ndarray:
        raise NotImplementedError("Single-party decryption is not supported in Threshold mode.")

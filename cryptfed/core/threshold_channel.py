from typing import List, Any
import time
import logging
import openfhe_numpy as onp

class ThresholdChannel:
    """
    Simulates a communication channel for orchestrating multi-party
    cryptographic protocols (e.g., threshold key generation and decryption).

    This class removes the "trusted dealer" assumption from the server.
    """
    def __init__(self, cc: Any):
        if cc is None:
            raise ValueError("ThresholdChannel requires a valid CryptoContext.")
        self.cc = cc

    def collaborative_keygen(self, key_holders: List[Any]) -> Any:
        """
        Orchestrates the full, decentralized multi-party key generation ceremony.
        Returns the single joint public key.
        """
        logging.getLogger(__name__).info("--- Channel: Starting Collaborative Key Generation ---")
        if len(key_holders) < 2:
            raise ValueError("Threshold key generation requires at least 2 parties.")

        # --- Joint Public Key Generation ---
        # 1. Lead party generates the first share
        lead_client = key_holders[0]
        joint_public_key = lead_client.generate_public_key_share(self.cc, is_lead=True)

        # 2. Other parties add their shares sequentially
        for client in key_holders[1:]:
            joint_public_key = client.generate_public_key_share(self.cc, joint_public_key)

        # --- Joint Evaluation Key Generation ---
        logging.getLogger(__name__).info("\n--- Channel: Starting Collaborative Evaluation Key Generation ---")
        # 1. Each party generates their eval key share for multiplication
        eval_mult_shares = [client.generate_eval_mult_key_share(self.cc) for client in key_holders]

        # 2. Fuse the shares to create the final joint key
        fused_key = self.cc.MultiAddEvalKeys(eval_mult_shares[0], eval_mult_shares[1])
        for i in range(2, len(eval_mult_shares)):
            fused_key = self.cc.MultiAddEvalKeys(fused_key, eval_mult_shares[i])

        # 3. Insert the final key into the context
        self.cc.InsertEvalMultKey([fused_key])

        logging.getLogger(__name__).info("--- Channel: Collaborative Key Generation Complete ---")
        return joint_public_key

    def collaborative_decryption(self, key_holders: List[Any], ciphertext: Any) -> Any:
        """
        Orchestrates the collaborative decryption of a single ciphertext.
        """
        # 1. Get decryption shares for this chunk from all key holders
        decryption_shares = [client.get_decryption_share(self.cc, ciphertext) for client in key_holders]

        # 2. Merge shares to get the final plaintext
        # For CKKS, the original ciphertext is needed for the fusion
        if "ckks" in str(type(self.cc)).lower():
             decryption_shares.insert(0, ciphertext.data if isinstance(ciphertext, onp.array) else ciphertext)

        final_plaintext = self.cc.MultipartyDecryptFusion(decryption_shares)
        return final_plaintext

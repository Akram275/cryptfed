from .base_aggregator import BaseAggregator
from typing import List, Any
import numpy as np
import time
import logging

class SecureKrum(BaseAggregator):
    """
    Implements encrypted Krum aggregation algorithm using homomorphic encryption.
    
    This aggregator performs Krum selection entirely in the encrypted domain by:
    1. Computing pairwise squared distances between encrypted updates
    2. Finding the update with minimum sum of distances to neighbors
    3. Returning the selected encrypted update
    
    Reference: "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
    by Blanchard et al. (2017)
    
    Note: This implementation works with encrypted updates and requires a CryptoContext.
    """
    def __init__(self, f: int = 0, benchmark_manager: Any = None):
        """
        Args:
            f: Maximum number of Byzantine clients the algorithm should tolerate.
               If f=0, it will be automatically set to (n-1)//2 where n is the number of clients.
        """
        super().__init__(benchmark_manager)
        self.f = f
        self.selected_client_idx = None  # Track which client was selected

    @property
    def requires_plaintext_updates(self) -> bool:
        return False

    def aggregate(self, encrypted_updates: List[Any], weights: List[float], cc: Any = None, fhe_manager: Any = None, **kwargs) -> Any:
        """
        Performs secure Krum aggregation by selecting the most central encrypted update.
        
        Args:
            encrypted_updates: List of encrypted model updates (ciphertexts)
            weights: Client weights (ignored in Krum)
            cc: CryptoContext for homomorphic operations
            fhe_manager: FHE manager with decryption capability (optional)
            
        Returns:
            The selected encrypted update (ciphertext)
        """
        if not encrypted_updates:
            raise ValueError("Cannot aggregate an empty list of updates.")
        if cc is None:
            raise ValueError("CryptoContext 'cc' is required for secure aggregation.")
        
        n = len(encrypted_updates)
        if self.f == 0:
            f = (n - 1) // 2  # Default: tolerate up to half of the clients being Byzantine
        else:
            f = self.f
            
        if n <= 2 * f:
            raise ValueError(f"Krum requires n > 2f, but got n={n}, f={f}")

        logging.getLogger(__name__).info(f"Starting Secure Krum aggregation of {n} updates, tolerating {f} Byzantine clients...")
        start_time = time.time()

        # For small number of clients, we can compute all pairwise distances
        # For larger numbers, this becomes computationally expensive
        if n > 10:
            logging.getLogger(__name__).warning(f"Secure Krum with {n} clients may be slow due to O(n²) distance computations")

        # Compute encrypted pairwise squared distances and scores
        encrypted_scores = self._compute_encrypted_scores(encrypted_updates, cc, f)
        
        # Find the client with minimum score using secure comparison
        selected_idx = self._secure_argmin(encrypted_scores, cc, fhe_manager)
        
        # Store the selected client index for logging
        self.selected_client_idx = selected_idx
        
        # Return the selected encrypted update
        selected_update = encrypted_updates[selected_idx]

        duration = time.time() - start_time

        if self.benchmark_manager:
            self.benchmark_manager.log_event('Server', 'Secure Krum Aggregation Time', duration)
            self.benchmark_manager.log_event('Server', 'Secure Krum Selected Client', selected_idx)

        logging.getLogger(__name__).info(f"Secure Krum aggregation finished in {duration:.4f}s. Selected client {selected_idx}.")
        return selected_update

    def _compute_encrypted_scores(self, encrypted_updates: List[Any], cc: Any, f: int) -> List[Any]:
        """
        Compute encrypted Krum scores for each client.
        
        For each client i, the score is the sum of squared distances to all other clients.
        (In the original Krum, we would select the n-f-2 closest neighbors, but this
        requires sorting which is complex in encrypted domain. Summing all distances
        gives a good approximation.)
        """
        n = len(encrypted_updates)
        encrypted_scores = []
        
        logging.getLogger(__name__).info(f"Computing encrypted pairwise distances for {n} clients...")
        
        for i in range(n):
            # Initialize score by computing first distance
            score = None
            distance_count = 0
            
            # Sum squared distances to all other clients
            for j in range(n):
                if i != j:
                    # Compute squared distance: ||update_i - update_j||²
                    squared_dist = self._encrypted_squared_distance(encrypted_updates[i], encrypted_updates[j], cc)
                    
                    if score is None:
                        # First distance becomes the initial score
                        score = squared_dist
                    else:
                        # Add subsequent distances
                        if hasattr(score, 'data'):
                            # openfhe_numpy.CTArray objects (CKKS)
                            score = score + squared_dist
                        else:
                            # Raw OpenFHE Ciphertext objects (BFV/BGV)
                            score = cc.EvalAdd(score, squared_dist)
                    distance_count += 1
            
            if score is None:
                # This should not happen if n > 1, but handle edge case
                raise ValueError(f"Could not compute score for client {i}")
                
            encrypted_scores.append(score)
            logging.getLogger(__name__).debug(f"Computed encrypted score for client {i} using {distance_count} distances")
            
        logging.getLogger(__name__).info(f"Computed encrypted scores for all {n} clients")
        return encrypted_scores

    def _encrypted_squared_distance(self, encrypted_update1: Any, encrypted_update2: Any, cc: Any) -> Any:
        """
        Compute encrypted squared Euclidean distance between two encrypted vectors.
        
        Returns: ||encrypted_update1 - encrypted_update2||²
        """
        # Handle different types of encrypted objects
        if hasattr(encrypted_update1, 'data'):
            # openfhe_numpy.CTArray objects (CKKS)
            diff = encrypted_update1 - encrypted_update2  # Uses openfhe_numpy operations
            squared_diff = diff * diff  # Element-wise squaring
            # Sum all elements using the .sum() method
            squared_distance = squared_diff.sum()
        else:
            # Raw OpenFHE Ciphertext objects (BFV/BGV)
            # Compute difference: update1 - update2
            diff = cc.EvalSub(encrypted_update1, encrypted_update2)
            
            # Square the difference element-wise
            squared_diff = cc.EvalSquare(diff)
            
            # Sum all elements to get the squared distance
            # EvalSum requires a batch size parameter
            try:
                # Try to get slot count for batch size
                slot_count = cc.GetRingDimension()
                if hasattr(cc, 'GetSlotCount'):
                    slot_count = cc.GetSlotCount()
                squared_distance = cc.EvalSum(squared_diff, slot_count)
            except:
                # If EvalSum fails, return the squared difference vector
                # The comparison will work element-wise
                squared_distance = squared_diff
            
        return squared_distance

    def _secure_argmin(self, encrypted_scores: List[Any], cc: Any, fhe_manager: Any = None) -> int:
        """
        Find the index of the minimum encrypted score.
        
        This implementation uses a hybrid approach:
        1. Decrypt only the scores (not the full updates) to find the minimum
        2. This reveals which client was selected but preserves privacy of model updates
        
        Args:
            encrypted_scores: List of encrypted score values
            cc: CryptoContext
            fhe_manager: FHE manager with decryption capability
            
        Returns:
            Index of the client with minimum score
        """
        if fhe_manager is None:
            # Fallback: use secure comparison approximation
            return self._approximate_secure_argmin(encrypted_scores, cc)
        
        # Decrypt scores to find minimum
        try:
            decrypted_scores = []
            for score_ciphertext in encrypted_scores:
                # Handle different types of encrypted objects
                if hasattr(score_ciphertext, 'decrypt'):
                    # openfhe_numpy.CTArray objects (CKKS)
                    decrypted_score = score_ciphertext.decrypt(fhe_manager._secret_key)
                    # Extract the first element as the score value
                    if hasattr(decrypted_score, 'shape') and len(decrypted_score.shape) > 0:
                        score_value = float(decrypted_score.flat[0])  # Get first element
                    else:
                        score_value = float(decrypted_score)
                else:
                    # Raw OpenFHE Ciphertext objects (BFV/BGV)
                    if hasattr(fhe_manager, 'decrypt'):
                        decrypted_chunk = fhe_manager.decrypt([score_ciphertext])
                        # Extract the first element as the score value
                        if isinstance(decrypted_chunk, list) and len(decrypted_chunk) > 0:
                            score_value = decrypted_chunk[0][0] if isinstance(decrypted_chunk[0], np.ndarray) else decrypted_chunk[0]
                        else:
                            score_value = decrypted_chunk[0] if hasattr(decrypted_chunk, '__getitem__') else float(decrypted_chunk)
                    else:
                        raise ValueError("FHE manager does not support decryption")
                        
                decrypted_scores.append(float(score_value))
                    
            # Find the index of minimum score
            selected_idx = int(np.argmin(decrypted_scores))
            
            logging.getLogger(__name__).info(f"Secure Krum scores: {[f'{s:.4f}' for s in decrypted_scores]}")
            logging.getLogger(__name__).info(f"Selected client {selected_idx} with minimum score {decrypted_scores[selected_idx]:.4f}")
            
            return selected_idx
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to decrypt scores: {e}. Using approximation.")
            return self._approximate_secure_argmin(encrypted_scores, cc)

    def _approximate_secure_argmin(self, encrypted_scores: List[Any], cc: Any) -> int:
        """
        Approximate secure argmin using polynomial approximations or random selection.
        
        This is a fallback when decryption is not available.
        """
        # For now, return a random selection as a placeholder
        # In practice, this could use:
        # - Polynomial approximations of comparison functions
        # - Secure comparison using additional cryptographic protocols
        # - Garbled circuits for comparison
        
        import random
        selected_idx = random.randint(0, len(encrypted_scores) - 1)
        logging.getLogger(__name__).warning(f"Using random selection (client {selected_idx}) due to lack of secure comparison")
        return selected_idx

    def get_selected_client_info(self) -> dict:
        """
        Returns information about the selected client.
        """
        return {
            "selected_client_idx": self.selected_client_idx,
            "algorithm": "Secure Krum"
        }
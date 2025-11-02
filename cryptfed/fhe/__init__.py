from .ckks_manager import CKKSManager
from .bfv_manager import BFVManager
from .bgv_manager import BGVManager
from .threshold_bfv_manager import ThresholdBFVManager
from .threshold_bgv_manager import ThresholdBGVManager # <-- Import new
from .threshold_ckks_manager import ThresholdCKKSManager # <-- Import new

# The FHE Manager Registry
fhe_manager_registry = {
    # Single-Key Schemes
    "ckks": CKKSManager,
    "bfv": BFVManager,
    "bgv": BGVManager,

    # Threshold (Multi-Key) Schemes
    "threshold_bfv": ThresholdBFVManager,
    "threshold_bgv": ThresholdBGVManager, # <-- Add to registry
    "threshold_ckks": ThresholdCKKSManager, # <-- Add to registry
}

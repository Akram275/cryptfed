# Import plaintext aggregators (always available)
from .fed_trimmed_mean import FedTrimmedMean
from .plaintext_fedavg import PlaintextFedAvg
from .fed_median import FedMedian
from .krum import Krum
from .multi_krum import MultiKrum
from .fed_prox import FedProx
from .flame_byzantine import FlameByzantine
from .fools_gold import FoolsGold

# Import modular aggregation framework
from .modular_aggregator import (
    ModularAggregator,
    GraphBasedAggregator,
    CustomAggregatorBuilder,
    create_fedavg_aggregator,
    create_fairness_aggregator
)

# Try to import FHE aggregators (only available if OpenFHE is installed)
FHE_AVAILABLE = True
try:
    from .fed_avg_momentum import FedAvgMomentum
    from .secure_ckks_fedavg import SecureCkksFedAvg
    from .secure_integer_fedavg import SecureIntegerFedAvg
    from .secure_trimmed_mean import SecureTrimmedMean
    from .secure_krum import SecureKrum
except ImportError as e:
    print(f"Warning: FHE aggregators not available. OpenFHE not properly installed.")
    print("Only plaintext federated learning will be available.")
    FedAvgMomentum = None
    SecureCkksFedAvg = None
    SecureIntegerFedAvg = None
    SecureTrimmedMean = None
    SecureKrum = None
    FHE_AVAILABLE = False

# The Aggregator Registry
# Maps a user-friendly string name to the aggregator class.
aggregator_registry = {
    # Plaintext-Only Aggregators (always available)
    "plaintext_fedavg": PlaintextFedAvg,
    "trimmed_mean": FedTrimmedMean,
    "median": FedMedian,
    "krum": Krum,
    "multi_krum": MultiKrum,
    "flame": FlameByzantine,
    "fools_gold": FoolsGold,
    "fed_prox": FedProx,
}

# Add FHE aggregators if available
if FHE_AVAILABLE:
    aggregator_registry.update({
        "ckks_fedavg": SecureCkksFedAvg,
        "integer_fedavg": SecureIntegerFedAvg,
        "fedavgm": FedAvgMomentum,  # Note: This currently only works with CKKS
        "secure_trimmed_mean": SecureTrimmedMean,
        "secure_krum": SecureKrum,
    })

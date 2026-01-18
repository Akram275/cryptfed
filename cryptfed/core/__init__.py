"""
Core components of CrypTFed.
"""

from .federated_client import FederatedClient
from .federated_server import FederatedServer
from .protocol_state import (
    ProtocolCoordinator,
    OrchestratorState,
    ServerState,
    ClientState,
    StateTransitionError,
    ProtocolViolationError
)
from .payload import (
    ClientPayload,
    PayloadItem,
    PayloadItemType,
    PayloadBuilder,
    create_legacy_payload,
    extract_legacy_updates
)
from .fhe_graph import (
    FHEComputationGraph,
    FHEOperation,
    FHEGraphBuilder,
    CommonGraphPatterns
)

__all__ = [
    'FederatedClient',
    'FederatedServer',
    'ProtocolCoordinator',
    'OrchestratorState',
    'ServerState',
    'ClientState',
    'StateTransitionError',
    'ProtocolViolationError',
    'ClientPayload',
    'PayloadItem',
    'PayloadItemType',
    'PayloadBuilder',
    'create_legacy_payload',
    'extract_legacy_updates',
    'FHEComputationGraph',
    'FHEOperation',
    'FHEGraphBuilder',
    'CommonGraphPatterns',
]

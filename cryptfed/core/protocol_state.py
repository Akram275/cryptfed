"""
Protocol State Machine for CrypTFed Federated Learning
======================================================

This module implements a formal state machine for the federated learning protocol,
ensuring that all entities (clients, server, orchestrator) perform the correct
actions during the appropriate protocol phases.

The state machine enforces:
- Proper initialization sequences
- Correct operation ordering
- Validation of state transitions
- Academic rigor in protocol execution
"""

from enum import Enum, auto
from typing import Optional, Set, Dict, Any
import logging
from dataclasses import dataclass, field
from datetime import datetime


class OrchestratorState(Enum):
    """States for the main orchestrator/coordinator"""
    UNINITIALIZED = auto()           # Initial state before any setup
    INITIALIZING = auto()            # Setting up the federated system
    SESSION_READY = auto()           # Session initialized, ready to start rounds
    ROUND_STARTING = auto()          # Beginning a new training round
    DISTRIBUTING_MODEL = auto()      # Sending global model to clients
    WAITING_FOR_CLIENTS = auto()     # Waiting for client updates
    COLLECTING_UPDATES = auto()      # Receiving encrypted updates from clients
    AGGREGATING = auto()             # Server aggregating updates
    DECRYPTING_MODEL = auto()        # Decrypting aggregated model (if FHE)
    EVALUATING = auto()              # Evaluating model performance
    ROUND_COMPLETE = auto()          # Round finished successfully
    TRAINING_COMPLETE = auto()       # All rounds finished
    ERROR = auto()                   # Error state
    TERMINATED = auto()              # Final state after cleanup


class ServerState(Enum):
    """States for the federated server"""
    UNINITIALIZED = auto()           # Initial state
    INITIALIZING_CRYPTO = auto()     # Setting up cryptographic context
    CRYPTO_READY = auto()            # Crypto setup complete
    MODEL_INITIALIZED = auto()       # Global model encrypted/initialized
    IDLE = auto()                    # Ready to process requests
    BROADCASTING_MODEL = auto()      # Sending model to clients
    RECEIVING_UPDATES = auto()       # Collecting client updates
    AGGREGATING_UPDATES = auto()     # Performing (secure) aggregation
    MODEL_UPDATED = auto()           # New global model ready
    BENCHMARKING = auto()            # Recording metrics
    ERROR = auto()                   # Error state


class ClientState(Enum):
    """States for federated clients"""
    UNINITIALIZED = auto()           # Initial state
    CONNECTING = auto()              # Connecting to server
    CONNECTED = auto()               # Connected and ready
    IDLE = auto()                    # Waiting for next round
    RECEIVING_MODEL = auto()         # Downloading global model
    DECRYPTING_MODEL = auto()        # Decrypting received model (if FHE)
    TRAINING = auto()                # Performing local training
    EVALUATING_LOCAL = auto()        # Local validation
    ENCRYPTING_UPDATE = auto()       # Encrypting model update (if FHE)
    SENDING_UPDATE = auto()          # Uploading update to server
    WAITING = auto()                 # Waiting for next round
    COLLABORATIVE_KEYGEN = auto()    # Threshold key generation (threshold only)
    COLLABORATIVE_DECRYPT = auto()   # Threshold decryption (threshold only)
    ERROR = auto()                   # Error state
    DISCONNECTED = auto()            # Disconnected from server


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted"""
    pass


class ProtocolViolationError(Exception):
    """Raised when protocol rules are violated"""
    pass


@dataclass
class StateTransition:
    """Records a state transition for audit purposes"""
    entity_id: str
    entity_type: str  # 'orchestrator', 'server', or 'client'
    from_state: Enum
    to_state: Enum
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class StateMachine:
    """Base state machine with transition validation and logging"""
    
    def __init__(self, entity_id: str, entity_type: str, initial_state: Enum):
        self.entity_id = entity_id
        self.entity_type = entity_type
        self._current_state = initial_state
        self._transition_history: list[StateTransition] = []
        self.logger = logging.getLogger(f"{__name__}.{entity_type}.{entity_id}")
        
        # Define allowed transitions (to be overridden by subclasses)
        self._allowed_transitions: Dict[Enum, Set[Enum]] = {}
        
    @property
    def current_state(self) -> Enum:
        """Get the current state"""
        return self._current_state
    
    def can_transition_to(self, new_state: Enum) -> bool:
        """Check if transition to new_state is allowed"""
        if self._current_state not in self._allowed_transitions:
            return False
        return new_state in self._allowed_transitions[self._current_state]
    
    def transition_to(self, new_state: Enum, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Transition to a new state with validation.
        
        Args:
            new_state: Target state
            metadata: Optional metadata about the transition
            
        Raises:
            StateTransitionError: If transition is not allowed
        """
        if not self.can_transition_to(new_state):
            error_msg = (
                f"Invalid state transition for {self.entity_type} '{self.entity_id}': "
                f"{self._current_state.name} -> {new_state.name}"
            )
            self.logger.error(error_msg)
            raise StateTransitionError(error_msg)
        
        old_state = self._current_state
        self._current_state = new_state
        
        # Record transition
        transition = StateTransition(
            entity_id=self.entity_id,
            entity_type=self.entity_type,
            from_state=old_state,
            to_state=new_state,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self._transition_history.append(transition)
        
        self.logger.info(
            f"State transition: {old_state.name} -> {new_state.name} "
            f"{f'({metadata})' if metadata else ''}"
        )
    
    def require_state(self, *required_states: Enum) -> None:
        """
        Validate that the current state is one of the required states.
        
        Args:
            *required_states: One or more required states
            
        Raises:
            ProtocolViolationError: If current state is not in required states
        """
        if self._current_state not in required_states:
            error_msg = (
                f"Protocol violation for {self.entity_type} '{self.entity_id}': "
                f"Expected state in {[s.name for s in required_states]}, "
                f"but current state is {self._current_state.name}"
            )
            self.logger.error(error_msg)
            raise ProtocolViolationError(error_msg)
    
    def get_transition_history(self) -> list[StateTransition]:
        """Get the complete transition history"""
        return self._transition_history.copy()
    
    def reset(self, initial_state: Optional[Enum] = None) -> None:
        """Reset the state machine to initial state"""
        if initial_state:
            self._current_state = initial_state
        self._transition_history.clear()
        self.logger.info(f"State machine reset to {self._current_state.name}")


class OrchestratorStateMachine(StateMachine):
    """State machine for the main orchestrator"""
    
    def __init__(self, orchestrator_id: str = "orchestrator"):
        super().__init__(orchestrator_id, "orchestrator", OrchestratorState.UNINITIALIZED)
        
        # Define allowed state transitions
        self._allowed_transitions = {
            OrchestratorState.UNINITIALIZED: {
                OrchestratorState.INITIALIZING,
                OrchestratorState.ERROR
            },
            OrchestratorState.INITIALIZING: {
                OrchestratorState.SESSION_READY,
                OrchestratorState.ERROR
            },
            OrchestratorState.SESSION_READY: {
                OrchestratorState.ROUND_STARTING,
                OrchestratorState.ERROR,
                OrchestratorState.TERMINATED
            },
            OrchestratorState.ROUND_STARTING: {
                OrchestratorState.DISTRIBUTING_MODEL,
                OrchestratorState.ERROR
            },
            OrchestratorState.DISTRIBUTING_MODEL: {
                OrchestratorState.WAITING_FOR_CLIENTS,
                OrchestratorState.ERROR
            },
            OrchestratorState.WAITING_FOR_CLIENTS: {
                OrchestratorState.COLLECTING_UPDATES,
                OrchestratorState.ERROR
            },
            OrchestratorState.COLLECTING_UPDATES: {
                OrchestratorState.AGGREGATING,
                OrchestratorState.ERROR
            },
            OrchestratorState.AGGREGATING: {
                OrchestratorState.DECRYPTING_MODEL,
                OrchestratorState.EVALUATING,  # Skip decryption for plaintext
                OrchestratorState.ERROR
            },
            OrchestratorState.DECRYPTING_MODEL: {
                OrchestratorState.EVALUATING,
                OrchestratorState.ERROR
            },
            OrchestratorState.EVALUATING: {
                OrchestratorState.ROUND_COMPLETE,
                OrchestratorState.ERROR
            },
            OrchestratorState.ROUND_COMPLETE: {
                OrchestratorState.ROUND_STARTING,  # Next round
                OrchestratorState.TRAINING_COMPLETE,  # All rounds done
                OrchestratorState.ERROR
            },
            OrchestratorState.TRAINING_COMPLETE: {
                OrchestratorState.TERMINATED
            },
            OrchestratorState.ERROR: {
                OrchestratorState.TERMINATED,
                OrchestratorState.UNINITIALIZED  # Allow reset
            },
            OrchestratorState.TERMINATED: set()  # Terminal state
        }


class ServerStateMachine(StateMachine):
    """State machine for the federated server"""
    
    def __init__(self, server_id: str = "server"):
        super().__init__(server_id, "server", ServerState.UNINITIALIZED)
        
        # Define allowed state transitions
        self._allowed_transitions = {
            ServerState.UNINITIALIZED: {
                ServerState.INITIALIZING_CRYPTO,
                ServerState.IDLE,  # Plaintext mode bypasses crypto
                ServerState.ERROR
            },
            ServerState.INITIALIZING_CRYPTO: {
                ServerState.CRYPTO_READY,
                ServerState.ERROR
            },
            ServerState.CRYPTO_READY: {
                ServerState.MODEL_INITIALIZED,
                ServerState.ERROR
            },
            ServerState.MODEL_INITIALIZED: {
                ServerState.IDLE,
                ServerState.ERROR
            },
            ServerState.IDLE: {
                ServerState.BROADCASTING_MODEL,
                ServerState.RECEIVING_UPDATES,
                ServerState.BENCHMARKING,
                ServerState.ERROR
            },
            ServerState.BROADCASTING_MODEL: {
                ServerState.RECEIVING_UPDATES,
                ServerState.IDLE,
                ServerState.ERROR
            },
            ServerState.RECEIVING_UPDATES: {
                ServerState.AGGREGATING_UPDATES,
                ServerState.ERROR
            },
            ServerState.AGGREGATING_UPDATES: {
                ServerState.MODEL_UPDATED,
                ServerState.ERROR
            },
            ServerState.MODEL_UPDATED: {
                ServerState.IDLE,
                ServerState.BENCHMARKING,
                ServerState.ERROR
            },
            ServerState.BENCHMARKING: {
                ServerState.IDLE,
                ServerState.ERROR
            },
            ServerState.ERROR: {
                ServerState.UNINITIALIZED  # Allow reset
            }
        }


class ClientStateMachine(StateMachine):
    """State machine for federated clients"""
    
    def __init__(self, client_id: str):
        super().__init__(client_id, "client", ClientState.UNINITIALIZED)
        
        # Define allowed state transitions
        self._allowed_transitions = {
            ClientState.UNINITIALIZED: {
                ClientState.CONNECTING,
                ClientState.ERROR
            },
            ClientState.CONNECTING: {
                ClientState.CONNECTED,
                ClientState.COLLABORATIVE_KEYGEN,  # Threshold mode
                ClientState.ERROR
            },
            ClientState.COLLABORATIVE_KEYGEN: {
                ClientState.CONNECTED,
                ClientState.ERROR
            },
            ClientState.CONNECTED: {
                ClientState.IDLE,
                ClientState.ERROR
            },
            ClientState.IDLE: {
                ClientState.RECEIVING_MODEL,
                ClientState.COLLABORATIVE_DECRYPT,  # Threshold decryption
                ClientState.DISCONNECTED,
                ClientState.ERROR
            },
            ClientState.RECEIVING_MODEL: {
                ClientState.DECRYPTING_MODEL,
                ClientState.TRAINING,  # Plaintext mode
                ClientState.ERROR
            },
            ClientState.DECRYPTING_MODEL: {
                ClientState.TRAINING,
                ClientState.ERROR
            },
            ClientState.TRAINING: {
                ClientState.EVALUATING_LOCAL,
                ClientState.ENCRYPTING_UPDATE,
                ClientState.SENDING_UPDATE,  # Plaintext mode
                ClientState.ERROR
            },
            ClientState.EVALUATING_LOCAL: {
                ClientState.ENCRYPTING_UPDATE,
                ClientState.SENDING_UPDATE,  # Plaintext mode
                ClientState.ERROR
            },
            ClientState.ENCRYPTING_UPDATE: {
                ClientState.SENDING_UPDATE,
                ClientState.ERROR
            },
            ClientState.SENDING_UPDATE: {
                ClientState.WAITING,
                ClientState.ERROR
            },
            ClientState.WAITING: {
                ClientState.IDLE,
                ClientState.RECEIVING_MODEL,  # Next round
                ClientState.COLLABORATIVE_DECRYPT,  # Threshold decryption
                ClientState.DISCONNECTED,
                ClientState.ERROR
            },
            ClientState.COLLABORATIVE_DECRYPT: {
                ClientState.IDLE,
                ClientState.WAITING,
                ClientState.ERROR
            },
            ClientState.ERROR: {
                ClientState.DISCONNECTED,
                ClientState.UNINITIALIZED  # Allow reset
            },
            ClientState.DISCONNECTED: set()  # Terminal state
        }


class ProtocolCoordinator:
    """
    Coordinates protocol execution across all state machines.
    Ensures global protocol consistency and provides audit capabilities.
    """
    
    def __init__(self):
        self.orchestrator_sm: Optional[OrchestratorStateMachine] = None
        self.server_sm: Optional[ServerStateMachine] = None
        self.client_sms: Dict[str, ClientStateMachine] = {}
        self.logger = logging.getLogger(f"{__name__}.ProtocolCoordinator")
        
    def register_orchestrator(self, orchestrator_id: str = "orchestrator") -> OrchestratorStateMachine:
        """Register the orchestrator state machine"""
        self.orchestrator_sm = OrchestratorStateMachine(orchestrator_id)
        self.logger.info(f"Registered orchestrator: {orchestrator_id}")
        return self.orchestrator_sm
    
    def register_server(self, server_id: str = "server") -> ServerStateMachine:
        """Register the server state machine"""
        self.server_sm = ServerStateMachine(server_id)
        self.logger.info(f"Registered server: {server_id}")
        return self.server_sm
    
    def register_client(self, client_id: str) -> ClientStateMachine:
        """Register a client state machine"""
        client_sm = ClientStateMachine(client_id)
        self.client_sms[client_id] = client_sm
        self.logger.info(f"Registered client: {client_id}")
        return client_sm
    
    def get_global_state_summary(self) -> Dict[str, Any]:
        """Get a summary of all entity states"""
        summary = {
            "orchestrator": self.orchestrator_sm.current_state.name if self.orchestrator_sm else None,
            "server": self.server_sm.current_state.name if self.server_sm else None,
            "clients": {
                client_id: sm.current_state.name
                for client_id, sm in self.client_sms.items()
            }
        }
        return summary
    
    def validate_protocol_phase(self, phase: str) -> bool:
        """
        Validate that all entities are in appropriate states for a protocol phase.
        
        Args:
            phase: Protocol phase name (e.g., 'initialization', 'training', 'aggregation')
            
        Returns:
            bool: True if all entities are in valid states for this phase
        """
        phase_requirements = {
            "initialization": {
                "orchestrator": [OrchestratorState.INITIALIZING],
                "server": [ServerState.UNINITIALIZED, ServerState.INITIALIZING_CRYPTO],
                "clients": [ClientState.UNINITIALIZED, ClientState.CONNECTING]
            },
            "model_distribution": {
                "orchestrator": [OrchestratorState.DISTRIBUTING_MODEL],
                "server": [ServerState.BROADCASTING_MODEL],
                "clients": [ClientState.RECEIVING_MODEL, ClientState.DECRYPTING_MODEL]
            },
            "local_training": {
                "orchestrator": [OrchestratorState.WAITING_FOR_CLIENTS],
                "server": [ServerState.IDLE, ServerState.RECEIVING_UPDATES],
                "clients": [ClientState.TRAINING, ClientState.EVALUATING_LOCAL]
            },
            "aggregation": {
                "orchestrator": [OrchestratorState.AGGREGATING],
                "server": [ServerState.AGGREGATING_UPDATES],
                "clients": [ClientState.WAITING, ClientState.IDLE]
            }
        }
        
        if phase not in phase_requirements:
            self.logger.warning(f"Unknown protocol phase: {phase}")
            return False
        
        requirements = phase_requirements[phase]
        
        # Check orchestrator
        if self.orchestrator_sm and self.orchestrator_sm.current_state not in requirements.get("orchestrator", []):
            self.logger.warning(
                f"Orchestrator in invalid state for {phase}: {self.orchestrator_sm.current_state.name}"
            )
            return False
        
        # Check server
        if self.server_sm and self.server_sm.current_state not in requirements.get("server", []):
            self.logger.warning(
                f"Server in invalid state for {phase}: {self.server_sm.current_state.name}"
            )
            return False
        
        # Check all clients
        for client_id, client_sm in self.client_sms.items():
            if client_sm.current_state not in requirements.get("clients", []):
                self.logger.warning(
                    f"Client {client_id} in invalid state for {phase}: {client_sm.current_state.name}"
                )
                return False
        
        return True
    
    def generate_audit_report(self) -> Dict[str, Any]:
        """Generate a comprehensive audit report of all state transitions"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "entities": {}
        }
        
        if self.orchestrator_sm:
            report["entities"]["orchestrator"] = {
                "current_state": self.orchestrator_sm.current_state.name,
                "transition_history": [
                    {
                        "from": t.from_state.name,
                        "to": t.to_state.name,
                        "timestamp": t.timestamp.isoformat(),
                        "metadata": t.metadata
                    }
                    for t in self.orchestrator_sm.get_transition_history()
                ]
            }
        
        if self.server_sm:
            report["entities"]["server"] = {
                "current_state": self.server_sm.current_state.name,
                "transition_history": [
                    {
                        "from": t.from_state.name,
                        "to": t.to_state.name,
                        "timestamp": t.timestamp.isoformat(),
                        "metadata": t.metadata
                    }
                    for t in self.server_sm.get_transition_history()
                ]
            }
        
        report["entities"]["clients"] = {}
        for client_id, client_sm in self.client_sms.items():
            report["entities"]["clients"][client_id] = {
                "current_state": client_sm.current_state.name,
                "transition_history": [
                    {
                        "from": t.from_state.name,
                        "to": t.to_state.name,
                        "timestamp": t.timestamp.isoformat(),
                        "metadata": t.metadata
                    }
                    for t in client_sm.get_transition_history()
                ]
            }
        
        return report

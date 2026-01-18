"""
Modular payload system for flexible client-to-server communication.

This module enables clients to send heterogeneous data (encrypted or plaintext)
and allows custom aggregation logic with FHE operation validation.
"""

from typing import Any, Dict, List, Union, Optional
from dataclasses import dataclass, field
import numpy as np
from enum import Enum


class PayloadItemType(Enum):
    """Type of payload item for semantic understanding"""
    MODEL_UPDATE = "model_update"
    GRADIENT = "gradient"
    STATISTIC = "statistic"
    FAIRNESS_METRIC = "fairness_metric"
    LOSS = "loss"
    ACCURACY = "accuracy"
    CUSTOM = "custom"


@dataclass
class PayloadItem:
    """
    A single item in a client payload.
    
    Attributes:
        name: Unique identifier for this item (e.g., 'model_weights', 'local_loss')
        data: The actual data (numpy array, encrypted ciphertext, or scalar)
        is_encrypted: Whether the data is encrypted
        item_type: Semantic type of the item
        metadata: Additional metadata (e.g., shape, dtype)
    """
    name: str
    data: Any
    is_encrypted: bool
    item_type: PayloadItemType = PayloadItemType.CUSTOM
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        encrypted_str = "encrypted" if self.is_encrypted else "plaintext"
        return f"PayloadItem(name='{self.name}', type={self.item_type.value}, {encrypted_str})"


@dataclass
class ClientPayload:
    """
    Complete payload sent from a client to the server.
    
    Attributes:
        client_id: Identifier of the sending client
        items: Dictionary mapping item names to PayloadItem objects
        weight: Client's contribution weight (e.g., dataset size)
    """
    client_id: str
    items: Dict[str, PayloadItem] = field(default_factory=dict)
    weight: float = 1.0
    
    def add_item(self, name: str, data: Any, is_encrypted: bool, 
                 item_type: PayloadItemType = PayloadItemType.CUSTOM,
                 metadata: Optional[Dict] = None):
        """Add an item to the payload"""
        if metadata is None:
            metadata = {}
        
        self.items[name] = PayloadItem(
            name=name,
            data=data,
            is_encrypted=is_encrypted,
            item_type=item_type,
            metadata=metadata
        )
    
    def get_item(self, name: str) -> Optional[PayloadItem]:
        """Retrieve an item by name"""
        return self.items.get(name)
    
    def get_encrypted_items(self) -> Dict[str, PayloadItem]:
        """Get all encrypted items"""
        return {k: v for k, v in self.items.items() if v.is_encrypted}
    
    def get_plaintext_items(self) -> Dict[str, PayloadItem]:
        """Get all plaintext items"""
        return {k: v for k, v in self.items.items() if not v.is_encrypted}
    
    def __repr__(self):
        encrypted_count = len(self.get_encrypted_items())
        plaintext_count = len(self.get_plaintext_items())
        return (f"ClientPayload(client='{self.client_id}', "
                f"items={len(self.items)} [{encrypted_count} encrypted, {plaintext_count} plaintext])")


class PayloadBuilder:
    """
    Helper class to construct client payloads.
    
    Example:
        builder = PayloadBuilder(client_id="client_0")
        builder.add_model_update(weights, encrypted=True)
        builder.add_statistic("local_loss", loss_value, encrypted=False)
        payload = builder.build()
    """
    
    def __init__(self, client_id: str, weight: float = 1.0):
        self.client_id = client_id
        self.weight = weight
        self.payload = ClientPayload(client_id=client_id, weight=weight)
    
    def add_model_update(self, data: Any, encrypted: bool = True, 
                        metadata: Optional[Dict] = None) -> 'PayloadBuilder':
        """Add model update (weights or gradients)"""
        self.payload.add_item(
            name="model_update",
            data=data,
            is_encrypted=encrypted,
            item_type=PayloadItemType.MODEL_UPDATE,
            metadata=metadata or {}
        )
        return self
    
    def add_statistic(self, name: str, value: Any, encrypted: bool = False,
                     metadata: Optional[Dict] = None) -> 'PayloadBuilder':
        """Add a statistical value (e.g., loss, accuracy, fairness metric)"""
        self.payload.add_item(
            name=name,
            data=value,
            is_encrypted=encrypted,
            item_type=PayloadItemType.STATISTIC,
            metadata=metadata or {}
        )
        return self
    
    def add_fairness_metric(self, name: str, value: Any, encrypted: bool = False,
                           metadata: Optional[Dict] = None) -> 'PayloadBuilder':
        """Add fairness-related metric"""
        self.payload.add_item(
            name=name,
            data=value,
            is_encrypted=encrypted,
            item_type=PayloadItemType.FAIRNESS_METRIC,
            metadata=metadata or {}
        )
        return self
    
    def add_custom(self, name: str, data: Any, encrypted: bool = False,
                   item_type: PayloadItemType = PayloadItemType.CUSTOM,
                   metadata: Optional[Dict] = None) -> 'PayloadBuilder':
        """Add custom item"""
        self.payload.add_item(
            name=name,
            data=data,
            is_encrypted=encrypted,
            item_type=item_type,
            metadata=metadata or {}
        )
        return self
    
    def build(self) -> ClientPayload:
        """Return the constructed payload"""
        return self.payload


# Backward compatibility: Simple payload conversion
def create_legacy_payload(client_id: str, model_update: Any, 
                         is_encrypted: bool, weight: float = 1.0) -> ClientPayload:
    """
    Create a simple payload compatible with legacy aggregators.
    
    This function ensures backward compatibility with existing code that
    expects a single model update per client.
    """
    builder = PayloadBuilder(client_id, weight)
    builder.add_model_update(model_update, encrypted=is_encrypted)
    return builder.build()


def extract_legacy_updates(payloads: List[ClientPayload]) -> tuple:
    """
    Extract updates in legacy format (List[updates], List[weights]).
    
    Used to maintain compatibility with existing aggregators.
    """
    updates = []
    weights = []
    
    for payload in payloads:
        model_item = payload.get_item("model_update")
        if model_item is None:
            raise ValueError(f"Payload from {payload.client_id} missing 'model_update'")
        updates.append(model_item.data)
        weights.append(payload.weight)
    
    return updates, weights

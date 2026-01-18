"""
Example 1: Basic Client Payloads

This example demonstrates:
- Creating client payloads with mixed encrypted/plaintext data
- Adding model updates, statistics, and fairness metrics
- Inspecting payload contents
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from cryptfed.core.payload import (
    ClientPayload,
    PayloadBuilder,
    PayloadItemType
)


def main():
    """Example 1: Creating and inspecting client payloads"""
    print("\n" + "="*70)
    print("Example 1: Basic Client Payloads".center(70))
    print("="*70 + "\n")
    
    # Create a payload with mixed data
    builder = PayloadBuilder(client_id="client_0", weight=100)
    
    # Add encrypted model update
    model_weights = np.random.randn(10)
    builder.add_model_update(model_weights, encrypted=True)
    
    # Add plaintext statistics
    builder.add_statistic("local_loss", 0.234, encrypted=False)
    builder.add_statistic("local_accuracy", 0.891, encrypted=False)
    
    # Add fairness metric
    builder.add_fairness_metric("demographic_parity", 0.95, encrypted=False)
    
    payload = builder.build()
    
    print(f"Created payload: {payload}")
    print(f"  - Encrypted items: {len(payload.get_encrypted_items())}")
    print(f"  - Plaintext items: {len(payload.get_plaintext_items())}")
    print(f"  - Client weight: {payload.weight}")
    
    print("\nPayload contents:")
    for name, item in payload.items.items():
        status = "encrypted" if item.is_encrypted else "plaintext"
        print(f"  - {name}: {item.item_type.value} ({status})")
    
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)
    
    return payload


if __name__ == "__main__":
    main()

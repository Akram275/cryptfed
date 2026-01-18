"""
Example 6: Common FHE Graph Patterns

This example demonstrates:
- Using pre-built patterns for common aggregation strategies
- FedAvg pattern for standard federated averaging
- Fairness-weighted pattern for fairness-aware aggregation
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from cryptfed.core.fhe_graph import CommonGraphPatterns


def main():
    """Example 6: Using common graph patterns"""
    print("\n" + "="*70)
    print("Example 6: Common FHE Graph Patterns".center(70))
    print("="*70 + "\n")
    
    print("Pre-built patterns for common aggregation strategies:\n")
    
    num_clients = 10
    
    # FedAvg pattern
    fedavg_graph = CommonGraphPatterns.fedavg_pattern(num_clients)
    valid, _ = fedavg_graph.validate()
    print(f"1. FedAvg Pattern (n={num_clients})")
    print(f"   Valid: {valid}")
    if valid:
        print(f"   Depth: {fedavg_graph.estimate_depth()}")
        print(f"   Nodes: {len(fedavg_graph.nodes)}")
    
    # Fairness-weighted pattern
    fairness_graph = CommonGraphPatterns.fairness_weighted_pattern(num_clients)
    valid, _ = fairness_graph.validate()
    print(f"\n2. Fairness-Weighted Pattern (n={num_clients})")
    print(f"   Valid: {valid}")
    if valid:
        print(f"   Depth: {fairness_graph.estimate_depth()}")
        print(f"   Nodes: {len(fairness_graph.nodes)}")
    
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()

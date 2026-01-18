"""
Example 4: Fairness-Weighted Aggregation

This example demonstrates:
- Using fairness metrics to compute aggregation weights
- Building FHE graphs for fairness-aware aggregation
- Normalizing fairness scores to aggregation weights
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from cryptfed.core.fhe_graph import FHEGraphBuilder


def main():
    """Example 4: Fairness-weighted aggregation"""
    print("\n" + "="*70)
    print("Example 4: Fairness-Weighted Aggregation".center(70))
    print("="*70 + "\n")
    
    print("Scenario: Clients send model updates + fairness metrics")
    print("Server computes aggregation weights based on fairness\n")
    
    # Simulate 5 clients with different fairness scores
    num_clients = 5
    fairness_scores = [0.95, 0.82, 0.91, 0.78, 0.88]
    
    print(f"Client fairness scores:")
    for i, score in enumerate(fairness_scores):
        print(f"  Client {i}: {score:.2f}")
    
    # Build FHE graph for fairness-weighted aggregation
    builder = FHEGraphBuilder("fairness_weighted")
    
    # Normalize fairness scores to weights
    total_fairness = sum(fairness_scores)
    weights = [score / total_fairness for score in fairness_scores]
    
    print(f"\nNormalized weights:")
    for i, weight in enumerate(weights):
        print(f"  Client {i}: {weight:.3f}")
    
    # Build the weighted sum graph
    update_vars = [f"update_{i}" for i in range(num_clients)]
    builder.weighted_sum(update_vars, weights, "fair_aggregate")
    
    graph = builder.build()
    
    print(f"\nGraph validation:")
    valid, errors = graph.validate()
    print(f"  Valid: {valid}")
    if valid:
        depth = graph.estimate_depth()
        print(f"  Multiplicative depth: {depth}")
        print(f"  Number of operations: {len(graph.nodes)}")
    
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()

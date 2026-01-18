"""
Example 3: Multiplicative Depth Estimation

This example demonstrates:
- Creating graphs with different complexity levels
- Estimating multiplicative depth for each graph
- Understanding depth implications for FHE parameters
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from cryptfed.core.fhe_graph import FHEGraphBuilder


def main():
    """Example 3: Depth estimation for complex graphs"""
    print("\n" + "="*70)
    print("Example 3: Multiplicative Depth Estimation".center(70))
    print("="*70 + "\n")
    
    # Create graphs with different depths
    graphs = {
        "Simple addition": FHEGraphBuilder("simple").add("a", "b", "result").build(),
        "Weighted sum": FHEGraphBuilder("weighted").weighted_sum(["a", "b"], [0.5, 0.5], "result").build(),
        "Multiplication": FHEGraphBuilder("mult").multiply("a", "b", "result").build(),
        "Nested": FHEGraphBuilder("nested")
            .multiply("a", "b", "temp1")
            .multiply("temp1", "c", "result")
            .build()
    }
    
    print("Comparing multiplicative depths:\n")
    for name, graph in graphs.items():
        valid, _ = graph.validate()
        if valid:
            depth = graph.estimate_depth()
            print(f"  {name:20s} - Depth: {depth}")
            if depth > 5:
                print(f"    Warning: High depth may require large FHE parameters")
    
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()

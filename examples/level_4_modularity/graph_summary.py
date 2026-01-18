"""
Example 7: FHE Graph Summaries

This example demonstrates:
- Creating complex FHE computation graphs
- Getting detailed summaries of graph structure
- Inspecting graph properties (depth, nodes, validity)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from cryptfed.core.fhe_graph import FHEGraphBuilder


def main():
    """Example 7: Getting graph summaries"""
    print("\n" + "="*70)
    print("Example 7: FHE Graph Summaries".center(70))
    print("="*70 + "\n")
    
    # Create a complex graph
    builder = FHEGraphBuilder("complex_aggregation")
    builder.weighted_sum(["u0", "u1", "u2"], [0.3, 0.4, 0.3], "avg1")
    builder.weighted_sum(["u3", "u4"], [0.5, 0.5], "avg2")
    builder.add("avg1", "avg2", "combined")
    builder.scalar_multiply("combined", 0.5, "final")
    
    graph = builder.build()
    summary = graph.get_summary()
    
    print("Graph Summary:")
    for key, value in summary.items():
        if key != "errors":
            print(f"  {key}: {value}")
    
    if summary["errors"]:
        print(f"  Errors:")
        for error in summary["errors"]:
            print(f"    - {error}")
    
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()

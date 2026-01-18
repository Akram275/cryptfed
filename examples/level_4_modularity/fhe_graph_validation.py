"""
Example 2: FHE Graph Validation

This example demonstrates:
- Creating valid FHE computation graphs
- Detecting unsupported operations (division, max, min)
- Validating graphs before execution
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from cryptfed.core.fhe_graph import (
    FHEGraphBuilder,
    FHEOperation
)


def main():
    """Example 2: FHE graph validation"""
    print("\n" + "="*70)
    print("Example 2: FHE Graph Validation".center(70))
    print("="*70 + "\n")
    
    print("Creating a VALID FHE graph (weighted average)...")
    valid_graph = FHEGraphBuilder("valid_aggregation")
    valid_graph.weighted_sum(
        ["update_0", "update_1", "update_2"],
        [0.3, 0.5, 0.2],
        "aggregated"
    )
    graph = valid_graph.build()
    
    is_valid, errors = graph.validate()
    print(f"  Valid: {is_valid}")
    if is_valid:
        depth = graph.estimate_depth()
        print(f"  Multiplicative depth: {depth}")
    print(f"  {graph}")
    
    print("\nCreating an INVALID FHE graph (contains division)...")
    try:
        invalid_graph = FHEGraphBuilder("invalid_aggregation")
        # Manually add an unsupported operation
        invalid_graph.graph.add_operation(
            FHEOperation.DIVIDE,
            ["update_0", "update_1"],
            "result"
        )
        graph = invalid_graph.build()
        
        is_valid, errors = graph.validate()
        print(f"  Valid: {is_valid}")
        if not is_valid:
            print(f"  Errors:")
            for error in errors:
                print(f"    - {error}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)
    
    return valid_graph.build()


if __name__ == "__main__":
    main()

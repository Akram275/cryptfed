# Level 4: Modular Aggregation System

This directory contains examples demonstrating the modular payload and aggregation system. Each example focuses on a specific aspect of the framework.

## Examples

1. **`basic_payload.py`** - Creating and inspecting client payloads with mixed encrypted/plaintext data
2. **`fhe_graph_validation.py`** - Validating FHE operations before execution
3. **`graph_depth_estimation.py`** - Estimating multiplicative depth for complex graphs
4. **`fairness_weighted_aggregation.py`** - Using fairness metrics to compute aggregation weights
5. **`custom_aggregation_function.py`** - Defining custom aggregation logic from functions
6. **`common_graph_patterns.py`** - Using pre-built patterns for common aggregation strategies
7. **`graph_summary.py`** - Getting detailed summaries of FHE computation graphs

## Running the Examples

Each example can be run independently:

```bash
cd examples/level_4_modularity
python basic_payload.py
python fhe_graph_validation.py
# ... etc
```

## Key Concepts

- **Modular Payloads**: Clients can send heterogeneous data (encrypted + plaintext)
- **FHE Graphs**: Define computation logic that is validated for FHE compatibility
- **Depth Estimation**: Automatically compute multiplicative depth for parameter selection
- **Custom Aggregation**: Easy definition of custom aggregation strategies
- **Operation Validation**: Unsupported operations (max, min, div) are rejected at validation time

## Prerequisites

Make sure you have the cryptfed library installed:

```bash
pip install -e ..
```

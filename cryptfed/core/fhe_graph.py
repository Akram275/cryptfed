"""
FHE Operation Graph: Validates and analyzes FHE computations.

This module provides tools to:
1. Define aggregation logic as computation graphs
2. Validate FHE operations (reject unsupported ops like max, min, division)
3. Estimate multiplicative depth for parameter selection
"""

from typing import Any, Callable, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class FHEOperation(Enum):
    """Supported FHE operations"""
    # Arithmetic operations (supported)
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    SCALAR_MULTIPLY = "scalar_multiply"
    
    # Aggregation operations (supported)
    SUM = "sum"
    WEIGHTED_SUM = "weighted_sum"
    
    # Unsupported operations (will be rejected)
    DIVIDE = "divide"
    MAX = "max"
    MIN = "min"
    COMPARISON = "comparison"
    MODULO = "modulo"
    EXPONENT = "exponent"
    LOGARITHM = "logarithm"
    SQRT = "sqrt"
    
    # Control flow
    CONDITIONAL = "conditional"


class OperationDepth:
    """Multiplicative depth tracking for FHE operations"""
    DEPTHS = {
        FHEOperation.ADD: 0,
        FHEOperation.SUBTRACT: 0,
        FHEOperation.SUM: 0,
        FHEOperation.SCALAR_MULTIPLY: 0,  # Scalar is plaintext
        FHEOperation.MULTIPLY: 1,  # Ciphertext-ciphertext multiplication
        FHEOperation.WEIGHTED_SUM: 1,  # Involves multiplication
    }
    
    @staticmethod
    def get_depth(operation: FHEOperation) -> int:
        """Get the multiplicative depth contribution of an operation"""
        return OperationDepth.DEPTHS.get(operation, 0)


@dataclass
class FHEOperationNode:
    """
    A node in the FHE computation graph.
    
    Attributes:
        operation: The FHE operation type
        inputs: Input node names or constants
        output_name: Name of the output variable
        metadata: Additional information
    """
    operation: FHEOperation
    inputs: List[Union[str, float, int]]
    output_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    depth: int = 0
    
    def __repr__(self):
        return f"FHENode({self.output_name} = {self.operation.value}({self.inputs}))"


class FHEComputationGraph:
    """
    Represents a computation graph for FHE operations.
    
    This graph can be validated and analyzed before execution.
    """
    
    # Operations that are supported in FHE
    SUPPORTED_OPS = {
        FHEOperation.ADD,
        FHEOperation.SUBTRACT,
        FHEOperation.MULTIPLY,
        FHEOperation.SCALAR_MULTIPLY,
        FHEOperation.SUM,
        FHEOperation.WEIGHTED_SUM,
    }
    
    # Operations that are NOT supported in FHE
    UNSUPPORTED_OPS = {
        FHEOperation.DIVIDE,
        FHEOperation.MAX,
        FHEOperation.MIN,
        FHEOperation.COMPARISON,
        FHEOperation.MODULO,
        FHEOperation.EXPONENT,
        FHEOperation.LOGARITHM,
        FHEOperation.SQRT,
        FHEOperation.CONDITIONAL,
    }
    
    def __init__(self, name: str = "aggregation"):
        self.name = name
        self.nodes: List[FHEOperationNode] = []
        self.input_names: Set[str] = set()
        self.output_names: Set[str] = set()
        self._validated = False
        self._max_depth = 0
    
    def add_operation(self, operation: FHEOperation, inputs: List[Union[str, float, int]], 
                     output_name: str, metadata: Optional[Dict] = None) -> 'FHEComputationGraph':
        """Add an operation node to the graph"""
        node = FHEOperationNode(
            operation=operation,
            inputs=inputs,
            output_name=output_name,
            metadata=metadata or {}
        )
        self.nodes.append(node)
        self.output_names.add(output_name)
        
        # Track input names (string inputs are variable names)
        for inp in inputs:
            if isinstance(inp, str):
                self.input_names.add(inp)
        
        self._validated = False  # Invalidate previous validation
        return self
    
    def validate(self) -> tuple[bool, List[str]]:
        """
        Validate the computation graph.
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check for unsupported operations
        for node in self.nodes:
            if node.operation in self.UNSUPPORTED_OPS:
                errors.append(
                    f"Operation '{node.operation.value}' at node '{node.output_name}' "
                    f"is not supported in FHE. Supported operations: "
                    f"{[op.value for op in self.SUPPORTED_OPS]}"
                )
        
        # Check for undefined inputs
        defined_vars = self.input_names.copy()
        for node in self.nodes:
            for inp in node.inputs:
                if isinstance(inp, str) and inp not in defined_vars:
                    errors.append(
                        f"Node '{node.output_name}' references undefined variable '{inp}'"
                    )
            defined_vars.add(node.output_name)
        
        # Check for cycles (simplified check)
        if self._has_cycle():
            errors.append("Computation graph contains a cycle (circular dependencies)")
        
        self._validated = len(errors) == 0
        return self._validated, errors
    
    def _has_cycle(self) -> bool:
        """Simple cycle detection using topological sort approach"""
        # Build dependency graph
        dependencies = {}
        for node in self.nodes:
            deps = [inp for inp in node.inputs if isinstance(inp, str)]
            dependencies[node.output_name] = deps
        
        # Attempt topological sort
        visited = set()
        rec_stack = set()
        
        def visit(var):
            if var in rec_stack:
                return True  # Cycle detected
            if var in visited:
                return False
            
            visited.add(var)
            rec_stack.add(var)
            
            for dep in dependencies.get(var, []):
                if visit(dep):
                    return True
            
            rec_stack.remove(var)
            return False
        
        for var in dependencies:
            if visit(var):
                return True
        
        return False
    
    def estimate_depth(self) -> int:
        """
        Estimate the maximum multiplicative depth of the computation.
        
        Returns:
            Maximum multiplicative depth
        """
        if not self._validated:
            valid, errors = self.validate()
            if not valid:
                raise ValueError(f"Cannot estimate depth of invalid graph: {errors}")
        
        # Compute depth for each node
        var_depths = {inp: 0 for inp in self.input_names}
        
        # Process nodes in topological order
        sorted_nodes = self._topological_sort()
        
        for node in sorted_nodes:
            # Compute depth based on inputs
            input_depths = []
            for inp in node.inputs:
                if isinstance(inp, str):
                    input_depths.append(var_depths.get(inp, 0))
                else:
                    input_depths.append(0)  # Constant
            
            max_input_depth = max(input_depths) if input_depths else 0
            node.depth = max_input_depth + OperationDepth.get_depth(node.operation)
            var_depths[node.output_name] = node.depth
        
        self._max_depth = max(var_depths.values()) if var_depths else 0
        return self._max_depth
    
    def _topological_sort(self) -> List[FHEOperationNode]:
        """Sort nodes in topological order (dependencies first)"""
        # Build dependency map
        dependencies = {}
        node_map = {}
        for node in self.nodes:
            deps = [inp for inp in node.inputs if isinstance(inp, str)]
            dependencies[node.output_name] = deps
            node_map[node.output_name] = node
        
        # Kahn's algorithm
        in_degree = {name: 0 for name in dependencies}
        for deps in dependencies.values():
            for dep in deps:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        queue = [name for name, degree in in_degree.items() if degree == 0]
        sorted_names = []
        
        while queue:
            name = queue.pop(0)
            sorted_names.append(name)
            
            for deps_name, deps in dependencies.items():
                if name in deps:
                    in_degree[deps_name] -= 1
                    if in_degree[deps_name] == 0:
                        queue.append(deps_name)
        
        return [node_map[name] for name in sorted_names if name in node_map]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the computation graph"""
        valid, errors = self.validate()
        
        summary = {
            "name": self.name,
            "num_nodes": len(self.nodes),
            "num_inputs": len(self.input_names),
            "num_outputs": len(self.output_names),
            "is_valid": valid,
            "errors": errors,
            "max_depth": None
        }
        
        if valid:
            summary["max_depth"] = self.estimate_depth()
        
        return summary
    
    def __repr__(self):
        status = "valid" if self._validated else "unvalidated"
        return f"FHEGraph('{self.name}', {len(self.nodes)} nodes, {status})"


class FHEGraphBuilder:
    """
    Helper class to build FHE computation graphs with a fluent API.
    
    Example:
        builder = FHEGraphBuilder("weighted_aggregation")
        builder.weighted_sum(["update_1", "update_2"], [0.5, 0.5], "aggregated")
        builder.scalar_multiply("aggregated", 0.1, "final")
        graph = builder.build()
    """
    
    def __init__(self, name: str = "aggregation"):
        self.graph = FHEComputationGraph(name)
    
    def add(self, var1: str, var2: str, output: str) -> 'FHEGraphBuilder':
        """Add two encrypted variables"""
        self.graph.add_operation(FHEOperation.ADD, [var1, var2], output)
        return self
    
    def subtract(self, var1: str, var2: str, output: str) -> 'FHEGraphBuilder':
        """Subtract two encrypted variables"""
        self.graph.add_operation(FHEOperation.SUBTRACT, [var1, var2], output)
        return self
    
    def multiply(self, var1: str, var2: str, output: str) -> 'FHEGraphBuilder':
        """Multiply two encrypted variables (increases depth)"""
        self.graph.add_operation(FHEOperation.MULTIPLY, [var1, var2], output)
        return self
    
    def scalar_multiply(self, var: str, scalar: Union[float, int], output: str) -> 'FHEGraphBuilder':
        """Multiply encrypted variable by plaintext scalar"""
        self.graph.add_operation(FHEOperation.SCALAR_MULTIPLY, [var, scalar], output)
        return self
    
    def sum(self, variables: List[str], output: str) -> 'FHEGraphBuilder':
        """Sum multiple encrypted variables"""
        self.graph.add_operation(FHEOperation.SUM, variables, output)
        return self
    
    def weighted_sum(self, variables: List[str], weights: List[float], output: str) -> 'FHEGraphBuilder':
        """Compute weighted sum of encrypted variables"""
        self.graph.add_operation(
            FHEOperation.WEIGHTED_SUM, 
            variables + weights, 
            output,
            metadata={"num_vars": len(variables), "num_weights": len(weights)}
        )
        return self
    
    def build(self) -> FHEComputationGraph:
        """Build and return the computation graph"""
        return self.graph


# Predefined common patterns
class CommonGraphPatterns:
    """Common FHE computation patterns"""
    
    @staticmethod
    def fedavg_pattern(num_clients: int) -> FHEComputationGraph:
        """
        Standard FedAvg pattern: weighted average of client updates.
        
        Inputs: update_0, update_1, ..., update_n, weight_0, ..., weight_n
        Output: aggregated_update
        """
        builder = FHEGraphBuilder("fedavg")
        
        update_vars = [f"update_{i}" for i in range(num_clients)]
        weights = [f"weight_{i}" for i in range(num_clients)]
        
        builder.weighted_sum(update_vars, weights, "aggregated_update")
        
        return builder.build()
    
    @staticmethod
    def fairness_weighted_pattern(num_clients: int) -> FHEComputationGraph:
        """
        Fairness-aware aggregation: weights computed from fairness metrics.
        
        Inputs: update_i, fairness_i for each client
        Intermediate: weight_i computed from fairness metrics
        Output: aggregated_update
        """
        builder = FHEGraphBuilder("fairness_weighted")
        
        # Compute normalized weights from fairness metrics
        for i in range(num_clients):
            # In practice, weight computation could be more complex
            builder.scalar_multiply(f"fairness_{i}", 1.0, f"weight_{i}")
        
        update_vars = [f"update_{i}" for i in range(num_clients)]
        weight_vars = [f"weight_{i}" for i in range(num_clients)]
        
        builder.weighted_sum(update_vars, weight_vars, "aggregated_update")
        
        return builder.build()

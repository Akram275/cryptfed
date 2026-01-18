"""
Modular Aggregation Framework with FHE Graph Support.

This module provides a flexible aggregation system that:
1. Works with ClientPayload objects
2. Supports custom aggregation logic via FHE graphs
3. Validates operations before execution
4. Maintains backward compatibility with existing aggregators
"""

from typing import Any, Callable, Dict, List, Optional, Union
from abc import ABC, abstractmethod
import numpy as np
import logging

from ..core.payload import ClientPayload, PayloadItem, PayloadItemType, extract_legacy_updates
from ..core.fhe_graph import (
    FHEComputationGraph, 
    FHEOperation, 
    FHEGraphBuilder,
    CommonGraphPatterns
)


class ModularAggregator(ABC):
    """
    Base class for modular aggregators that work with ClientPayload objects.
    
    This extends the concept of aggregation to support:
    - Heterogeneous data (encrypted + plaintext)
    - Custom aggregation logic
    - FHE operation validation
    """
    
    def __init__(self, benchmark_manager: Any = None):
        self.benchmark_manager = benchmark_manager
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._fhe_graph: Optional[FHEComputationGraph] = None
    
    @property
    @abstractmethod
    def requires_plaintext_updates(self) -> bool:
        """Whether this aggregator requires plaintext updates"""
        pass
    
    @property
    def is_interactive(self) -> bool:
        """Whether this aggregator requires interactive protocol"""
        return False
    
    @property
    def fhe_graph(self) -> Optional[FHEComputationGraph]:
        """Get the FHE computation graph if defined"""
        return self._fhe_graph
    
    def set_fhe_graph(self, graph: FHEComputationGraph):
        """
        Set and validate the FHE computation graph.
        
        Raises:
            ValueError: If the graph contains invalid operations
        """
        valid, errors = graph.validate()
        if not valid:
            error_msg = "FHE graph validation failed:\n" + "\n".join(errors)
            raise ValueError(error_msg)
        
        depth = graph.estimate_depth()
        self.logger.info(
            f"FHE graph '{graph.name}' validated successfully. "
            f"Max multiplicative depth: {depth}"
        )
        
        self._fhe_graph = graph
    
    @abstractmethod
    def aggregate_payloads(self, payloads: List[ClientPayload], 
                          cc: Any = None, **kwargs) -> Any:
        """
        Aggregate client payloads.
        
        Args:
            payloads: List of ClientPayload objects from clients
            cc: Optional FHE CryptoContext
            **kwargs: Additional arguments
            
        Returns:
            Aggregated result (format depends on implementation)
        """
        pass
    
    def aggregate(self, updates: List[Any], weights: List[float], 
                 cc: Any = None, vec_len: int = 0, **kwargs) -> Any:
        """
        Legacy interface for backward compatibility.
        
        This method converts legacy format to payload format and delegates
        to aggregate_payloads.
        """
        # Import here to avoid circular dependency
        from ..core.payload import create_legacy_payload
        
        # Convert to payload format
        payloads = []
        for i, (update, weight) in enumerate(zip(updates, weights)):
            client_id = kwargs.get('client_ids', [f"client_{i}"])[i] if 'client_ids' in kwargs else f"client_{i}"
            is_encrypted = not self.requires_plaintext_updates
            payload = create_legacy_payload(client_id, update, is_encrypted, weight)
            payloads.append(payload)
        
        # Delegate to new interface
        result = self.aggregate_payloads(payloads, cc, **kwargs)
        
        return result


class GraphBasedAggregator(ModularAggregator):
    """
    Aggregator that executes a predefined FHE computation graph.
    
    This allows users to define custom aggregation logic that is validated
    for FHE compatibility before execution.
    """
    
    def __init__(self, graph: Optional[FHEComputationGraph] = None,
                 benchmark_manager: Any = None):
        super().__init__(benchmark_manager)
        if graph is not None:
            self.set_fhe_graph(graph)
        self._graph_executor = None
    
    @property
    def requires_plaintext_updates(self) -> bool:
        # Graph-based aggregators work with encrypted data
        return False
    
    def aggregate_payloads(self, payloads: List[ClientPayload], 
                          cc: Any = None, **kwargs) -> Any:
        """
        Execute the FHE computation graph on client payloads.
        
        Args:
            payloads: Client payloads
            cc: FHE CryptoContext (if using FHE)
            
        Returns:
            Aggregated result
        """
        if self._fhe_graph is None:
            raise ValueError("No FHE graph defined for this aggregator")
        
        # Extract data from payloads based on graph inputs
        variables = self._extract_variables(payloads)
        
        # Execute the graph
        result = self._execute_graph(variables, cc)
        
        return result
    
    def _extract_variables(self, payloads: List[ClientPayload]) -> Dict[str, Any]:
        """Extract variables needed for graph execution from payloads"""
        variables = {}
        
        for i, payload in enumerate(payloads):
            # Extract model updates
            model_item = payload.get_item("model_update")
            if model_item:
                variables[f"update_{i}"] = model_item.data
                variables[f"weight_{i}"] = payload.weight
            
            # Extract other items (statistics, fairness metrics, etc.)
            for name, item in payload.items.items():
                if name != "model_update":
                    var_name = f"{payload.client_id}_{name}"
                    variables[var_name] = item.data
        
        return variables
    
    def _execute_graph(self, variables: Dict[str, Any], cc: Any = None) -> Any:
        """
        Execute the FHE computation graph.
        
        This is a simplified execution engine. In practice, this would
        interface with the actual FHE library.
        """
        # Create execution context
        context = variables.copy()
        
        # Process nodes in topological order
        sorted_nodes = self._fhe_graph._topological_sort()
        
        for node in sorted_nodes:
            result = self._execute_node(node, context, cc)
            context[node.output_name] = result
        
        # Return final outputs
        final_output = self._get_final_output(context)
        return final_output
    
    def _execute_node(self, node, context: Dict[str, Any], cc: Any) -> Any:
        """Execute a single graph node"""
        from .fhe_graph import FHEOperation
        
        # Get input values
        input_values = []
        for inp in node.inputs:
            if isinstance(inp, str):
                if inp not in context:
                    raise ValueError(f"Variable '{inp}' not found in execution context")
                input_values.append(context[inp])
            else:
                input_values.append(inp)  # Constant value
        
        # Execute operation
        if node.operation == FHEOperation.ADD:
            return self._fhe_add(input_values[0], input_values[1], cc)
        elif node.operation == FHEOperation.SUBTRACT:
            return self._fhe_subtract(input_values[0], input_values[1], cc)
        elif node.operation == FHEOperation.MULTIPLY:
            return self._fhe_multiply(input_values[0], input_values[1], cc)
        elif node.operation == FHEOperation.SCALAR_MULTIPLY:
            return self._fhe_scalar_multiply(input_values[0], input_values[1], cc)
        elif node.operation == FHEOperation.SUM:
            return self._fhe_sum(input_values, cc)
        elif node.operation == FHEOperation.WEIGHTED_SUM:
            num_vars = node.metadata.get("num_vars", len(input_values) // 2)
            variables = input_values[:num_vars]
            weights = input_values[num_vars:]
            return self._fhe_weighted_sum(variables, weights, cc)
        else:
            raise NotImplementedError(f"Operation {node.operation} not implemented")
    
    def _get_final_output(self, context: Dict[str, Any]) -> Any:
        """Get the final output from execution context"""
        # Return the last computed value or a specific output
        if "aggregated_update" in context:
            return context["aggregated_update"]
        
        # Return last output in topological order
        if self._fhe_graph.output_names:
            output_name = list(self._fhe_graph.output_names)[-1]
            return context.get(output_name)
        
        return None
    
    # FHE operation implementations (these delegate to the actual FHE library)
    def _fhe_add(self, a, b, cc):
        """Add two encrypted values"""
        try:
            # Try OpenFHE addition
            return a + b
        except:
            # Fallback to numpy for plaintext
            return np.add(a, b)
    
    def _fhe_subtract(self, a, b, cc):
        """Subtract two encrypted values"""
        try:
            return a - b
        except:
            return np.subtract(a, b)
    
    def _fhe_multiply(self, a, b, cc):
        """Multiply two encrypted values"""
        try:
            return a * b
        except:
            return np.multiply(a, b)
    
    def _fhe_scalar_multiply(self, a, scalar, cc):
        """Multiply encrypted value by plaintext scalar"""
        try:
            return a * scalar
        except:
            return a * scalar
    
    def _fhe_sum(self, values, cc):
        """Sum multiple encrypted values"""
        result = values[0]
        for val in values[1:]:
            result = self._fhe_add(result, val, cc)
        return result
    
    def _fhe_weighted_sum(self, values, weights, cc):
        """Compute weighted sum of encrypted values"""
        result = self._fhe_scalar_multiply(values[0], weights[0], cc)
        for val, weight in zip(values[1:], weights[1:]):
            weighted = self._fhe_scalar_multiply(val, weight, cc)
            result = self._fhe_add(result, weighted, cc)
        return result


class CustomAggregatorBuilder:
    """
    Helper to build custom aggregators from aggregation functions.
    
    Example:
        def my_aggregation(payloads, cc=None):
            # Custom aggregation logic
            updates = [p.get_item("model_update").data for p in payloads]
            return sum(updates) / len(updates)
        
        aggregator = CustomAggregatorBuilder.from_function(
            my_aggregation,
            requires_plaintext=False
        )
    """
    
    @staticmethod
    def from_function(agg_fn: Callable, requires_plaintext: bool = False,
                     name: str = "custom") -> ModularAggregator:
        """Create a ModularAggregator from a function"""
        
        class FunctionAggregator(ModularAggregator):
            def __init__(self, benchmark_manager=None):
                super().__init__(benchmark_manager)
                self._name = name
                self._agg_fn = agg_fn
                self._requires_plaintext = requires_plaintext
            
            @property
            def requires_plaintext_updates(self) -> bool:
                return self._requires_plaintext
            
            def aggregate_payloads(self, payloads, cc=None, **kwargs):
                return self._agg_fn(payloads, cc=cc, **kwargs)
        
        return FunctionAggregator()
    
    @staticmethod
    def from_graph(graph: FHEComputationGraph, 
                   benchmark_manager: Any = None) -> GraphBasedAggregator:
        """Create a GraphBasedAggregator from an FHE graph"""
        return GraphBasedAggregator(graph, benchmark_manager)


# Quick factory functions for common patterns
def create_fedavg_aggregator(num_clients: int) -> GraphBasedAggregator:
    """Create FedAvg aggregator using FHE graph"""
    graph = CommonGraphPatterns.fedavg_pattern(num_clients)
    return GraphBasedAggregator(graph)


def create_fairness_aggregator(num_clients: int) -> GraphBasedAggregator:
    """Create fairness-aware aggregator using FHE graph"""
    graph = CommonGraphPatterns.fairness_weighted_pattern(num_clients)
    return GraphBasedAggregator(graph)

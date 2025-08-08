"""
Optimization module for parameter tuning and overfitting prevention.

This module provides a meta-optimization layer that sits above the core backtester,
enabling systematic parameter exploration with robust validation.
"""

from .orchestrator import OptimizationOrchestrator
from .algorithms.base import BaseOptimizer
from .search_space.parameter import Parameter, ParameterType
from .search_space.space import SearchSpace

__all__ = [
    'OptimizationOrchestrator',
    'BaseOptimizer', 
    'Parameter',
    'ParameterType',
    'SearchSpace'
]
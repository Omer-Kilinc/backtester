import itertools
from typing import Dict, Any, Optional, List
import numpy as np
from sdk.optimization.search_space.space import SearchSpace
from .base import BaseOptimizer, OptimizationResult


class GridSearchOptimizer(BaseOptimizer):
    def __init__(self, default_n_points: int = 10):
        """
        Initialize GridSearchOptimizer.
        
        Args:
            default_n_points: Default number of points for numeric parameters without step size or n_points
        """
        super().__init__()
        self.default_n_points = default_n_points
        self.grid: List[Dict[str, Any]] = []
        self.current_index: int = 0
        self.history: List[OptimizationResult] = []
        self.maximize: bool = False

    def initialize(self, search_space: SearchSpace, n_trials: Optional[int] = None, maximize: bool = False):
        self.search_space = search_space
        self.n_trials = n_trials
        self.maximize = maximize
        self.grid = self._generate_grid()
        self.current_index = 0
        self.history.clear()
        self.is_initialized = True

    def _generate_grid(self) -> List[Dict[str, Any]]:
        param_grids = []
        param_names = []

        for param in self.search_space.parameters:
            if param.name is None:
                raise ValueError("Parameter name cannot be None for grid search")
                
            param_names.append(param.name)

            if param.param_type == "boolean":
                values = [True, False]

            elif param.param_type == "categorical":
                if not param.choices:
                    raise ValueError(f"Categorical parameter '{param.name}' requires choices")
                values = param.choices

            elif param.param_type in ["integer", "float"]:
                if param.step is not None:
                    # Use step size if provided
                    values = np.arange(param.low, param.high + param.step / 10, param.step)
                    if param.param_type == "integer":
                        values = values.round().astype(int)
                    values = values.tolist()
                elif param.n_points is not None:
                    # Use parameter-specific n_points if provided
                    values = np.linspace(param.low, param.high, param.n_points)
                    if param.param_type == "integer":
                        values = np.unique(values.round().astype(int)).tolist()
                    else:
                        values = values.tolist()
                else:
                    # Use default n_points
                    values = np.linspace(param.low, param.high, self.default_n_points)
                    if param.param_type == "integer":
                        values = np.unique(values.round().astype(int)).tolist()
                    else:
                        values = values.tolist()

            else:
                raise ValueError(f"Unsupported parameter type: {param.param_type}")

            param_grids.append(values)

        full_grid = [
            dict(zip(param_names, combo))
            for combo in itertools.product(*param_grids)
        ]
        return full_grid if self.n_trials is None else full_grid[:self.n_trials]

    def suggest_parameters(self) -> Optional[Dict[str, Any]]:
        if self.current_index >= len(self.grid):
            return None
        params = self.grid[self.current_index]
        self.current_index += 1
        return params

    def update_with_result(self, parameters: Dict[str, Any], result: OptimizationResult):
        self.history.append(result)

    def is_finished(self) -> bool:
        return self.current_index >= len(self.grid)

    def get_total_trials(self) -> int:
        return len(self.grid)

    def get_best_result(self) -> Optional[OptimizationResult]:
        if not self.history:
            return None
        
        # Simple, clear logic - no negation needed
        if self.maximize:
            return max(self.history, key=lambda x: x.objective_value)
        else:
            return min(self.history, key=lambda x: x.objective_value)

    def get_optimization_history(self) -> List[OptimizationResult]:
        # Return trials in the order they were evaluated
        return self.history.copy()

    def get_sorted_results(self) -> List[OptimizationResult]:
        # Return trials sorted by objective (best first)
        if not self.history:
            return []
        
        return sorted(self.history, 
                     key=lambda x: x.objective_value, 
                     reverse=self.maximize)
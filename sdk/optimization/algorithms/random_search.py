from typing import Dict, Any, Optional, List
import numpy as np
from sdk.optimization.search_space.space import SearchSpace
from .base import BaseOptimizer, OptimizationResult


class RandomSearchOptimizer(BaseOptimizer):
    def __init__(self, seed: Optional[int] = None, default_n_trials: int = 100):
        """
        Initialize RandomSearchOptimizer.
        
        Args:
            seed: Random seed for reproducible results
            default_n_trials: Default number of trials if not specified in initialize
        """
        super().__init__()
        self.seed = seed
        self.default_n_trials = default_n_trials
        self.rng: Optional[np.random.RandomState] = None
        self.trials_completed: int = 0
        self.history: List[OptimizationResult] = []
        self.maximize: bool = False

    def initialize(self, search_space: SearchSpace, n_trials: Optional[int] = None, maximize: bool = False):
        """
        Initialize the random search optimizer.
        
        Args:
            search_space: Parameter search space
            n_trials: Number of random trials to perform
            maximize: Whether to maximize (True) or minimize (False) the objective
        """
        self.search_space = search_space
        self.n_trials = n_trials or self.default_n_trials
        self.maximize = maximize
        self.trials_completed = 0
        self.history.clear()
        self.rng = np.random.RandomState(self.seed)
        self.is_initialized = True

        # Validate that all parameters have names
        for param in self.search_space.parameters:
            if param.name is None:
                raise ValueError("Parameter name cannot be None for random search")

    def suggest_parameters(self) -> Optional[Dict[str, Any]]:
        """
        Suggest the next random parameter configuration.
        
        Returns:
            Dict of parameter names to values, or None if finished
        """
        if self.is_finished():
            return None
            
        # Sample parameters using the parameter's built-in sample method
        parameters = {}
        for param in self.search_space.parameters:
            parameters[param.name] = param.sample(random_state=self.rng)
            
        return parameters

    def update_with_result(self, parameters: Dict[str, Any], result: OptimizationResult):
        """
        Update the optimizer with the result from evaluating a configuration.
        
        Args:
            parameters: The parameter configuration that was evaluated
            result: The result from evaluating the configuration
        """
        self.history.append(result)
        self.trials_completed += 1

    def is_finished(self) -> bool:
        """
        Check if the optimizer has completed all trials.
        
        Returns:
            True if all trials have been completed
        """
        return self.trials_completed >= self.n_trials

    def get_total_trials(self) -> int:
        """
        Get the total number of trials this optimizer will run.
        
        Returns:
            Total number of trials
        """
        return self.n_trials

    def get_best_result(self) -> Optional[OptimizationResult]:
        """
        Get the best result found so far.
        
        Returns:
            Best optimization result, or None if no results yet
        """
        if not self.history:
            return None
        
        # Simple, clear logic - no negation needed
        if self.maximize:
            return max(self.history, key=lambda x: x.objective_value)
        else:
            return min(self.history, key=lambda x: x.objective_value)

    def get_optimization_history(self) -> List[OptimizationResult]:
        """
        Get the complete history of evaluations in order.
        
        Returns:
            List of all optimization results in evaluation order
        """
        return self.history.copy()

    def get_sorted_results(self) -> List[OptimizationResult]:
        """
        Get all results sorted by objective value (best first).
        
        Returns:
            List of all optimization results sorted by performance
        """
        if not self.history:
            return []
        
        return sorted(self.history, 
                     key=lambda x: x.objective_value, 
                     reverse=self.maximize)

    def reset(self):
        """
        Reset the optimizer to its initial state.
        Useful for running multiple optimization runs.
        """
        self.trials_completed = 0
        self.history.clear()
        if self.seed is not None:
            self.rng = np.random.RandomState(self.seed)
        else:
            self.rng = np.random.RandomState()
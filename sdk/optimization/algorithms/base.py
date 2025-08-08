from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from sdk.optimization.search_space.space import SearchSpace


class OptimizationResult(BaseModel):
    """
    Result from a single optimization trial.
    """
    parameters: Dict[str, Any]
    objective_value: float
    metrics: Optional[Any] = None
    backtest_result: Optional[Dict[str, Any]] = None
    success: bool = True
    error: Optional[str] = None


class BaseOptimizer(ABC):
    """
    Abstract base class for all optimizers.
    """

    def __init__(self, **kwargs):
        self.search_space: Optional[SearchSpace] = None
        self.n_trials: Optional[int] = None
        self.is_initialized = False

    @abstractmethod
    def initialize(self, search_space: SearchSpace, n_trials: Optional[int] = None):
        """
        Prepare the optimizer with the search space and trial budget.
        """
        pass

    @abstractmethod
    def suggest_parameters(self) -> Optional[Dict[str, Any]]:
        """
        Suggest the next configuration to evaluate.
        """
        pass

    @abstractmethod
    def update_with_result(self, parameters: Dict[str, Any], result: OptimizationResult):
        """
        Inform the optimizer of the result from evaluating a configuration.
        """
        pass

    @abstractmethod
    def is_finished(self) -> bool:
        """
        Whether the optimizer has completed its trial budget or convergence.
        """
        pass

    @abstractmethod
    def get_total_trials(self) -> int:
        """
        Total number of trials the optimizer expects to run.
        """
        pass

    @abstractmethod
    def get_best_result(self) -> Optional[OptimizationResult]:
        """
        Best result seen so far.
        """
        pass

    @abstractmethod
    def get_optimization_history(self) -> List[OptimizationResult]:
        """
        Complete list of all evaluated configurations.
        """
        pass

    @abstractmethod
    def get_sorted_results(self) -> List[OptimizationResult]:
        """
        Complete list of all evaluated configurations sorted by objective value.
        """
        pass


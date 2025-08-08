import optuna
from typing import Dict, Any, Optional, List
from sdk.optimization.search_space.space import SearchSpace
from .base import BaseOptimizer, OptimizationResult


class BayesianOptimizer(BaseOptimizer):
    def __init__(self, sampler: Optional[optuna.samplers.BaseSampler] = None):
        super().__init__()
        self.study: Optional[optuna.study.Study] = None
        self.search_space: Optional[SearchSpace] = None
        self.maximize: bool = False
        self.n_trials: Optional[int] = None
        self.history: List[OptimizationResult] = []
        self.trial_mapping: Dict[int, optuna.trial.FrozenTrial] = {}
        self.sampler = sampler or optuna.samplers.TPESampler()

    def initialize(self, search_space: SearchSpace, n_trials: Optional[int] = None, maximize: bool = False):
        self.search_space = search_space
        self.n_trials = n_trials
        self.maximize = maximize
        direction = "maximize" if maximize else "minimize"
        self.study = optuna.create_study(direction=direction, sampler=self.sampler)
        self.history.clear()
        self.trial_mapping.clear()
        self.is_initialized = True

    def _convert_search_space(self) -> Dict[str, Any]:
        converted = {}
        for param in self.search_space.parameters:
            if param.param_type == "boolean":
                converted[param.name] = [True, False]
            elif param.param_type == "categorical":
                converted[param.name] = param.choices
            elif param.param_type == "integer":
                converted[param.name] = {
                    "low": int(param.low),
                    "high": int(param.high),
                    "step": int(param.step) if param.step else 1,
                    "type": int
                }
            elif param.param_type == "float":
                converted[param.name] = {
                    "low": float(param.low),
                    "high": float(param.high),
                    "step": float(param.step) if param.step else None,
                    "type": float
                }
            else:
                raise ValueError(f"Unsupported parameter type: {param.param_type}")
        return converted

    def suggest_parameters(self) -> Optional[Dict[str, Any]]:
        if self.n_trials is not None and len(self.history) >= self.n_trials:
            return None

        def optuna_search_space(trial: optuna.Trial):
            params = {}
            for param in self.search_space.parameters:
                name = param.name
                if param.param_type == "boolean":
                    params[name] = trial.suggest_categorical(name, [True, False])
                elif param.param_type == "categorical":
                    params[name] = trial.suggest_categorical(name, param.choices)
                elif param.param_type == "integer":
                    if param.step:
                        params[name] = trial.suggest_int(name, int(param.low), int(param.high), step=int(param.step))
                    else:
                        params[name] = trial.suggest_int(name, int(param.low), int(param.high))
                elif param.param_type == "float":
                    if param.step:
                        params[name] = trial.suggest_float(name, float(param.low), float(param.high), step=float(param.step))
                    else:
                        params[name] = trial.suggest_float(name, float(param.low), float(param.high))
                else:
                    raise ValueError(f"Unsupported parameter type: {param.param_type}")
            return params

        trial = self.study.ask()
        params = optuna_search_space(trial)
        self.trial_mapping[trial.number] = trial
        return {"__trial_number__": trial.number, **params}

    def update_with_result(self, parameters: Dict[str, Any], result: OptimizationResult):
        trial_number = parameters.pop("__trial_number__", None)
        if trial_number is None:
            raise ValueError("Missing __trial_number__ in parameters")
        trial = self.trial_mapping.pop(trial_number)
        self.study.tell(trial, result.objective_value)
        self.history.append(result)

    def is_finished(self) -> bool:
        return self.n_trials is not None and len(self.history) >= self.n_trials

    def get_total_trials(self) -> int:
        return self.n_trials if self.n_trials is not None else -1

    def get_best_result(self) -> Optional[OptimizationResult]:
        if not self.history:
            return None
        return max(self.history, key=lambda x: x.objective_value) if self.maximize else min(self.history, key=lambda x: x.objective_value)

    def get_optimization_history(self) -> List[OptimizationResult]:
        return self.history.copy()

    def get_sorted_results(self) -> List[OptimizationResult]:
        return sorted(self.history, key=lambda x: x.objective_value, reverse=self.maximize)

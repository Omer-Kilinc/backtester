"""
Search space management for optimization parameters.
"""

from typing import Dict, List, Any, Optional
import numpy as np
from .parameter import Parameter


class SearchSpace:
    """
    Manages the complete parameter search space for optimization.
    
    The search space defines all parameters that will be optimized,
    their types, ranges, and constraints.
    
    Example:
        space = SearchSpace([
            Parameter("sma_period", ParameterType.INTEGER, low=5, high=50),
            Parameter("rsi_period", ParameterType.INTEGER, low=10, high=30),
            Parameter("threshold", ParameterType.FLOAT, low=0.01, high=0.1),
            Parameter("strategy_type", ParameterType.CATEGORICAL, choices=["aggressive", "conservative"])
        ])
    """
    
    def __init__(self, parameters: List[Parameter]):
        """
        Initialize search space with parameter definitions.
        
        Args:
            parameters: List of Parameter objects defining the search space
        """
        self.parameters = {param.name: param for param in parameters}
        self._validate_search_space()
    
    def _validate_search_space(self):
        """Validate the search space for consistency."""
        if not self.parameters:
            raise ValueError("Search space cannot be empty")
            
        # Check for duplicate parameter names
        param_names = [param.name for param in self.parameters.values()]
        if len(param_names) != len(set(param_names)):
            duplicates = [name for name in param_names if param_names.count(name) > 1]
            raise ValueError(f"Duplicate parameter names found: {duplicates}")
    
    def add_parameter(self, parameter: Parameter):
        """
        Add a parameter to the search space.
        
        Args:
            parameter: Parameter to add
        """
        if parameter.name in self.parameters:
            raise ValueError(f"Parameter '{parameter.name}' already exists in search space")
        self.parameters[parameter.name] = parameter
    
    def remove_parameter(self, name: str):
        """
        Remove a parameter from the search space.
        
        Args:
            name: Name of parameter to remove
        """
        if name not in self.parameters:
            raise ValueError(f"Parameter '{name}' not found in search space")
        del self.parameters[name]
    
    def get_parameter(self, name: str) -> Parameter:
        """
        Get a parameter by name.
        
        Args:
            name: Parameter name
            
        Returns:
            Parameter object
        """
        if name not in self.parameters:
            raise ValueError(f"Parameter '{name}' not found in search space")
        return self.parameters[name]
    
    def get_parameter_names(self) -> List[str]:
        """Get list of all parameter names."""
        return list(self.parameters.keys())
    
    def get_dimensionality(self) -> int:
        """Get the dimensionality of the search space."""
        return len(self.parameters)
    
    def sample(self, n_samples: int = 1, random_state: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Sample parameter configurations from the search space.
        
        Args:
            n_samples: Number of samples to generate
            random_state: Random seed for reproducible sampling
            
        Returns:
            List of parameter dictionaries
        """
        rng = np.random.RandomState(random_state)
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            for param_name, param in self.parameters.items():
                sample[param_name] = param.sample(rng)
            samples.append(sample)
            
        return samples
    
    def validate_configuration(self, config: Dict[str, Any]) -> bool:
        """
        Validate if a parameter configuration is valid for this search space.
        
        Args:
            config: Parameter configuration to validate
            
        Returns:
            True if configuration is valid
        """
        # Check all required parameters are present
        if set(config.keys()) != set(self.parameters.keys()):
            return False
            
        # Validate each parameter value
        for param_name, value in config.items():
            if param_name not in self.parameters:
                return False
            if not self.parameters[param_name].validate_value(value):
                return False
                
        return True
    
    def clip_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clip a parameter configuration to be within the search space bounds.
        
        Args:
            config: Parameter configuration to clip
            
        Returns:
            Clipped configuration
        """
        clipped_config = {}
        
        for param_name, param in self.parameters.items():
            if param_name in config:
                clipped_config[param_name] = param.clip_value(config[param_name])
            else:
                # Use default sampling for missing parameters
                clipped_config[param_name] = param.sample()
                
        return clipped_config
    
    def get_bounds(self) -> Dict[str, tuple]:
        """
        Get bounds for all numeric parameters.
        
        Returns:
            Dict mapping parameter names to (low, high) bounds
        """
        bounds = {}
        
        for param_name, param in self.parameters.items():
            if hasattr(param, 'low') and hasattr(param, 'high'):
                if param.low is not None and param.high is not None:
                    bounds[param_name] = (param.low, param.high)
                    
        return bounds
    
    def get_categorical_parameters(self) -> Dict[str, List[Any]]:
        """
        Get all categorical parameters and their choices.
        
        Returns:
            Dict mapping categorical parameter names to their choice lists
        """
        categorical = {}
        
        for param_name, param in self.parameters.items():
            if hasattr(param, 'choices') and param.choices is not None:
                categorical[param_name] = param.choices
                
        return categorical
    
    def create_grid(self, n_points_per_dim: int = 10) -> List[Dict[str, Any]]:
        """
        Create a grid of parameter configurations for grid search.
        
        Args:
            n_points_per_dim: Number of points per dimension for numeric parameters
            
        Returns:
            List of parameter configurations forming a grid
        """
        from itertools import product
        
        param_grids = {}
        
        for param_name, param in self.parameters.items():
            if param.param_type.value == "categorical":
                param_grids[param_name] = param.choices
            elif param.param_type.value == "boolean":
                param_grids[param_name] = [True, False]
            elif param.param_type.value == "integer":
                if param.high - param.low + 1 <= n_points_per_dim:
                    # Use all integer values if range is small
                    param_grids[param_name] = list(range(int(param.low), int(param.high) + 1))
                else:
                    # Sample n_points_per_dim integer values
                    param_grids[param_name] = [
                        int(x) for x in np.linspace(param.low, param.high, n_points_per_dim)
                    ]
            elif param.param_type.value in ["float", "log_uniform"]:
                if param.param_type.value == "log_uniform":
                    # Use log scale for log_uniform parameters
                    log_points = np.linspace(np.log(param.low), np.log(param.high), n_points_per_dim)
                    param_grids[param_name] = [float(np.exp(x)) for x in log_points]
                else:
                    param_grids[param_name] = [
                        float(x) for x in np.linspace(param.low, param.high, n_points_per_dim)
                    ]
            else:
                raise ValueError(f"Unsupported parameter type for grid: {param.param_type}")
        
        # Generate all combinations
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())
        
        grid_configs = []
        for combination in product(*param_values):
            config = dict(zip(param_names, combination))
            grid_configs.append(config)
            
        return grid_configs
    
    def __len__(self) -> int:
        """Get number of parameters in search space."""
        return len(self.parameters)
    
    def __repr__(self) -> str:
        """String representation of search space."""
        param_info = []
        for name, param in self.parameters.items():
            if hasattr(param, 'low') and param.low is not None:
                param_info.append(f"{name}: {param.param_type.value}[{param.low}, {param.high}]")
            elif hasattr(param, 'choices') and param.choices is not None:
                param_info.append(f"{name}: {param.param_type.value}{param.choices}")
            else:
                param_info.append(f"{name}: {param.param_type.value}")
                
        return f"SearchSpace({', '.join(param_info)})"
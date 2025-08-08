"""
Parameter definition and types for optimization search spaces using Pydantic.
"""

from enum import Enum
from typing import Any, List, Union, Optional
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class ParameterType(str, Enum):
    """Types of parameters that can be optimized."""
    INTEGER = "integer"
    FLOAT = "float" 
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


class ParameterDistribution(str, Enum):
    """Distributions for parameter sampling."""
    UNIFORM = "uniform"
    LOG_UNIFORM = "log_uniform"
    NORMAL = "normal"
    LOG_NORMAL = "log_normal"
    CHOICE = "choice"

# TODO: If I decide to allow different variables to have different search strategies, implement this
class SearchStrategy(str, Enum):
    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian" 

class Parameter(BaseModel):
    """
    Definition of a single parameter in the optimization search space.
    
    Args:
        name: Name of the parameter
        param_type: Type of the parameter (integer, float, categorical, boolean)
        low: Lower bound for numeric parameters
        high: Upper bound for numeric parameters
        choices: List of choices for categorical parameters
        distribution: Distribution for numeric parameters
        mean: Mean for normal distribution
        std: Standard deviation for normal distribution
        step: Step size for discrete numeric parameters
    
    Examples:
        # Integer parameter
        Parameter(name="lookback_period", param_type=ParameterType.INTEGER, low=5, high=50)
        
        # Float parameter with normal distribution
        Parameter(name="threshold", param_type=ParameterType.FLOAT, low=0.01, high=0.1, 
                 distribution=ParameterDistribution.NORMAL, mean=0.05, std=0.02)
        
        # Categorical parameter
        Parameter(name="signal_type", param_type=ParameterType.CATEGORICAL, choices=["sma", "ema", "rsi"])
        
        # Log-uniform parameter (good for learning rates, etc.)
        Parameter(name="learning_rate", param_type=ParameterType.INTEGER, low=1e-5, high=1e-1)
    """

    name: Optional[str] = None
    param_type: ParameterType
    
    # Numeric parameters
    low: Optional[float] = None
    high: Optional[float] = None
    
    # Categorical parameters
    choices: Optional[List[Any]] = None
    
    # Distribution parameters
    distribution: ParameterDistribution = ParameterDistribution.UNIFORM
    mean: Optional[float] = None
    std: Optional[float] = None
    
    # Constraints
    step: Optional[float] = Field(
        None, 
        description="Spacing between discrete values for numeric parameters"
    )

    n_points: Optional[int] = Field(
        None, 
        description="Number of grid points for this parameter if step is not provided"
    )
    

    @field_validator('choices')
    @classmethod
    def validate_choices(cls, v):
        """Validate choices are not empty if provided."""
        if v is not None and len(v) == 0:
            raise ValueError("Choices list cannot be empty")
        return v
    
    @model_validator(mode='after')
    def validate_parameter(self):
        """Validate parameter definition after initialization."""
        
        # Validate numeric parameters
        if self.param_type in [ParameterType.INTEGER, ParameterType.FLOAT]:
            if self.low is None or self.high is None:
                raise ValueError(f"Numeric parameter '{self.name}' requires low and high bounds")
            if self.low >= self.high:
                raise ValueError(f"Parameter '{self.name}': low ({self.low}) must be < high ({self.high})")
                
        # Validate categorical parameters
        elif self.param_type == ParameterType.CATEGORICAL:
            if not self.choices or len(self.choices) == 0:
                raise ValueError(f"Categorical parameter '{self.name}' requires non-empty choices")
                
        
        # Validate distribution-specific parameters
        if self.distribution in [ParameterDistribution.NORMAL, ParameterDistribution.LOG_NORMAL] and (self.mean is None or self.std is None):
            raise ValueError(f"Normal distribution for '{self.name}' requires mean and std")

        # Validate step and n_points exclusivity
        if self.step is not None and self.n_points is not None:
            raise ValueError(f"Parameter '{self.name}': specify either 'step' or 'n_points', not both")

        # Validate n_points
        if self.n_points is not None:
            if self.param_type not in [ParameterType.INTEGER, ParameterType.FLOAT]:
                raise ValueError(f"Parameter '{self.name}': 'n_points' is only valid for numeric parameters")
            if self.low is None or self.high is None:
                raise ValueError(f"Parameter '{self.name}': 'n_points' requires 'low' and 'high'")
            if self.n_points < 2:
                raise ValueError(f"Parameter '{self.name}': 'n_points' must be at least 2")

            
        return self
    
    def sample(self, random_state: Optional[np.random.RandomState] = None) -> Any:
        """
        Sample a value from this parameter's distribution.
        
        Args:
            random_state: Random state for reproducible sampling
            
        Returns:
            Sampled parameter value
        """
        rng = random_state or np.random.RandomState()
        
        if self.param_type == ParameterType.BOOLEAN:
            return rng.choice([True, False])
            
        elif self.param_type == ParameterType.CATEGORICAL:
            return rng.choice(self.choices)
            
        elif self.param_type == ParameterType.INTEGER:
            if self.distribution == ParameterDistribution.UNIFORM:
                value = rng.randint(self.low, self.high + 1)  # +1 for inclusive upper bound
            elif self.distribution == ParameterDistribution.LOG_UNIFORM:
                log_low = np.log(max(1, self.low))  # Avoid log(0)
                log_high = np.log(self.high)
                log_value = rng.uniform(log_low, log_high)
                value = int(np.round(np.exp(log_value)))
                value = np.clip(value, self.low, self.high)
            elif self.distribution == ParameterDistribution.NORMAL:
                value = int(np.round(rng.normal(self.mean, self.std)))
                value = np.clip(value, self.low, self.high)
            else:
                raise ValueError(f"Unsupported distribution {self.distribution} for integer parameter")
                
            if self.step is not None:
                value = int(self.low + self.step * np.round((value - self.low) / self.step))
                
            return value
            
        elif self.param_type == ParameterType.FLOAT:
            if self.distribution == ParameterDistribution.UNIFORM:
                value = rng.uniform(self.low, self.high)
            elif self.distribution == ParameterDistribution.LOG_UNIFORM:
                log_low = np.log(max(1e-10, self.low))  # Avoid log(0)
                log_high = np.log(self.high)
                log_value = rng.uniform(log_low, log_high)
                value = np.exp(log_value)
            elif self.distribution == ParameterDistribution.NORMAL:
                value = rng.normal(self.mean, self.std)
                value = np.clip(value, self.low, self.high)
            elif self.distribution == ParameterDistribution.LOG_NORMAL:
                log_mean = np.log(self.mean) if self.mean else (np.log(max(1e-10, self.low)) + np.log(self.high)) / 2
                log_std = self.std or 0.5
                value = rng.lognormal(log_mean, log_std)
                value = np.clip(value, self.low, self.high)
            else:
                raise ValueError(f"Unsupported distribution {self.distribution} for float parameter")
                
            if self.step is not None:
                value = self.low + self.step * np.round((value - self.low) / self.step)
                
            return float(value)
        
        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")
    
    def validate_value(self, value: Any) -> bool:
        """
        Validate if a value is valid for this parameter.
        
        Args:
            value: Value to validate
            
        Returns:
            True if value is valid for this parameter
        """
        try:
            if self.param_type == ParameterType.BOOLEAN:
                return isinstance(value, bool)
                
            elif self.param_type == ParameterType.CATEGORICAL:
                return value in self.choices
                
            elif self.param_type == ParameterType.INTEGER:
                return (isinstance(value, int) and 
                       self.low <= value <= self.high)
                       
            elif self.param_type == ParameterType.FLOAT:
                return (isinstance(value, (int, float)) and 
                       self.low <= value <= self.high)
                       
            return False
            
        except (TypeError, ValueError):
            return False
    
    def clip_value(self, value: Any) -> Any:
        """
        Clip a value to be within this parameter's valid range.
        
        Args:
            value: Value to clip
            
        Returns:
            Clipped value
        """
        if self.param_type == ParameterType.BOOLEAN:
            return bool(value)
            
        elif self.param_type == ParameterType.CATEGORICAL:
            if value in self.choices:
                return value
            else:
                # Return closest choice (for ordered choices) or random choice
                return self.choices[0]  # Default to first choice
                
        elif self.param_type == ParameterType.INTEGER:
            return int(np.clip(value, self.low, self.high))
            
        elif self.param_type == ParameterType.FLOAT:
            return float(np.clip(value, self.low, self.high))
            
        return value

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
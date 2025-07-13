from pydantic import Field, model_validator
from typing import Optional, Literal, Dict, Any
import pandas as pd
from ..base import BaseConfig


class OrnsteinUhlenbeckSimulatorConfig(BaseConfig):
    simulator_type: Literal['ou'] = 'ou'
    x0: float = Field(100.0, gt=0)
    mu: float = Field(100.0, gt=0)
    theta: float = Field(0.1, gt=0)
    sigma: float = Field(1.0, gt=0)
    start_date: str
    end_date: str
    output_freq: str = "1D"
    precision_factor: int = Field(24, gt=0)
    seed: Optional[int] = None

    @model_validator(mode='before')
    @classmethod
    def validate_dates(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        start_date_str = values.get('start_date')
        end_date_str = values.get('end_date')

        if not start_date_str or not end_date_str:
            return values

        try:
            start_date = pd.to_datetime(start_date_str)
            end_date = pd.to_datetime(end_date_str)
        except pd.errors.ParserError as e:
            raise ValueError(f"Invalid date format: {e}") from e

        if start_date >= end_date:
            raise ValueError('start_date must be before end_date')

        return values

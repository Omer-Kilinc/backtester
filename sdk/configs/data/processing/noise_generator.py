import pandas as pd
from pydantic import Field, model_validator
from typing import Dict, Any, List, Optional

from ..base import BaseConfig


class NoiseGeneratorConfig(BaseConfig):
    df: pd.DataFrame
    scale: Dict[str, float] = Field(default_factory=lambda: {
        "open": 0.001, "high": 0.001, "low": 0.001, "close": 0.001
    })
    gap_chance: float = Field(0.01, ge=0, le=1)
    gap_magnitude: Dict[str, float] = Field(default_factory=lambda: {"min": 0.005, "max": 0.02})
    required_columns: List[str] = Field(default_factory=lambda: ['open', 'high', 'low', 'close'])
    seed: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='before')
    @classmethod
    def validate_inputs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        df = values.get('df')
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Input DataFrame cannot be empty.")

        required_cols = values.get('required_columns', ['open', 'high', 'low', 'close'])
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        gap_magnitude = values.get('gap_magnitude', {"min": 0.005, "max": 0.02})
        if gap_magnitude['min'] > gap_magnitude['max']:
            raise ValueError('min gap_magnitude cannot be greater than max')

        return values


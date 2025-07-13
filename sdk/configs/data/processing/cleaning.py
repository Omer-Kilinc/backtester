import pandas as pd
from pydantic import Field, model_validator
from typing import List, Literal, Dict, Any

from ..base import BaseConfig


class CleaningConfig(BaseConfig):
    data: pd.DataFrame
    fill_nulls: bool = True
    extreme_threshold: float = Field(1e6, gt=0)
    handle_extremes: Literal['flag', 'remove', 'ignore'] = 'remove'
    strict_timestamps: bool = True
    required_columns: List[str] = Field(default_factory=lambda: ['timestamp', 'open', 'close'])
    optional_columns: List[str] = Field(default_factory=lambda: ['high', 'low', 'volume'])

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='before')
    @classmethod
    def validate_data_and_columns(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        df = values.get('data')
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'data' must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Input DataFrame cannot be empty.")

        required_cols = values.get('required_columns', ['timestamp', 'open', 'close'])
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")

        return values


from pydantic import Field, field_validator, model_validator
from typing import Optional, Literal, Any
import pandas as pd
from ..base import BaseConfig


class BootstrapReturnSimulatorConfig(BaseConfig):
    simulator_type: Literal['bootstrap'] = 'bootstrap'
    cleaned_price_df: pd.DataFrame
    start_date: str
    end_date: str
    output_freq: str = "1D"
    use_path_simulation: bool = True
    inject_gaps: bool = False
    gap_volatility: float = Field(0.01, ge=0)
    seed: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True

    @field_validator('cleaned_price_df')
    @classmethod
    def validate_dataframe(cls, v: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(v, pd.DataFrame):
            raise ValueError('cleaned_price_df must be a pandas DataFrame')
        if v.empty:
            raise ValueError('cleaned_price_df cannot be empty')
        if 'close' not in v.columns:
            raise ValueError('cleaned_price_df must contain a "close" column')
        if not pd.api.types.is_numeric_dtype(v['close']):
            raise ValueError('"close" column must be numeric')
        return v

    @model_validator(mode='before')
    @classmethod
    def validate_dates(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        start_date_str = values.get('start_date')
        end_date_str = values.get('end_date')

        if not start_date_str or not end_date_str:
            # This will be caught by individual field validators, but good to have a check
            return values

        try:
            start_date = pd.to_datetime(start_date_str)
            end_date = pd.to_datetime(end_date_str)
        except pd.errors.ParserError as e:
            raise ValueError(f"Invalid date format: {e}") from e

        if start_date >= end_date:
            raise ValueError('start_date must be before end_date')

        return values

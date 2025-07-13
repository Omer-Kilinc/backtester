import pandas as pd
from pydantic import model_validator
from typing import Optional, Dict, Any

from ..base import BaseConfig


class DataTransformerConfig(BaseConfig):
    df: pd.DataFrame
    user_mapping: Optional[Dict[str, str]] = None
    keyword_detection: bool = True

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode='before')
    @classmethod
    def validate_inputs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # Validate DataFrame
        df = values.get('df')
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Input DataFrame cannot be empty.")

        # Validate mapping method
        user_mapping = values.get('user_mapping')
        keyword_detection = values.get('keyword_detection', True)

        if user_mapping and keyword_detection:
            raise ValueError('Cannot have both user_mapping and keyword_detection enabled.')
        if not user_mapping and not keyword_detection:
            raise ValueError('Must have either user_mapping or keyword_detection enabled.')

        return values


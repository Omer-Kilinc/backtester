import pandas as pd
import os
from pydantic import Field, model_validator
from typing import Dict, Any

from ..base import BaseConfig


class DataConversionConfig(BaseConfig):
    df: pd.DataFrame
    folder_path: str
    filename: str = "simulated_data.parquet"

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

        # Validate folder_path
        folder_path = values.get('folder_path')
        if not folder_path or not os.path.isdir(folder_path):
            raise ValueError(f"The folder path '{folder_path}' is not a valid directory.")

        # Validate and normalize filename
        filename = values.get('filename', "simulated_data.parquet")
        if not filename.endswith(".parquet"):
            values['filename'] = f"{filename}.parquet"

        return values


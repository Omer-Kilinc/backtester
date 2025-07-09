from pydantic import Field, field_validator
from .base import BaseConfig

class DataConversionConfig(BaseConfig):
    folder_path: str
    filename: str = "simulated_data.parquet"

    @field_validator('filename')
    def validate_filename(cls, v: str) -> str:
        if not v.endswith(".parquet"):
            return f"{v}.parquet"
        return v

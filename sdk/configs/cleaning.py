from pydantic import BaseModel, Field
from typing import List, Literal
from .base import BaseConfig

class CleaningConfig(BaseConfig):
    fill_nulls: bool = True
    extreme_threshold: float = 1e6
    handle_extremes: Literal['flag', 'remove', 'ignore'] = 'remove'
    strict_timestamps: bool = True
    required_columns: List[str] = Field(default_factory=lambda: ['timestamp', 'open', 'close'])
    optional_columns: List[str] = Field(default_factory=lambda: ['high', 'low', 'volume'])

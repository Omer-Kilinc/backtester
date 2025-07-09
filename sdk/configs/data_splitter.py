from typing import Optional
from datetime import datetime
from pydantic import Field, field_validator, model_validator, ValidationInfo
from enum import Enum
import numpy as np
from .base import BaseConfig

class SplitType(str, Enum):
    RATIO = "ratio"
    DATE = "date"

class DataSplitterConfig(BaseConfig):
    split_type: SplitType = SplitType.RATIO
    train_ratio: Optional[float] = Field(0.8, ge=0, le=1)  # Optional but defaults to 0.8
    test_ratio: Optional[float] = Field(0.2, ge=0, le=1)   # Optional but defaults to 0.2
    train_start_date: Optional[datetime] = None
    train_end_date: Optional[datetime] = None
    test_start_date: Optional[datetime] = None
    test_end_date: Optional[datetime] = None

    @field_validator('train_ratio', 'test_ratio')
    def ratios_sum_to_one(cls, v: float, info: ValidationInfo) -> float:
        """Ensure train_ratio + test_ratio ≈ 1.0 when split_type=RATIO."""
        if info.data.get('split_type') == SplitType.RATIO:
            other_ratio = info.data.get('train_ratio' if v == 'test_ratio' else 'test_ratio', 0.0)
            if not np.isclose(v + other_ratio, 1.0, rtol=1e-9):  # rtol for floating-point tolerance
                raise ValueError('train_ratio and test_ratio must sum to 1 (within floating-point tolerance)')
        return v

    @model_validator(mode='after')
    def validate_dates(self) -> 'DataSplitterConfig':
        """Validate date ranges when split_type=DATE."""
        if self.split_type == SplitType.DATE:
            if self.train_end_date and self.train_start_date and self.train_end_date <= self.train_start_date:
                raise ValueError("train_end_date must be after train_start_date")
            if self.test_end_date and self.test_start_date and self.test_end_date <= self.test_start_date:
                raise ValueError("test_end_date must be after test_start_date")
        return self
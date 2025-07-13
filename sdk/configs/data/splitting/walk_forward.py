from pydantic import BaseModel, Field
from .base import BaseConfig

class WalkForwardSplitConfig(BaseConfig):
    train_window: int = Field(..., gt=0)
    test_window: int = Field(..., gt=0)
    step_size: int = Field(..., gt=0)

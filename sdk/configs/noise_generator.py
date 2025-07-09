from pydantic import BaseModel, Field, model_validator
from typing import Dict
from .base import BaseConfig

class NoiseGeneratorConfig(BaseConfig):
    scale: Dict[str, float] = Field(default_factory=lambda: {
        "open": 0.001,
        "high": 0.001,
        "low": 0.001,
        "close": 0.001
    })
    gap_chance: float = Field(0.01, ge=0, le=1)
    gap_magnitude: Dict[str, float] = Field(default_factory=lambda: {
        "min": 0.005,
        "max": 0.02
    })

    @model_validator(mode='after')
    def validate_gap_magnitude(self) -> 'NoiseGeneratorConfig':
        if self.gap_magnitude['min'] > self.gap_magnitude['max']:
            raise ValueError('min gap_magnitude cannot be greater than max')
        return self

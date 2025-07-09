from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict
from .base import BaseConfig

class DataTransformerConfig(BaseConfig):
    user_mapping: Optional[Dict[str, str]] = None
    keyword_detection: bool = True

    @model_validator(mode='after')
    def validate_mapping_method(self) -> 'DataTransformerConfig':
        if self.user_mapping and self.keyword_detection:
            raise ValueError('Cannot have both user_mapping and keyword_detection enabled')
        if not self.user_mapping and not self.keyword_detection:
            raise ValueError('Must have either user_mapping or keyword_detection enabled')
        return self

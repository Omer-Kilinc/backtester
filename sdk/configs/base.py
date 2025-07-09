from pydantic import BaseModel

class BaseConfig(BaseModel):
    seed: int = 42

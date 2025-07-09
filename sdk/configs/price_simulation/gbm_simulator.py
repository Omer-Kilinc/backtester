from pydantic import Field, Literal
from typing import Optional
from ..base import BaseConfig

class GBMSimulatorConfig(BaseConfig):
    simulator_type: Literal['gbm'] = 'gbm'
    start_price: float
    mu: float
    sigma: float
    start_date: str
    end_date: str
    output_freq: str = "1D"
    precision_factor: int = 24

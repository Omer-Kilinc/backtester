from pydantic import Field, Literal
from typing import Optional
from ..base import BaseConfig

class OrnsteinUhlenbeckSimulatorConfig(BaseConfig):
    simulator_type: Literal['ou'] = 'ou'
    x0: float = 100.0
    mu: float = 100.0
    theta: float = 0.1
    sigma: float = 1.0
    start_date: str = "2020-01-01"
    end_date: str = "2020-12-31"
    output_freq: str = "1D"
    precision_factor: int = 24

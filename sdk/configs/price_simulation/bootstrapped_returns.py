from pydantic import Field, Literal
from typing import Optional
import pandas as pd
from ..base import BaseConfig

class BootstrapReturnSimulatorConfig(BaseConfig):
    simulator_type: Literal['bootstrap'] = 'bootstrap'
    cleaned_price_df: pd.DataFrame
    start_date: str
    end_date: str
    output_freq: str = "1D"
    use_path_simulation: bool = True
    inject_gaps: bool = False
    gap_volatility: float = 0.01

    class Config:
        arbitrary_types_allowed = True

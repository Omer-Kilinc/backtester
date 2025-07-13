from pydantic import BaseModel, Field, model_validator
from typing import Literal, Dict, Any
from sdk.strategy.base import StrategyConfig
from sdk.data.data_pipeline import DataPipelineConfig

class BacktesterConfig(BaseModel):
    """
    Configuration for the backtester.

    Attributes:
        strategy (StrategyConfig): Configuration for the strategy to be backtested.
        data_pipeline (DataPipelineConfig): Configuration for the data pipeline.
        commission_model (Literal['fixed', 'percentage']): Commission model to use. Defaults to 'fixed'.
        commission_rate (float): Commission rate to use. Defaults to 0.001.
        slippage_model (float): Slippage model to use. Defaults to 0.001.
        leverage (float): Leverage to use. Defaults to 1.0.
        margin_requirement (float): Margin requirement to use. Defaults to 1.0.
        allow_shorting (bool): Whether to allow shorting. Defaults to True.
        allow_short_selling (bool): Whether to allow short selling. Defaults to True.
        parameters (Dict[str, Any]): Additional parameters to pass to the strategy. Defaults to {}.
    """
    strategy: StrategyConfig
    # TODO: Decide whether to keep other processes like Data processing within the backtester or to keep the backtester seperate and combine them in a seperate function.
    data_pipeline: DataPipelineConfig
    commission_model: Literal['fixed', 'percentage'] = 'fixed'
    commission_rate: float = 0.001
    slippage_model: float = 0.001
    leverage: float = 1.0
    margin_requirement: float = 1.0
    allow_shorting: bool = True
    allow_short_selling: bool = True
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    def validate_commission_model(cls, values):
        if values.get("commission_model") == "percentage" and values.get("commission_rate") < 0:
            raise ValueError("Commission rate must be non-negative for percentage commission model")
        return values

    class Config:
        extra = "forbid"
    
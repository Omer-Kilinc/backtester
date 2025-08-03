from pydantic import BaseModel, Field, model_validator
from typing import Literal, Dict, Any

class BacktesterConfig(BaseModel):
    """
    Configuration for the backtester.

    Attributes:
        # Capital and margin settings
        initial_capital (float): Starting capital for the backtest.
        margin_requirement_rate (float): Default margin requirement rate (e.g., 0.5 for 50%).
        
        # Commission and fees
        commission_model (Literal): Commission model - 'fixed' for per-trade fee, 'percentage' for % of trade value.
        commission_rate (float): Commission rate - dollar amount for 'fixed', percentage for 'percentage'.
        slippage_model (float): Slippage as percentage of trade price.
        
        # Trading rules
        allow_short_selling (bool): Whether short selling is allowed.
        require_margin_for_shorts (bool): Whether short selling requires a margin account.
        
        # Leverage and margin
        max_leverage (float): Maximum allowed leverage ratio.
        
        # Symbol settings (for single-symbol backtests)
        symbol (str): The symbol to trade.
        
        # Order handling
        failed_order_action (Literal): What to do with failed orders - 'remove', 'keep', or 'callback'.
        
        # Execution settings
        execution_model (Literal): When orders execute - 'next_bar_open' or 'same_bar_close'.
        
        # Reporting and logging
        progress_log_freq (float): Progress logging frequency (0-100%).
        
        # Strategy parameters
        parameters (Dict): Additional parameters to pass to the strategy.
    """
    
    # Capital and margin settings
    initial_capital: float = Field(100_000.0, gt=0, description="Starting capital for the backtest")
    margin_requirement_rate: float = Field(0.5, gt=0, le=1.0, description="Default margin requirement (50%)")
    
    # Commission and fees
    commission_model: Literal['fixed', 'percentage'] = Field('percentage', description="Commission model")
    commission_rate: float = Field(0.001, ge=0, description="Commission rate (0.1% default)")
    slippage_model: float = Field(0.001, ge=0, description="Slippage as percentage")
    
    # Trading rules
    allow_short_selling: bool = Field(True, description="Whether short selling is allowed")
    require_margin_for_shorts: bool = Field(True, description="Whether shorts require margin account")
    
    # Leverage and margin
    max_leverage: float = Field(10.0, gt=1.0, le=100.0, description="Maximum allowed leverage")
    
    # Symbol settings
    symbol: str = Field("AAPL", description="Symbol for single-symbol backtests")
    
    # Order handling
    failed_order_action: Literal['remove', 'keep', 'callback'] = Field('remove', description="Action for failed orders")
    
    # Execution settings
    execution_model: Literal['next_bar_open', 'same_bar_close'] = Field('next_bar_open', description="Order execution timing")
    
    # Reporting and logging
    progress_log_freq: float = Field(10.0, ge=0, le=100, description="Progress logging frequency")
    
    # Strategy parameters
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Strategy parameters")

    @model_validator(mode="before")
    @classmethod
    def validate_commission_model(cls, values):
        """Validate commission settings"""
        commission_model = values.get("commission_model")
        commission_rate = values.get("commission_rate", 0)
        
        if commission_model == "percentage" and commission_rate < 0:
            raise ValueError("Commission rate must be non-negative for percentage commission model")
        
        if commission_model == "fixed" and commission_rate < 0:
            raise ValueError("Fixed commission must be non-negative")
            
        return values

    @model_validator(mode="after")
    def validate_short_selling_logic(self) -> "BacktesterConfig":
        """Validate short selling configuration"""
        if not self.allow_short_selling and self.require_margin_for_shorts:
            pass
        
        return self

    class Config:
        extra = "forbid"  # Prevent extra fields
        
    def calculate_commission(self, trade_value: float) -> float:
        """Calculate commission for a trade"""
        if self.commission_model == 'fixed':
            return self.commission_rate
        else:  # percentage
            return trade_value * self.commission_rate
    
    def calculate_slippage(self, price: float) -> float:
        """Calculate slippage for a trade"""
        return price * self.slippage_model
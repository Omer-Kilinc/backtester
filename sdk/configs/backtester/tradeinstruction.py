from datetime import datetime
from typing import Optional, Callable, Union, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
from pydantic import model_validator, field_validator
import pandas as pd

class OrderDirection(Enum):
    BUY = 'buy'
    SELL = 'sell'

class OrderType(Enum):
    MARKET = 'market'
    LIMIT = 'limit'
    STOP = 'stop'
    STOP_LIMIT = 'stop_limit'

class TimeInForce(Enum):
    GTC = 'GTC'
    DAY = 'DAY'
    GTD = 'GTD'
    FOK = 'FOK'
    IOC = 'IOC'

class FailureReason(Enum):
    InsufficientFunds = 'InsufficientFunds'
    InsufficientMargin = 'InsufficientMargin'
    GoodTillDateExpired = 'GoodTillDateExpired'
    

class TradeInstruction(BaseModel):
    """
    TradeInstruction is a class that represents a trade instruction.

    Attributes:
        direction (OrderDirection): The direction of the trade (buy or sell).
        order_type (OrderType): The type of the order (market, limit, stop, stop_limit).
        quantity (Optional[float]): The quantity of the asset to trade. Mutually exclusive with amount.
        amount (Optional[float]): The amount of the asset to trade. Mutually exclusive with quantity.
        price (Optional[float]): The price at which to execute the order. Required for LIMIT or STOP_LIMIT orders.
        stop_price (Optional[float]): The stop price for the order. Required for STOP or STOP_LIMIT orders.
        time_in_force (Optional[TimeInForce]): The time in force for the order. Defaults to GTC.
        good_till_date (Optional[int]): The good till date for the order. Defaults to None. Candlestick count.
        target_price (Optional[float]): The target price for the order.
        entry_stop_loss (Optional[float]): The entry stop loss for the order (absolute price).
        entry_trailing_stop (Optional[float]): The entry trailing stop for the order (absolute price).
        take_profit (Optional[Union[float, str]]): Absolute price (100) or percentage ('5%') for take profit.
        exit_stop_loss (Optional[Union[float, str]]): Absolute price (90) or percentage ('5%') for stop loss after entry.
        exit_trailing_stop (Optional[Union[float, str]]): Percentage ('5%') or absolute price for trailing stop after entry.
        exit_condition (Optional[Callable[[pd.DataFrame, float], bool]]): The exit condition for the order. Takes in all the current market data and current position value, returns boolean indicating whether the order should be closed.
        on_fail (Optional[Callable[[pd.DataFrame, TradeInstruction, FailureReason], None]]): The on fail callback for the order.
        metadata (Optional[Dict[str, Any]]): Metadata for the order.
    """
    direction: OrderDirection
    order_type: OrderType
    quantity: Optional[float] = None
    amount: Optional[float] = None  
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: Optional[TimeInForce] = TimeInForce.GTC
    good_till_date: Optional[int] = None  # Candlestick count # TODO: Change to good_till_candlestick_count?
    target_price: Optional[float] = None
    
    # Entry-level stops (used during order placement)
    entry_stop_loss: Optional[float] = None
    entry_trailing_stop: Optional[float] = None
    
    # Exit-level stops (used after position is established)
    take_profit: Optional[Union[float, str]] = Field(
        None,
        description="Absolute price (100) or percentage ('5%') for take profit"
    )
    exit_stop_loss: Optional[Union[float, str]] = Field(
        None,
        description="Absolute price (90) or percentage ('5%') for stop loss after entry"
    )
    exit_trailing_stop: Optional[Union[float, str]] = Field(
        None,
        description="Percentage ('5%') or absolute price for trailing stop after entry"
    )
    
    exit_condition: Optional[Callable[[pd.DataFrame, float], bool]] = None
    on_fail: Optional[Callable[[pd.DataFrame, "TradeInstruction", FailureReason], None]] = None
    metadata: Optional[Dict[str, Any]] = None

    is_margin: bool = Field(
        False,
        description="Whether this trade uses margin"
    )
    leverage: Optional[float] = Field(
        None,
        description="Leverage ratio (e.g., 2 for 2:1)",
        gt=1.0, le=100.0  
    )
    margin_requirement: Optional[float] = Field(
        None,
        description="Custom margin requirement (overrides default)",
        gt=0.0, le=1.0
    )



    # Validators
    @field_validator("take_profit", "exit_stop_loss", "exit_trailing_stop")
    @classmethod
    def validate_price_or_percent(cls, v):
        if v is None:
            return v
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str) and v.endswith("%"):
            # Keep as string for now, will be processed during execution
            try:
                float(v.rstrip("%"))
                return v
            except ValueError:
                raise ValueError(f"Invalid percentage format: {v}")
        raise ValueError("Value must be a number or percentage string (e.g., '5%')")

    @field_validator("quantity", "amount", "price", "stop_price", "target_price", "entry_stop_loss", "entry_trailing_stop")
    @classmethod
    def must_be_positive(cls, v, info):
        if v is not None and v <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return v

    @model_validator(mode="after")
    def validate_trade_instruction(self) -> "TradeInstruction":
        # Enforce mutually exclusive amount and quantity
        if (self.amount is None and self.quantity is None) or (self.amount is not None and self.quantity is not None):
            raise ValueError("Specify exactly one of 'amount' or 'quantity'")

        # Require price for LIMIT or STOP_LIMIT orders
        if self.order_type in {OrderType.LIMIT, OrderType.STOP_LIMIT} and self.price is None:
            raise ValueError(f"'price' is required for order_type '{self.order_type.value}'")

        # Require stop_price for STOP or STOP_LIMIT orders
        if self.order_type in {OrderType.STOP, OrderType.STOP_LIMIT} and self.stop_price is None:
            raise ValueError(f"'stop_price' is required for order_type '{self.order_type.value}'")

        # Validate GTD requires good_till_date
        if self.time_in_force == TimeInForce.GTD and self.good_till_date is None:
            raise ValueError("'good_till_date' is required when time_in_force is 'GTD'")

        return self

    @model_validator(mode="after")
    def validate_exit_conditions(self) -> "TradeInstruction":
        """Validate exit conditions only when we have a reference price"""
        # Only validate exit conditions for limit orders where we know the entry price
        if self.price is None:
            return self  # Skip validation for market orders
            
        def parse_value(value, reference_price):
            """Helper to parse percentage or absolute values"""
            if isinstance(value, str) and value.endswith("%"):
                percent = float(value.rstrip("%")) / 100
                return reference_price * (1 + percent) if self.direction == OrderDirection.BUY else reference_price * (1 - percent)
            return float(value)
        
        # Validate take profit logic
        if self.take_profit is not None:
            tp_price = parse_value(self.take_profit, self.price)
            if self.direction == OrderDirection.BUY and tp_price <= self.price:
                raise ValueError("Take profit must be above entry price for BUY orders")
            elif self.direction == OrderDirection.SELL and tp_price >= self.price:
                raise ValueError("Take profit must be below entry price for SELL orders")
        
        # Validate exit stop loss logic
        if self.exit_stop_loss is not None:
            sl_price = parse_value(self.exit_stop_loss, self.price)
            if self.direction == OrderDirection.BUY and sl_price >= self.price:
                raise ValueError("Exit stop loss must be below entry price for BUY orders")
            elif self.direction == OrderDirection.SELL and sl_price <= self.price:
                raise ValueError("Exit stop loss must be above entry price for SELL orders")
            
        return self
    
    @model_validator(mode="after")
    def validate_margin_params(self) -> "TradeInstruction":
        if self.is_margin:
            # Set default leverage if not specified
            if self.leverage is None:
                self.leverage = 2.0  # Default 2:1 leverage
        
        elif self.leverage is not None:
            raise ValueError("Leverage specified for non-margin trade")
            
        return self
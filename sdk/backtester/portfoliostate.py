from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from pydantic import BaseModel
from configs.backtester.tradeinstruction import (
    OrderDirection, OrderType, TimeInForce, TradeInstruction, FailureReason
)

# class TradeStatus(Enum):
#     PENDING = "pending"
#     FILLED = "filled" 
#     PARTIAL_FILL = "partial_fill"
#     CANCELLED = "cancelled"
#     REJECTED = "rejected"


@dataclass
class Position:
    """Represents an active position in the portfolio"""
    symbol: str
    direction: OrderDirection
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    
    # Exit conditions
    take_profit: Optional[Union[float, str]] = None
    exit_stop_loss: Optional[Union[float, str]] = None
    exit_trailing_stop: Optional[Union[float, str]] = None
    exit_condition: Optional[Callable[[pd.DataFrame, float], bool]] = None
    
    # Margin info
    is_margin: bool = False
    leverage: Optional[float] = None
    margin_used: float = 0.0
    initial_margin_req: float = 0.5 
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = None
    
    
    
    # @property
    # def market_value(self) -> float:
    #     """Current market value of the position"""
    #     return self.quantity * self.current_price
    
    # @property
    # def unrealized_pnl(self) -> float:
    #     """Unrealized profit/loss"""
    #     if self.direction == OrderDirection.BUY:
    #         return (self.current_price - self.entry_price) * self.quantity
    #     else:  # SELL
    #         return (self.entry_price - self.current_price) * self.quantity
    
    # @property
    # def unrealized_pnl_percent(self) -> float:
    #     """Unrealized PnL as percentage"""
    #     entry_value = self.entry_price * self.quantity
    #     return (self.unrealized_pnl / entry_value) * 100 if entry_value != 0 else 0
    
    # def update_price(self, new_price: float):
    #     """Update current price of the position"""
    #     self.current_price = new_price
    
@dataclass
class ExecutedTrade:
    """Simple record of a completed trade execution"""
    symbol: str
    direction: OrderDirection
    quantity: float
    price: float
    timestamp: datetime
    is_entry: bool  # True for opening position, False for closing
    realized_pnl: Optional[float] = None  # Only for exit trades
    commission: float = 0.0
    fees: float = 0.0
    
# TODO: Change as dataclass or stick to pydantic?
class PortfolioState(BaseModel):
    """Main portfolio state management"""
    
    # Core balances
    cash: float = Field(100_000.0, description="Available cash balance")
    initial_capital: float = Field(100_000.0, description="Starting capital")
    margin_requirement_rate: float = Field(0.5, description="Default margin requirement (50%)")
    
    # Positions and orders 
    open_positions: Dict[str, Position] = Field(default_factory=dict, description="Active positions by symbol")
    pending_orders: Dict[str, TradeInstruction] = Field(default_factory=dict, description="Pending orders by order_id")
    
    # Trade history
    executed_trades: List[ExecutedTrade] = Field(default_factory=list, description="All completed trade executions")
    closed_positions_history: List[Position] = Field(default_factory=list, description="All closed positions")
    
    class Config:
        arbitrary_types_allowed = True

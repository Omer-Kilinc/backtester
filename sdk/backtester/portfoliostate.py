from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from pydantic import BaseModel, Field
from sdk.configs.backtester.tradeinstruction import (  
    OrderDirection, OrderType, TimeInForce, TradeInstruction, FailureReason
)
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class Position:
    """Represents an active position in the portfolio"""
    symbol: str
    direction: OrderDirection
    quantity: float
    entry_price: float
    entry_time: datetime
    
    # Exit conditions
    take_profit: Optional[Union[float, str]] = None
    exit_stop_loss: Optional[Union[float, str]] = None
    exit_trailing_stop: Optional[Union[float, str]] = None
    exit_condition: Optional[Callable[[pd.DataFrame, float], bool]] = None
    
    # Margin info
    is_margin: bool = False
    leverage: Optional[float] = None
    initial_margin_req: float = 0.5
    initial_margin_locked: Optional[float] = None 
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = None
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized profit/loss at given price"""
        if self.direction == OrderDirection.BUY:
            return (current_price - self.entry_price) * self.quantity
        else:  # SELL
            return (self.entry_price - current_price) * self.quantity
    
    def get_market_value(self, current_price: float) -> float:
        """Get current market value of the position"""
        return self.quantity * current_price
    
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
    position_id: Optional[str] = None  # Link to position ID
    
class PortfolioState(BaseModel):
    """Main portfolio state management"""
    
    # Core balances
    cash: float = Field(100_000.0, description="Available cash balance")
    initial_capital: float = Field(100_000.0, description="Starting capital")
    margin_requirement_rate: float = Field(0.5, description="Default margin requirement (50%)")
    used_initial_margin: float = Field(0.0, description="Total margin currently locked for open positions")
    
    # Positions and orders
    open_positions: Dict[str, Position] = Field(default_factory=dict, description="Active positions by position_id")
    pending_orders: Dict[str, TradeInstruction] = Field(default_factory=dict, description="Pending orders by order_id")
    
    # Trade history
    executed_trades: List[ExecutedTrade] = Field(default_factory=list, description="All completed trade executions")
    closed_positions_history: List[Position] = Field(default_factory=list, description="All closed positions")
    
    # Counters for unique IDs
    position_counter: int = Field(0, description="Counter for generating unique position IDs")
    order_counter: int = Field(0, description="Counter for generating unique order IDs")
    
    class Config:
        arbitrary_types_allowed = True

    @property
    def free_cash(self) -> float:
        """Calculate free cash (cash - locked margin)"""
        return self.cash - self.used_initial_margin
    
    def lock_margin(self, amount: float):
        """Lock margin for a position"""
        if self.free_cash < amount:
            raise ValueError(f"Insufficient free cash to lock margin. Free: {self.free_cash:.2f}, Requested: {amount:.2f}")
        self.used_initial_margin += amount

    def unlock_margin(self, amount: float):
        """Unlock margin from a closed position"""
        self.used_initial_margin -= amount
        if self.used_initial_margin < 0:
            self.used_initial_margin = 0  # Safety check

    def liquidate_position(self, position_id: str, exit_price: float, reason: str, timestamp: datetime) -> bool:
        """
        Liquidate (force close) a position at market price.
        
        Args:
            position_id: The position ID to liquidate
            exit_price: Price to liquidate at (current market price)
            reason: Reason for liquidation (e.g., "Margin call", "Stop loss")
            timestamp: When the liquidation occurred
            
        Returns:
            bool: True if liquidation successful, False if position not found
        """
        if position_id not in self.open_positions:
            logger.error(f"Cannot liquidate {position_id} - position not found")
            return False
        
        position = self.open_positions[position_id]
        
        # Calculate realized P&L
        realized_pnl = position.get_unrealized_pnl(exit_price)
        
        # Update cash based on position direction
        if position.direction == OrderDirection.BUY:
            # Selling shares - add proceeds to cash
            self.cash += position.quantity * exit_price
        else:  # SHORT position
            # Covering short - subtract cost from cash  
            self.cash -= position.quantity * exit_price
        
        # Unlock margin that was locked for this position
        if position.is_margin and position.initial_margin_locked:
            self.unlock_margin(position.initial_margin_locked)
        
        # Create trade record
        trade = ExecutedTrade(
            symbol=position.symbol,
            direction=OrderDirection.SELL if position.direction == OrderDirection.BUY else OrderDirection.BUY,
            quantity=position.quantity,
            price=exit_price,
            timestamp=timestamp,
            is_entry=False,
            realized_pnl=realized_pnl,
            position_id=position_id
        )
        
        # Update records
        self.executed_trades.append(trade)
        self.closed_positions_history.append(position)
        del self.open_positions[position_id]
        
        logger.info(f"Liquidated {position.symbol} (ID: {position_id}) at {exit_price:.2f} due to {reason}. P&L: {realized_pnl:.2f}")
        return True
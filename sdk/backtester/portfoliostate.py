from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
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
    exit_condition: Optional[Callable[[pd.DataFrame, "Position", "PortfolioState"], Union[bool, float, dict, List[dict]]]] = None
    
    # Enhanced exit condition control
    original_quantity: float = field(init=False)  # Track original position size
    exit_condition_frequency: int = 1  # Check exit condition every N bars (1=every bar)
    last_exit_check_bar: int = -1  # Track when exit condition was last checked
    partial_closes: List[dict] = field(default_factory=list)  # History of partial closes
    
    # Margin info
    is_margin: bool = False
    leverage: Optional[float] = None
    initial_margin_req: float = 0.5
    initial_margin_locked: Optional[float] = None 
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set original_quantity after initialization"""
        self.original_quantity = self.quantity
    
    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized profit/loss at given price"""
        if self.direction == OrderDirection.BUY:
            return (current_price - self.entry_price) * self.quantity
        else:  # SELL
            return (self.entry_price - current_price) * self.quantity
    
    def get_market_value(self, current_price: float) -> float:
        """Get current market value of the position"""
        return self.quantity * current_price
    
    def get_remaining_percentage(self) -> float:
        """Get percentage of original position still remaining"""
        if self.original_quantity == 0:
            return 0.0
        return self.quantity / self.original_quantity
    
    def add_partial_close(self, percentage: float, price: float, reason: str, timestamp: datetime):
        """Record a partial close"""
        self.partial_closes.append({
            'timestamp': timestamp,
            'percentage_closed': percentage,
            'price': price,
            'reason': reason,
            'remaining_quantity': self.quantity
        })
    
    def should_check_exit_condition(self, current_bar_index: int) -> bool:
        """Check if exit condition should be evaluated this bar"""
        if self.exit_condition is None:
            return False
        return (current_bar_index - self.last_exit_check_bar) >= self.exit_condition_frequency

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
    is_partial_close: bool = False  # True if this was a partial close
    
class PortfolioState(BaseModel):
    """Main portfolio state management with enhanced position tracking"""
    
    # Core balances
    cash: float = Field(100_000.0, description="Available cash balance")
    initial_capital: float = Field(100_000.0, description="Starting capital")
    margin_requirement_rate: float = Field(0.5, description="Default margin requirement (50%)")
    used_initial_margin: float = Field(0.0, description="Total margin currently locked for open positions")
    short_sale_proceeds: float = Field(0.0, description="Locked cash collateral from short sales")
    
    # Positions and orders
    open_positions: Dict[str, Position] = Field(default_factory=dict, description="Active positions by position_id")
    pending_orders: Dict[str, TradeInstruction] = Field(default_factory=dict, description="Pending orders by order_id")
    
    # Trade history
    executed_trades: List[ExecutedTrade] = Field(default_factory=list, description="All completed trade executions")
    closed_positions_history: List[Position] = Field(default_factory=list, description="All closed positions")
    
    # Counters for unique IDs
    position_counter: int = Field(0, description="Counter for generating unique position IDs")
    order_counter: int = Field(0, description="Counter for generating unique order IDs")

    order_placement_bars: Dict[str, int] = Field(default_factory=dict, description="Track when orders were placed (bar index)")
    
    class Config:
        arbitrary_types_allowed = True

    @property
    def total_collateral(self) -> float:
        """Total collateral available for margin requirements (cash + short proceeds)"""
        return self.cash + self.short_sale_proceeds
    
    @property
    def free_cash(self) -> float:
        """Cash available for new trades (total collateral - locked margin)"""
        return self.total_collateral - self.used_initial_margin
    
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
    
    def lock_short_proceeds(self, amount: float):
        """Lock proceeds from short sale as collateral"""
        self.short_sale_proceeds += amount
    
    def unlock_short_proceeds(self, amount: float):
        """Unlock short sale proceeds when position is closed"""
        self.short_sale_proceeds -= amount
        if self.short_sale_proceeds < 0:
            self.short_sale_proceeds = 0  # Safety check

    def execute_buy(self, symbol: str, quantity: float, price: float, timestamp: datetime,
                    is_margin: bool = False, leverage: float = None, margin_requirement: float = None,
                    take_profit=None, exit_stop_loss=None, exit_trailing_stop=None, 
                    exit_condition=None, exit_condition_frequency: int = 1, metadata=None, 
                    commission_rate: float = 0.001, slippage_rate: float = 0.001) -> str:
        """
        Execute a buy order and create a new position.
        """
        # TODO: Make slippage configurable with mean/variance instead of fixed rate
        execution_price = price * (1 + slippage_rate)  # Buy higher due to slippage
        
        # 1. Calculate costs
        trade_cost = quantity * execution_price
        commission = trade_cost * commission_rate if commission_rate else 0.0
        
        # 2. Validate commission can be paid from cash (before locking margin)
        if commission > self.cash:
            raise ValueError(f"Insufficient cash for commission: need {commission:.2f}, have {self.cash:.2f}")
        
        # 3. Calculate margin requirements
        if is_margin:
            margin_req_rate = margin_requirement or self.margin_requirement_rate
            margin_needed = trade_cost * margin_req_rate
        else:
            margin_needed = 0.0
            total_cash_needed = trade_cost + commission  # Full amount + commission
            if total_cash_needed > self.cash:
                raise ValueError(f"Insufficient cash: need {total_cash_needed:.2f}, have {self.cash:.2f}")
        
        # 4. Lock funds (validation should have already passed in backtester)
        if is_margin:
            self.lock_margin(margin_needed)
            self.cash -= commission
        else:
            self.cash -= total_cash_needed
        
        # 5. Create position
        position_id = f"pos_{self.position_counter}"
        self.position_counter += 1
        
        position = Position(
            symbol=symbol,
            direction=OrderDirection.BUY,
            quantity=quantity,
            entry_price=execution_price,  # Use actual execution price with slippage
            entry_time=timestamp,
            take_profit=take_profit,
            exit_stop_loss=exit_stop_loss,
            exit_trailing_stop=exit_trailing_stop,
            exit_condition=exit_condition,
            exit_condition_frequency=exit_condition_frequency,
            is_margin=is_margin,
            leverage=leverage,
            initial_margin_req=margin_req_rate if is_margin else 0.0,
            initial_margin_locked=margin_needed if is_margin else None,
            metadata=metadata
        )
        
        # 6. Record position and trade
        self.open_positions[position_id] = position
        
        trade = ExecutedTrade(
            symbol=symbol,
            direction=OrderDirection.BUY,
            quantity=quantity,
            price=execution_price,  # Record actual execution price
            timestamp=timestamp,
            is_entry=True,
            realized_pnl=None,
            commission=commission,
            fees=0.0,
            position_id=position_id
        )
        self.executed_trades.append(trade)
        
        logger.info(f"Executed BUY: {quantity} {symbol} at {execution_price:.2f} (slippage from {price:.2f}) → position {position_id}")
        return position_id

    def execute_sell(self, symbol: str, quantity: float, price: float, timestamp: datetime,
                     is_margin: bool = False, leverage: float = None, margin_requirement: float = None,
                     take_profit=None, exit_stop_loss=None, exit_trailing_stop=None,
                     exit_condition=None, exit_condition_frequency: int = 1, metadata=None, 
                     commission_rate: float = 0.001, slippage_rate: float = 0.001, 
                     allow_short_selling: bool = True, require_margin_for_shorts: bool = True) -> str:
        """
        Execute a sell order and create a new short position.
        """
        # TODO: Make slippage configurable with mean/variance instead of fixed rate
        execution_price = price * (1 - slippage_rate)  # Sell lower due to slippage
        
        # 1. Calculate trade details
        trade_value = quantity * execution_price  # Proceeds from short sale
        commission = trade_value * commission_rate if commission_rate else 0.0
        
        # 2. Validate commission can be paid from cash (before locking anything)
        if commission > self.cash:
            raise ValueError(f"Insufficient cash for commission: need {commission:.2f}, have {self.cash:.2f}")
        
        # 3. Calculate margin requirements
        if is_margin:
            margin_req_rate = margin_requirement or self.margin_requirement_rate
            margin_needed = trade_value * margin_req_rate  # Additional margin from cash
        else:
            margin_needed = 0.0
        
        # 4. Execute the short sale
        # Lock short sale proceeds as collateral (not spendable)
        self.lock_short_proceeds(trade_value)
        
        # Lock additional margin from cash if margin trade
        if is_margin:
            self.lock_margin(margin_needed)
        
        # Subtract commission from spendable cash
        self.cash -= commission
        
        # 5. Create short position with safe metadata handling
        position_id = f"pos_{self.position_counter}"
        self.position_counter += 1
        
        # Ensure metadata is a dict
        if metadata is None:
            metadata = {}
        elif not isinstance(metadata, dict):
            metadata = {'original_metadata': metadata}
        
        # Add short proceeds tracking
        metadata['short_proceeds_locked'] = trade_value
        
        position = Position(
            symbol=symbol,
            direction=OrderDirection.SELL,  # Short position
            quantity=quantity,
            entry_price=execution_price,
            entry_time=timestamp,
            take_profit=take_profit,
            exit_stop_loss=exit_stop_loss,
            exit_trailing_stop=exit_trailing_stop,
            exit_condition=exit_condition,
            exit_condition_frequency=exit_condition_frequency,
            is_margin=is_margin,
            leverage=leverage,
            initial_margin_req=margin_req_rate if is_margin else 0.0,
            initial_margin_locked=margin_needed if is_margin else None,
            metadata=metadata
        )
        
        # 6. Record position and trade
        self.open_positions[position_id] = position
        
        trade = ExecutedTrade(
            symbol=symbol,
            direction=OrderDirection.SELL,
            quantity=quantity,
            price=execution_price,
            timestamp=timestamp,
            is_entry=True,
            realized_pnl=None,
            commission=commission,
            fees=0.0,
            position_id=position_id
        )
        self.executed_trades.append(trade)
        
        logger.info(f"Executed SELL (short): {quantity} {symbol} at {execution_price:.2f} → position {position_id}")
        logger.info(f"Locked {trade_value:.2f} short proceeds, {margin_needed:.2f} additional margin")
        return position_id

    def close_position_partially(self, position_id: str, percentage: float, exit_price: float, 
                                reason: str, timestamp: datetime, commission_rate: float = 0.001) -> bool:
        """
        Close a portion of a position.
        
        Args:
            position_id: Position to partially close
            percentage: Percentage to close (0.0 to 1.0)
            exit_price: Exit price
            reason: Reason for partial close
            timestamp: Close timestamp
            commission_rate: Commission rate
            
        Returns:
            bool: True if successful
        """
        if position_id not in self.open_positions:
            logger.error(f"Cannot partially close {position_id} - position not found")
            return False
        
        if not 0 < percentage <= 1.0:
            logger.error(f"Invalid percentage {percentage} - must be between 0 and 1")
            return False
        
        position = self.open_positions[position_id]
        close_quantity = position.quantity * percentage
        
        # Calculate commission and P&L
        trade_value = close_quantity * exit_price
        commission = trade_value * commission_rate
        realized_pnl = close_quantity * (exit_price - position.entry_price)
        if position.direction == OrderDirection.SELL:
            realized_pnl = -realized_pnl  # Invert for short positions
        
        # Update cash based on position direction
        if position.direction == OrderDirection.BUY:
            # Closing long position: sell shares, add proceeds minus commission
            net_proceeds = trade_value - commission
            self.cash += net_proceeds
        else:  # OrderDirection.SELL (short position)
            # Closing short position: buy back shares plus commission
            buyback_cost = trade_value + commission
            self.cash -= buyback_cost
            
            # Unlock proportional short sale proceeds
            if position.metadata and isinstance(position.metadata, dict):
                total_short_proceeds = position.metadata.get('short_proceeds_locked', 0.0)
                unlock_amount = total_short_proceeds * percentage
                self.unlock_short_proceeds(unlock_amount)
                position.metadata['short_proceeds_locked'] = total_short_proceeds - unlock_amount
        
        # Unlock proportional margin
        if position.is_margin and position.initial_margin_locked:
            unlock_margin = position.initial_margin_locked * percentage
            self.unlock_margin(unlock_margin)
            position.initial_margin_locked -= unlock_margin
        
        # Update position quantity
        position.quantity -= close_quantity
        
        # Record the partial close
        position.add_partial_close(percentage, exit_price, reason, timestamp)
        
        # Create trade record
        trade = ExecutedTrade(
            symbol=position.symbol,
            direction=OrderDirection.SELL if position.direction == OrderDirection.BUY else OrderDirection.BUY,
            quantity=close_quantity,
            price=exit_price,
            timestamp=timestamp,
            is_entry=False,
            realized_pnl=realized_pnl - commission,
            commission=commission,
            fees=0.0,
            position_id=position_id,
            is_partial_close=True
        )
        self.executed_trades.append(trade)
        
        logger.info(f"Partially closed {percentage*100:.1f}% of {position.symbol} (ID: {position_id}) "
                   f"at {exit_price:.2f}. P&L: {realized_pnl:.2f}, Commission: {commission:.2f}")
        
        # If position is fully closed, move to history
        if position.quantity <= 0.001:  # Account for floating point precision
            self.closed_positions_history.append(position)
            del self.open_positions[position_id]
            logger.info(f"Position {position_id} fully closed")
        
        return True

    def validate_trade(self, direction: OrderDirection, quantity: float, price: float, 
                      is_margin: bool = False, leverage: float = None, margin_requirement: float = None,
                      commission_rate: float = 0.001, allow_short_selling: bool = True, 
                      require_margin_for_shorts: bool = True, max_leverage: float = 10.0) -> tuple[bool, str]:
        """
        Validate if a trade can be executed without actually executing it.
        
        Args:
            direction: BUY or SELL
            quantity: Number of shares
            price: Price per share
            is_margin: Whether this is a margin trade
            leverage: Leverage ratio
            margin_requirement: Custom margin requirement
            commission_rate: Commission rate
            allow_short_selling: Whether short selling allowed
            require_margin_for_shorts: Whether shorts need margin
            max_leverage: Maximum allowed leverage
            
        Returns:
            tuple[bool, str]: (is_valid, error_message)
        """
        try:
            # 1. Basic validation
            if quantity <= 0:
                return False, "Quantity must be positive"
            if price <= 0:
                return False, "Price must be positive"
            
            # 2. Leverage validation
            if leverage and leverage > max_leverage:
                return False, f"Leverage {leverage} exceeds maximum {max_leverage}"
            
            # 3. Short selling validation
            if direction == OrderDirection.SELL:
                if not allow_short_selling:
                    return False, "Short selling is not allowed"
                if require_margin_for_shorts and not is_margin:
                    return False, "Short selling requires a margin account"
            
            # 4. Calculate costs and requirements
            trade_cost = quantity * price
            commission = trade_cost * commission_rate if commission_rate else 0.0
            
            if is_margin:
                margin_req_rate = margin_requirement or self.margin_requirement_rate
                margin_needed = trade_cost * margin_req_rate
                
                if direction == OrderDirection.BUY:
                    # Buy with margin: need margin for position + cash for commission
                    if margin_needed > self.free_cash:
                        return False, f"Insufficient margin: need {margin_needed:.2f}, have {self.free_cash:.2f}"
                    if commission > self.cash:
                        return False, f"Insufficient cash for commission: need {commission:.2f}, have {self.cash:.2f}"
                else:
                    # Short with margin: need margin for short position
                    if margin_needed > self.free_cash:
                        return False, f"Insufficient margin for short: need {margin_needed:.2f}, have {self.free_cash:.2f}"
                    if commission > self.cash:
                        return False, f"Insufficient cash for commission: need {commission:.2f}, have {self.cash:.2f}"
            else:
                # Cash trade
                if direction == OrderDirection.BUY:
                    total_needed = trade_cost + commission
                    if total_needed > self.cash:
                        return False, f"Insufficient cash: need {total_needed:.2f}, have {self.cash:.2f}"
                else:
                    # Cash short (if allowed) - just need commission
                    if commission > self.cash:
                        return False, f"Insufficient cash for commission: need {commission:.2f}, have {self.cash:.2f}"
            
            return True, "Trade validation passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def liquidate_position(self, position_id: str, exit_price: float, reason: str, timestamp: datetime) -> bool:
        """
        Liquidate (force close) a position at market price.
        """
        if position_id not in self.open_positions:
            logger.error(f"Cannot liquidate {position_id} - position not found")
            return False
        
        return self.close_position_partially(position_id, 1.0, exit_price, reason, timestamp)
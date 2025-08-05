from typing import Optional, Union, List
import pandas as pd
from logging import getLogger

from sdk.strategy.base import Strategy
from sdk.configs.backtester.tradeinstruction import TradeInstruction, OrderDirection, OrderType

logger = getLogger(__name__)


class BuyAndHoldStrategy(Strategy):
    """
    Simple buy-and-hold benchmark strategy
    
    Buys the asset with all available capital at the first bar and holds until the end.
    Used for benchmarking strategy performance against a passive investment approach.
    """
    
    def __init__(self, symbol: str, initial_capital: float):
        """
        Initialize the buy-and-hold strategy
        
        Args:
            symbol: Symbol to buy and hold
            initial_capital: Initial capital available for investment
        """
        super().__init__()
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.has_bought = False
        
        logger.info(f"Initialized BuyAndHoldStrategy for {symbol} with ${initial_capital:,.2f}")
    
    def on_init(self):
        """Called before backtest starts"""
        logger.info(f"BuyAndHoldStrategy initialized - will buy {self.symbol} with full capital")
        self.has_bought = False
    
    def on_bar(self, data: pd.DataFrame) -> Optional[Union[TradeInstruction, List[TradeInstruction]]]:
        """
        Strategy logic: Buy with all capital on first bar, then hold
        
        Args:
            data: Market data up to current bar
            
        Returns:
            TradeInstruction to buy on first bar, None afterwards
        """
        # Only buy once on the first bar where we have data
        if not self.has_bought and len(data) > 0:
            self.has_bought = True
            
            current_price = data.iloc[-1]['close']
            logger.info(f"BuyAndHoldStrategy: Buying {self.symbol} at {current_price:.2f} with ${self.initial_capital:,.2f}")
            
            return TradeInstruction(
                symbol=self.symbol,
                direction=OrderDirection.BUY,
                order_type=OrderType.MARKET,
                amount=self.initial_capital,  # Use all available capital
                reason="Buy and hold benchmark entry"
            )
        
        # After buying, just hold (return None)
        return None
    
    def on_teardown(self):
        """Called after backtest ends"""
        logger.info("BuyAndHoldStrategy completed - held position until end")


class SellAndHoldStrategy(Strategy):
    """
    Sell-and-hold benchmark strategy (for comparison with short strategies)
    
    Sells the asset with all available capital at the first bar and holds the short position until the end.
    Used for benchmarking against a passive short investment approach.
    """
    
    def __init__(self, symbol: str, initial_capital: float):
        """
        Initialize the sell-and-hold strategy
        
        Args:
            symbol: Symbol to sell and hold short
            initial_capital: Initial capital available for investment
        """
        super().__init__()
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.has_sold = False
        
        logger.info(f"Initialized SellAndHoldStrategy for {symbol} with ${initial_capital:,.2f}")
    
    def on_init(self):
        """Called before backtest starts"""
        logger.info(f"SellAndHoldStrategy initialized - will short {self.symbol} with full capital")
        self.has_sold = False
    
    def on_bar(self, data: pd.DataFrame) -> Optional[Union[TradeInstruction, List[TradeInstruction]]]:
        """
        Strategy logic: Sell short with all capital on first bar, then hold
        
        Args:
            data: Market data up to current bar
            
        Returns:
            TradeInstruction to sell short on first bar, None afterwards
        """
        # Only sell once on the first bar where we have data
        if not self.has_sold and len(data) > 0:
            self.has_sold = True
            
            current_price = data.iloc[-1]['close']
            logger.info(f"SellAndHoldStrategy: Shorting {self.symbol} at {current_price:.2f} with ${self.initial_capital:,.2f}")
            
            return TradeInstruction(
                symbol=self.symbol,
                direction=OrderDirection.SELL,
                order_type=OrderType.MARKET,
                amount=self.initial_capital,  # Use all available capital
                reason="Sell and hold benchmark entry"
            )
        
        # After selling, just hold the short (return None)
        return None
    
    def on_teardown(self):
        """Called after backtest ends"""
        logger.info("SellAndHoldStrategy completed - held short position until end")
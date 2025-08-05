import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from logging import getLogger

from sdk.backtester.portfoliostate import PortfolioState
from sdk.configs.analytics.analytics import AnalyticsConfig

# TODO: Create an __init__.py?

logger = getLogger(__name__)


class RealtimeTracker:
    """Tracks per-bar analytics metrics during backtest execution"""
    
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.peak_portfolio_value = 0.0
        self.custom_metrics = {}
        
    def initialize_data_columns(self, data: pd.DataFrame):
        """Initialize analytics columns in the main DataFrame"""
        logger.info("Initializing analytics columns in DataFrame")
        
        # Core per-bar metrics
        data['portfolio_value'] = np.nan
        data['drawdown'] = np.nan
        data['drawdown_pct'] = np.nan
        
        # Optional rolling metrics
        if self.config.calculate_rolling_metrics:
            data['rolling_volatility'] = np.nan
            
        logger.info(f"Added analytics columns: portfolio_value, drawdown, drawdown_pct")
        
    def track_bar(self, portfoliostate: PortfolioState, current_bar_data: pd.DataFrame, bar_index: int):
        """Update analytics values for current bar"""
        try:
            current_prices = self._extract_current_prices(current_bar_data)
            portfolio_value = self.get_portfolio_value(portfoliostate, current_prices)

            current_bar_data.iloc[-1, current_bar_data.columns.get_loc('portfolio_value')] = portfolio_value

            drawdown_abs, drawdown_pct = self.update_drawdown(portfolio_value)
            current_bar_data.iloc[-1, current_bar_data.columns.get_loc('drawdown')] = drawdown_abs
            current_bar_data.iloc[-1, current_bar_data.columns.get_loc('drawdown_pct')] = drawdown_pct
            
            if self.config.calculate_rolling_metrics and bar_index >= self.config.rolling_window:
                rolling_vol = self._calculate_rolling_volatility(current_bar_data, bar_index)
                current_bar_data.iloc[-1, current_bar_data.columns.get_loc('rolling_volatility')] = rolling_vol
            
            self._process_custom_metrics(current_bar_data, portfoliostate, bar_index)
            
        except Exception as e:
            logger.error(f"Error tracking analytics for bar {bar_index}: {e}")
            
    def get_portfolio_value(self, portfoliostate: PortfolioState, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        # Base cash and proceeds
        portfolio_value = portfoliostate.cash + portfoliostate.short_sale_proceeds
        
        # Add market value of all open positions
        for position in portfoliostate.open_positions.values():
            if position.symbol in current_prices:
                market_value = position.get_market_value(current_prices[position.symbol])
                portfolio_value += market_value
                
        return portfolio_value
        
    def update_drawdown(self, current_portfolio_value: float) -> tuple[float, float]:
        """Update peak and calculate current drawdown"""
        # Update peak
        if current_portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = current_portfolio_value
            
        # Calculate drawdown
        if self.peak_portfolio_value > 0:
            drawdown_abs = self.peak_portfolio_value - current_portfolio_value
            drawdown_pct = (drawdown_abs / self.peak_portfolio_value) * 100
        else:
            drawdown_abs = 0.0
            drawdown_pct = 0.0
            
        return drawdown_abs, drawdown_pct
        
    def _extract_current_prices(self, current_bar_data: pd.DataFrame) -> Dict[str, float]:
        """Extract current prices from bar data for position valuation"""
        current_bar = current_bar_data.iloc[-1]
        return {'default_symbol': current_bar['close']}
        
    def _calculate_rolling_volatility(self, data: pd.DataFrame, current_index: int) -> float:
        """Calculate rolling volatility of portfolio returns"""
        if current_index < self.config.rolling_window:
            return np.nan
            
        # Get portfolio values for rolling window
        start_idx = max(0, current_index - self.config.rolling_window + 1)
        portfolio_values = data['portfolio_value'].iloc[start_idx:current_index + 1]
        
        # Calculate returns and volatility
        returns = portfolio_values.pct_change().dropna()
        if len(returns) > 1:
            return returns.std() * np.sqrt(252)  # Annualized volatility
        return np.nan
        
    def _process_custom_metrics(self, current_bar_data: pd.DataFrame, portfoliostate: PortfolioState, bar_index: int):
        """Process any registered custom per-bar metrics"""
        for name, metric_func in self.custom_metrics.items():
            try:
                if name not in current_bar_data.columns:
                    current_bar_data[name] = np.nan
                    
                value = metric_func(current_bar_data, portfoliostate, bar_index=bar_index)
                current_bar_data.iloc[-1, current_bar_data.columns.get_loc(name)] = value
            except Exception as e:
                logger.error(f"Error calculating custom metric '{name}': {e}")
                
    def register_custom_metric(self, name: str, metric_func):
        """Register a custom per-bar metric"""
        self.custom_metrics[name] = metric_func
        logger.info(f"Registered custom per-bar metric: {name}")
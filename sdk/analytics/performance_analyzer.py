import pandas as pd
from functools import wraps
from typing import Optional, Dict, Any, Callable
from logging import getLogger

from sdk.configs.analytics.analytics import AnalyticsConfig
from sdk.backtester.portfoliostate import PortfolioState
from .realtime_tracker import RealtimeTracker
from .summary_calculator import SummaryCalculator

logger = getLogger(__name__)


def analytics_metric(timing: str):
    """
    Decorator for registering custom analytics metrics
    
    Args:
        timing: 'per_bar' or 'end_of_backtest'
    
    Usage:
        @analytics_metric(timing='per_bar')
        def my_custom_metric(self, data: pd.DataFrame, portfoliostate: PortfolioState, **kwargs) -> float:
            return some_calculation
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Store metadata on the function
        wrapper._analytics_timing = timing
        wrapper._analytics_name = func.__name__
        wrapper._is_analytics_metric = True
        
        return wrapper
    return decorator


class PerformanceAnalyzer:
    """Main coordinator for all performance analytics"""
    
    def __init__(self, config: AnalyticsConfig, symbol: str, initial_capital: float):
        self.config = config
        self.symbol = symbol
        self.initial_capital = initial_capital
        
        # Initialize components
        self.realtime_tracker = RealtimeTracker(config)
        self.summary_calculator = SummaryCalculator(config, initial_capital)
        
        # Custom metrics storage
        self.end_of_backtest_metrics = {}
        
        logger.info(f"Performance analytics initialized for {symbol} with ${initial_capital:,.2f} capital")
        
    def initialize_data_columns(self, data: pd.DataFrame):
        """Initialize all analytics columns in the DataFrame"""
        self.realtime_tracker.initialize_data_columns(data)
        
    def track_bar(self, portfoliostate: PortfolioState, current_bar_data: pd.DataFrame, bar_index: int):
        """Track analytics for current bar - called after live indicators, before strategy"""
        self.realtime_tracker.track_bar(portfoliostate, current_bar_data, bar_index)
        
    def compute_final_metrics(self, data: pd.DataFrame, portfoliostate: PortfolioState) -> Dict[str, Any]:
        """Compute final summary metrics"""
        logger.info("Computing final performance metrics...")
        
        try:
            # Calculate core metrics
            final_metrics = self.summary_calculator.calculate_all_metrics(data, portfoliostate)
            
            # Add custom end-of-backtest metrics
            final_metrics.update(self._calculate_custom_final_metrics(data, portfoliostate))
            
            # Add metadata
            final_metrics['_metadata'] = {
                'symbol': self.symbol,
                'initial_capital': self.initial_capital,
                'risk_free_rate': self.config.risk_free_rate,
                'var_confidence_level': self.config.var_confidence_level,
                'calculation_timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"Calculated {len(final_metrics)} final metrics")
            return final_metrics
            
        except Exception as e:
            logger.error(f"Error computing final metrics: {e}")
            return {}
        
    def register_custom_metric(self, name: str, metric_func: Callable, timing: str):
        """Register custom metrics"""
        if timing == 'per_bar':
            self.realtime_tracker.register_custom_metric(name, metric_func)
        elif timing == 'end_of_backtest':
            self.end_of_backtest_metrics[name] = metric_func
            logger.info(f"Registered end-of-backtest custom metric: {name}")
        else:
            raise ValueError(f"Invalid timing '{timing}'. Must be 'per_bar' or 'end_of_backtest'")
    
    def register_strategy_metrics(self, strategy_instance):
        """
        Auto-discover and register metrics from strategy class using decorators
        
        Args:
            strategy_instance: The strategy instance to scan for decorated methods
        """
        logger.info("Scanning strategy for decorated analytics metrics...")
        
        for attr_name in dir(strategy_instance):
            attr = getattr(strategy_instance, attr_name)
            
            # Check if it's a decorated analytics metric
            if (callable(attr) and 
                hasattr(attr, '_is_analytics_metric') and 
                attr._is_analytics_metric):
                
                timing = attr._analytics_timing
                metric_name = attr._analytics_name
                
                logger.info(f"Found decorated metric: {metric_name} (timing: {timing})")
                
                if timing == 'per_bar':
                    self.realtime_tracker.register_custom_metric(metric_name, attr)
                elif timing == 'end_of_backtest':
                    self.end_of_backtest_metrics[metric_name] = attr
                    
        logger.info(f"Auto-registered {len(self.end_of_backtest_metrics)} end-of-backtest metrics")
    
    def _calculate_custom_final_metrics(self, data: pd.DataFrame, portfoliostate: PortfolioState) -> Dict[str, Any]:
        """Calculate custom end-of-backtest metrics"""
        custom_results = {}
        
        for name, metric_func in self.end_of_backtest_metrics.items():
            try:
                result = metric_func(data, portfoliostate)
                custom_results[name] = result
                logger.debug(f"Calculated custom metric '{name}': {result}")
            except Exception as e:
                logger.error(f"Error calculating custom metric '{name}': {e}")
                custom_results[name] = None
                
        return custom_results
    
    def get_performance_summary(self, final_metrics: Dict[str, Any]) -> str:
        """Generate a human-readable performance summary"""
        if not final_metrics:
            return "No performance metrics available"
        
        try:
            summary_lines = [
                "=" * 50,
                "PERFORMANCE SUMMARY",
                "=" * 50,
                f"Symbol: {final_metrics.get('_metadata', {}).get('symbol', 'Unknown')}",
                f"Initial Capital: ${self.initial_capital:,.2f}",
                "",
                "RETURNS:",
                f"  Total Return: {final_metrics.get('total_return_pct', 0):.2f}%",
                f"  Annualized Return: {final_metrics.get('annualized_return', 0):.2f}%",
                f"  Best Day: {final_metrics.get('best_day_return_pct', 0):.2f}%",
                f"  Worst Day: {final_metrics.get('worst_day_return_pct', 0):.2f}%",
                "",
                "RISK METRICS:",
                f"  Sharpe Ratio: {final_metrics.get('sharpe_ratio', 0):.3f}",
                f"  Sortino Ratio: {final_metrics.get('sortino_ratio', 0):.3f}",
                f"  Max Drawdown: {final_metrics.get('max_drawdown_pct', 0):.2f}%",
                f"  Volatility: {final_metrics.get('volatility_annualized', 0):.2f}%",
                "",
                "TRADE STATISTICS:",
                f"  Total Trades: {final_metrics.get('total_trades', 0)}",
                f"  Win Rate: {final_metrics.get('win_rate', 0):.2f}%",
                f"  Profit Factor: {final_metrics.get('profit_factor', 0):.3f}",
                f"  Expectancy: ${final_metrics.get('expectancy', 0):.2f}",
                "=" * 50
            ]
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return "Error generating performance summary"
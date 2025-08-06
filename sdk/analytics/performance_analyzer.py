import pandas as pd
from functools import wraps
from typing import Optional, Dict, Any, Callable
from logging import getLogger

from sdk.configs.analytics.analytics import AnalyticsConfig
from sdk.configs.backtester.backtester import BacktesterConfig
from sdk.backtester.portfoliostate import PortfolioState
from .realtime_tracker import RealtimeTracker
from .summary_calculator import SummaryCalculator
from .benchmark_strategy import BuyAndHoldStrategy, SellAndHoldStrategy

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
        
    def compute_final_metrics(self, data: pd.DataFrame, portfoliostate: PortfolioState, 
                             backtester_config: Optional[BacktesterConfig] = None) -> Dict[str, Any]:
        """Compute final summary metrics with optional benchmark comparison"""
        logger.info("Computing final performance metrics...")
        
        try:
            # Calculate core metrics for main strategy
            final_metrics = self.summary_calculator.calculate_all_metrics(data, portfoliostate)
            
            # Add custom end-of-backtest metrics
            final_metrics.update(self._calculate_custom_final_metrics(data, portfoliostate))
            
            # Run benchmark comparison if enabled and backtester config provided
            if self.config.enable_buy_hold_benchmark and backtester_config is not None:
                benchmark_metrics = self.run_benchmark_comparison(data, backtester_config)
                
                # Now calculate actual comparison metrics with both main and benchmark data
                comparison_metrics = self.calculate_excess_return_metrics(final_metrics, benchmark_metrics)
                
                # Add both benchmark and comparison metrics
                final_metrics.update(benchmark_metrics)
                final_metrics.update(comparison_metrics)
            
            # Add metadata
            final_metrics['_metadata'] = {
                'symbol': self.symbol,
                'initial_capital': self.initial_capital,
                'risk_free_rate': self.config.risk_free_rate,
                'var_confidence_level': self.config.var_confidence_level,
                'benchmark_enabled': self.config.enable_buy_hold_benchmark,
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
    
    def run_benchmark_comparison(self, original_data: pd.DataFrame, backtester_config: BacktesterConfig) -> Dict[str, Any]:
        """
        Run benchmark comparison using buy-and-hold strategy
        
        Args:
            original_data: Original market data used in main backtest
            backtester_config: Backtester configuration to replicate settings
            
        Returns:
            Dictionary with benchmark metrics (prefixed with 'benchmark_')
        """
        logger.info("Running buy-and-hold benchmark comparison...")
        
        try:
            # Import here to avoid circular imports
            from sdk.backtester.backtester import Backtester
            
            # Determine benchmark symbol (use configured or same as main strategy)
            benchmark_symbol = self.config.benchmark_symbol or self.symbol
            
            # Create benchmark strategy
            benchmark_strategy = BuyAndHoldStrategy(
                symbol=benchmark_symbol,
                initial_capital=self.initial_capital
            )
            
            # Create benchmark backtester with same settings but no analytics (avoid recursion)
            # Pass data directly and use None for data_pipeline_config since we're bypassing prepare_data()
            benchmark_backtester = Backtester(
                strategy=benchmark_strategy,
                data_pipeline_config=None,  # Not needed since we pass data directly
                config=backtester_config,
                data=original_data.copy(),  # Use same data - this bypasses prepare_data()
                analytics_config=None  # No analytics to avoid recursion
            )
            
            # Run benchmark backtest (prepare_data() will be skipped since data is provided)
            logger.info("Executing benchmark backtest...")
            benchmark_results = benchmark_backtester.execute_backtest()
            
            # Extract benchmark data and calculate metrics
            benchmark_data = benchmark_results['data']
            benchmark_portfoliostate = benchmark_results['portfoliostate']
            
            # Calculate benchmark metrics using same calculator
            benchmark_calculator = SummaryCalculator(self.config, self.initial_capital)
            benchmark_metrics = benchmark_calculator.calculate_all_metrics(benchmark_data, benchmark_portfoliostate)
            
            # Prefix benchmark metrics and return them
            prefixed_benchmark = {f"benchmark_{k}": v for k, v in benchmark_metrics.items() if not k.startswith('_')}
            
            logger.info(f"Benchmark comparison completed with {len(prefixed_benchmark)} metrics")
            return prefixed_benchmark
            
        except Exception as e:
            logger.error(f"Error running benchmark comparison: {e}")
            return {
                'benchmark_error': str(e),
                'benchmark_comparison_failed': True
            }
    
    def calculate_excess_return_metrics(self, main_metrics: Dict[str, Any], benchmark_metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate excess return metrics comparing main strategy to benchmark
        
        Args:
            main_metrics: Main strategy performance metrics
            benchmark_metrics: Benchmark strategy performance metrics (with 'benchmark_' prefix)
            
        Returns:
            Dictionary with excess return calculations
        """
        try:
            # Extract benchmark values (remove 'benchmark_' prefix for comparison)
            bench_total_return = benchmark_metrics.get('benchmark_total_return_pct', 0)
            bench_annualized_return = benchmark_metrics.get('benchmark_annualized_return', 0)
            bench_sharpe = benchmark_metrics.get('benchmark_sharpe_ratio', 0)
            
            # Calculate excess returns
            excess_total_return = main_metrics.get('total_return_pct', 0) - bench_total_return
            excess_annualized_return = main_metrics.get('annualized_return', 0) - bench_annualized_return
            
            # Risk-adjusted excess returns
            main_sharpe = main_metrics.get('sharpe_ratio', 0)
            sharpe_difference = main_sharpe - bench_sharpe
            
            return {
                'excess_total_return_pct': excess_total_return,
                'excess_annualized_return_pct': excess_annualized_return,
                'sharpe_ratio_difference': sharpe_difference,
                'outperformed_benchmark': excess_total_return > 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating excess return metrics: {e}")
            return {}
    
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
            ]
            
            # Add benchmark comparison if available
            if 'benchmark_total_return_pct' in final_metrics:
                summary_lines.extend([
                    "",
                    "BENCHMARK COMPARISON:",
                    f"  Benchmark Return: {final_metrics.get('benchmark_total_return_pct', 0):.2f}%",
                    f"  Excess Return: {final_metrics.get('excess_total_return_pct', 0):.2f}%",
                    f"  Outperformed: {final_metrics.get('outperformed_benchmark', False)}"
                ])
            
            summary_lines.append("=" * 50)
            return "\n".join(summary_lines)
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return "Error generating performance summary"
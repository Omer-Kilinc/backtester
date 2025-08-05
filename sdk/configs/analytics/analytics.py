from typing import Optional

class AnalyticsConfig:
    """Configuration for performance analytics"""
    
    # Basic settings
    risk_free_rate: float = 0.03              # Annual risk-free rate (3%)
    var_confidence_level: float = 0.95        # VaR confidence level (95%)
    enable_custom_metrics: bool = True        # Allow user-defined metrics
    
    # Benchmark settings
    enable_buy_hold_benchmark: bool = True    # Calculate buy-and-hold comparison
    benchmark_symbol: Optional[str] = None    # Use same symbol as strategy by default
    
    # Computation settings
    calculate_rolling_metrics: bool = True    # Rolling volatility, Sharpe, etc.
    rolling_window: int = 252                 # Rolling window (1 year for daily data)